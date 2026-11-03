import numpy as np
import time
import scipy.optimize

N = 400
DEADLINE_SECONDS = 115.0  # Safety margin against 120s limit

def _autoconv_lin(x):
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2 including k=0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _metric_offdiag(x):
    """Off-diagonal metric: max |c_k| for k!=0 / S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _objective_offdiag(x, beta=50.0):
    """Objective for off-diagonal metric optimization using soft-max."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max / (s * s)

def _generate_m_sequence(n, poly, init):
    """Generate an m-sequence of length n using a linear feedback shift register."""
    seq = []
    state = init
    for _ in range(n):
        seq.append(1.0 if (state & 1) == 0 else -1.0)
        feedback = 0
        for tap in poly:
            feedback ^= (state >> tap) & 1
        state = (state >> 1) | (feedback << (len(poly) - 1))
    return np.array(seq[:n])

def _refine_subgrad(x0, deadline, iters=20000, use_softmax=True, beta=50.0):
    """Projected subgradient minimization of off-diagonal max|c| s.t. sum(x)=1.
    
    The ratio max|c|/S^2 is scale-invariant, so we fix sum=1 and minimize
    max|c_k| for k != 0. The zero-lag c0 = sum(x^2) >= 1 by Cauchy-Schwarz.
    """
    n = N
    x = np.array(x0, dtype=float).copy()
    s = np.sum(x)
    if abs(s) < 1e-15:
        x += 1e-9
    else:
        x = x / s
    
    best = _metric(x)
    bestx = x.copy()
    lr = 5e-4
    no_improve = 0
    patience = 2000
    for it in range(iters):
        if time.time() > deadline:
            break
        c = _autoconv_lin(x)
        if use_softmax:
            abs_c = np.abs(c[1:])
            m = np.max(abs_c)
            if m < 1e-15:
                break
            weights = np.exp(beta * (abs_c - m))
            weights /= weights.sum()
            g = np.zeros(n)
            for k in range(1, n):
                wk = weights[k - 1]
                if wk < 1e-8:
                    continue
                sign_ck = np.sign(c[k])
                if k < n:
                    g[:n - k] += wk * sign_ck * x[k:]
                    g[k:] += wk * sign_ck * x[:n - k]
        else:
            abs_c = np.abs(c[1:])
            k = int(np.argmax(abs_c)) + 1
            ck = float(c[k])
            g = np.zeros(n)
            sign_ck = np.sign(ck)
            if k < n:
                g[:n - k] += sign_ck * x[k:]
                g[k:] += sign_ck * x[:n - k]
        
        x = x - lr * g
        x = x - (np.sum(x) - 1.0) / n
        x = np.clip(x, -2.0, 2.0)
        x = x - (np.sum(x) - 1.0) / n
        
        v = _metric(x)
        if v < best:
            best = v
            bestx = x.copy()
            no_improve = 0
            lr = min(lr * 1.03, 2e-3)
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.85
                no_improve = 0
                if lr < 1e-6:
                    break
    return bestx, best

def _refine_polish(x0, deadline, iters=30000):
    """High-precision polish: very small steps, tight patience."""
    n = N
    x = np.array(x0, dtype=float).copy()
    s = np.sum(x)
    if abs(s) < 1e-15:
        x += 1e-9
    else:
        x = x / s
        
    best = _metric(x)
    bestx = x.copy()
    lr = 1e-5
    no_improve = 0
    patience = 5000
    for it in range(iters):
        if time.time() > deadline:
            break
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        k = int(np.argmax(abs_c)) + 1
        ck = float(c[k])
        g = np.zeros(n)
        sign_ck = np.sign(ck)
        if k < n:
            g[:n - k] += sign_ck * x[k:]
            g[k:] += sign_ck * x[:n - k]
        
        x = x - lr * g
        x = x - (np.sum(x) - 1.0) / n
        x = np.clip(x, -1.5, 1.5)
        x = x - (np.sum(x) - 1.0) / n
        
        v = _metric(x)
        if v < best:
            best = v
            bestx = x.copy()
            no_improve = 0
            lr = min(lr * 1.02, 5e-5)
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.9
                no_improve = 0
                if lr < 1e-8:
                    break
    return bestx, best

def _snap_and_refine(x0, deadline, iters=10000):
    """Snap to +/-1, refine, snap, refine (alternating)."""
    x = x0.copy()
    best = _metric(x)
    bestx = x.copy()
    for _ in range(3):
        if time.time() > deadline:
            break
        xs = np.where(x >= 0, 1.0, -1.0)
        s = np.sum(xs)
        if abs(s) > 1e-9:
            xs = xs / s
        x, v = _refine_subgrad(xs, deadline, iters)
        if v < best:
            best = v
            bestx = x.copy()
    return bestx, best

def _try_m_sequences(deadline, best_x, best_val):
    """Try m-sequence based shapes."""
    polys = [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 1], [7, 3],
             [8, 4], [9, 5], [10, 7], [11, 6], [12, 8], [13, 4], [14, 10], [15, 12]]
    for poly in polys:
        if time.time() > deadline:
            break
        for init in range(min(8, (1 << len(poly)) - 1)):
            if time.time() > deadline:
                break
            seq = _generate_m_sequence(N, poly, init + 1)
            s = np.sum(seq)
            if abs(s) > 1e-9:
                seq = seq / s
            v = _metric(seq)
            if v < best_val:
                best_val = v
                best_x = seq.copy()
                # Refine
                x_ref, v_ref = _refine_subgrad(seq, deadline, iters=5000)
                if v_ref < best_val:
                    best_val = v_ref
                    best_x = x_ref.copy()
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    rng = np.random.default_rng(42)
    
    # Initialize with constant sequence
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    # Strategy 1: L-BFGS-B optimization on continuous relaxation
    # Start from random signs normalized to sum 1
    x_init = rng.choice([-1.0, 1.0], size=N).astype(float)
    s = np.sum(x_init)
    if abs(s) > 1e-6:
        x_init = x_init / s
    else:
        x_init = np.ones(N) / N
        
    try:
        res = scipy.optimize.minimize(
            _objective_offdiag, 
            x_init, 
            method='L-BFGS-B', 
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        x_opt = res.x
        v_opt = _metric_std(x_opt)
        if v_opt < best_val:
            best_val = v_opt
            best_x = x_opt.copy()
    except Exception:
        pass
    
    # Strategy 2: m-sequence search
    if time.time() < deadline:
        best_x, best_val = _try_m_sequences(deadline, best_x, best_val)
    
    # Strategy 3: Subgradient refinement on best candidate found
    if time.time() < deadline:
        x_ref, v_ref = _refine_subgrad(best_x, deadline, iters=15000)
        if v_ref < best_val:
            best_val = v_ref
            best_x = x_ref.copy()
    
    # Strategy 4: High-precision polish
    if time.time() < deadline:
        x_pol, v_pol = _refine_polish(best_x, deadline, iters=20000)
        if v_pol < best_val:
            best_val = v_pol
            best_x = x_pol.copy()
            
    return best_x
