import numpy as np
import time

N = 400

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

def _refine_subgrad(x0, deadline, rng, iters=50000, use_softmax=False, beta=50.0):
    """Projected subgradient minimization of off-diagonal max|c| s.t. sum(x)=1.
    
    The ratio max|c|/S^2 is scale-invariant, so we fix sum=1 and minimize
    max|c_k| for k != 0. The zero-lag c0 = sum(x^2) >= 1 by Cauchy-Schwarz
    and equals 1 iff |x_i| = 1/sqrt(N) for all i.
    """
    n = N
    x = np.array(x0, dtype=float).copy()
    s = np.sum(x)
    if abs(s) < 1e-15:
        return x, _metric_std(x)
    x = x / s
    
    best = _metric_std(x)
    bestx = x.copy()
    lr = 5e-4
    no_improve = 0
    patience = 2000
    for it in range(iters):
        if time.time() > deadline:
            break
        c = _autoconv_lin(x)
        
        if use_softmax:
            # Use soft-max for smoother gradient
            abs_c = np.abs(c[1:])  # off-diagonal only
            m = np.max(abs_c)
            if m < 1e-15:
                break
            weights = np.exp(beta * (abs_c - m))
            weights /= weights.sum()
            # Gradient of soft-max w.r.t. c_k for k != 0
            g = np.zeros(n)
            # For each off-diagonal lag k, gradient of c_k w.r.t. x
            for k in range(1, n):
                wk = weights[k - 1]
                if wk < 1e-8:
                    continue
                sign_ck = np.sign(c[k])
                # gradient of c_k = sum_i x_i x_{i+k} w.r.t. x
                # dc_k/dx_i = x_{i+k} for i < n-k, x_{i-k} for i >= k
                if k < n:
                    g[:n - k] += wk * sign_ck * x[k:]
                    g[k:] += wk * sign_ck * x[:n - k]
        else:
            # Hard max: pick the worst off-diagonal lag
            abs_c = np.abs(c[1:])
            k = int(np.argmax(abs_c)) + 1  # +1 because we excluded k=0
            ck = float(c[k])
            g = np.zeros(n)
            sign_ck = np.sign(ck)
            if k < n:
                g[:n - k] += sign_ck * x[k:]
                g[k:] += sign_ck * x[:n - k]
        
        # Step
        x = x - lr * g
        # Project to sum=1
        x = x - (np.sum(x) - 1.0) / n
        # Keep magnitudes bounded to avoid blowup
        x = np.clip(x, -2.0, 2.0)
        # Re-project after clip
        x = x - (np.sum(x) - 1.0) / n
        
        v = _metric_std(x)
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

def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    rng = np.random.default_rng(42)
    
    # Start with constant sequence
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    # 1. Generate diverse initializations
    inits = []
    
    # Constant
    inits.append(np.ones(N))
    
    # Random signs
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # M-sequences
    polys = [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 1], [7, 3],
             [8, 4], [9, 5], [10, 7], [11, 6], [12, 8], [13, 4], [14, 10], [15, 12]]
    for poly in polys:
        if time.time() > deadline:
            break
        for init in range(min(4, (1 << len(poly)) - 1)):
            if time.time() > deadline:
                break
            seq = _generate_m_sequence(N, poly, init + 1)
            inits.append(seq)
    
    # Low-frequency sinusoids
    for freq in [1, 2, 3, 5]:
        t = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
        inits.append(np.sign(np.sin(t)))
    
    # 2. Refine each initialization
    for init in inits:
        if time.time() > deadline:
            break
        
        # Normalize to sum=1
        s = np.sum(init)
        if abs(s) < 1e-12:
            continue
        x_norm = init / s
        
        # Try both hard max and soft max refinement
        for use_softmax in [False, True]:
            if time.time() > deadline:
                break
            
            x_ref, v = _refine_subgrad(x_norm, deadline, rng, iters=20000, use_softmax=use_softmax)
            
            if v < best_val:
                best_val = v
                best_x = x_ref
    
    # 3. Final polish on the best solution
    if time.time() < deadline - 5.0:
        x_polish = best_x.copy()
        s = np.sum(x_polish)
        if abs(s) > 1e-12:
            x_polish = x_polish / s
        
        # Very small steps for high precision
        n = N
        best_polish = _metric_std(x_polish)
        bestx_polish = x_polish.copy()
        lr = 1e-5
        no_improve = 0
        patience = 5000
        
        for it in range(30000):
            if time.time() > deadline:
                break
            c = _autoconv_lin(x_polish)
            abs_c = np.abs(c[1:])
            if len(abs_c) == 0:
                break
            k = int(np.argmax(abs_c)) + 1
            ck = float(c[k])
            g = np.zeros(n)
            sign_ck = np.sign(ck)
            if k < n:
                g[:n - k] += sign_ck * x_polish[k:]
                g[k:] += sign_ck * x_polish[:n - k]
            
            x_polish = x_polish - lr * g
            x_polish = x_polish - (np.sum(x_polish) - 1.0) / n
            x_polish = np.clip(x_polish, -1.5, 1.5)
            x_polish = x_polish - (np.sum(x_polish) - 1.0) / n
            
            v = _metric_std(x_polish)
            if v < best_polish:
                best_polish = v
                bestx_polish = x_polish.copy()
                no_improve = 0
                lr = min(lr * 1.02, 5e-5)
            else:
                no_improve += 1
                if no_improve > patience:
                    lr *= 0.9
                    no_improve = 0
                    if lr < 1e-8:
                        break
        
        if best_polish < best_val:
            best_val = best_polish
            best_x = bestx_polish
    
    return best_x
