import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
DEADLINE = 110.0  # safety margin before 150s kill

dx_step = 0.5 / N  # step width = 0.01
step_centers = -0.25 + (np.arange(N) + 0.5) * dx_step

def _autocorr_max_exact(h):
    """Compute max of autoconvolution of step function h exactly.
    For step functions on uniform grid, (h*h)(k*dx) = dx * R[k] where
    R[k] = sum_i h[i]*h[i+k] is the discrete autocorrelation.
    """
    h = np.asarray(h, dtype=float)
    n = 1
    while n < 2 * N:
        n *= 2
    g = np.zeros(n)
    g[:N] = h
    F = np.fft.rfft(g)
    ac = np.fft.irfft(F * F.conj(), n=n)
    # ac[k] for k=0..N-1 gives R[k] = sum_i h[i]*h[i+k]
    # ac[N-k] for k=1..N-1 gives R[-k]
    R_pos = ac[:N]
    R_neg = ac[N-1:0:-1]  # R[-(N-1)], ..., R[-1]
    R = np.concatenate([R_neg, R_pos])
    max_R = float(np.max(R))
    if max_R <= 0:
        return 0.0
    return dx_step * max_R

def _f_from_h(h):
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    int_f2 = dx_step * float(np.sum(h * h))
    int_f = dx_step * float(np.sum(h))
    if int_f <= 0:
        return 0.0
    acmax = _autocorr_max_exact(h)
    if acmax <= 0:
        return 0.0
    return int_f2 / (int_f * acmax)

def _objective(x):
    x = np.maximum(x, 0.0)
    mx = np.max(x)
    if mx <= 0:
        return 0.0
    h = x / mx
    return -_f_from_h(h)

def _param_shapes():
    shapes = []
    # Delta at center
    s = np.zeros(N); s[N//2] = 1.0; shapes.append(s)
    # Delta at various positions
    for pos in [0, 5, 10, 15, 20, 24, 25, 30, 35, 40, 45, 49]:
        s = np.zeros(N); s[pos] = 1.0; shapes.append(s)
    # Two adjacent deltas
    for pos in [0, 5, 10, 15, 20, 24, 25, 30, 35, 40, 45]:
        s = np.zeros(N); s[pos] = 1.0; s[pos+1] = 1.0; shapes.append(s)
    # Two distant deltas
    for d in [1, 2, 3, 5, 10, 15, 20]:
        c = N // 2
        s = np.zeros(N)
        s[c - d] = 1.0; s[c + d] = 1.0
        shapes.append(s)
    # Narrow plateau (2-3 steps)
    for width in [2, 3]:
        for c in [10, 15, 20, 25, 30, 35, 40]:
            s = np.zeros(N)
            s[c:c+width] = 1.0
            shapes.append(s)
    # Gaussian shapes
    for sig in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]:
        shapes.append(np.exp(-step_centers**2 / (2*sig*sig)))
    # Lorentzian
    for sig in [0.02, 0.05, 0.1]:
        shapes.append(1.0/(1.0 + (step_centers/sig)**2))
    # Triangle
    for w in [0.02, 0.05, 0.1, 0.2]:
        shapes.append(np.clip(1.0 - np.abs(step_centers)/w, 0, None))
    # Plateau
    for w in [0.02, 0.05, 0.1]:
        shapes.append((np.abs(step_centers) < w).astype(float))
    # Cosine bump
    for w in [0.05, 0.1, 0.2]:
        shapes.append(np.clip(np.cos(np.pi * step_centers / w), 0, None))
    # Two Gaussians
    for d in [0.05, 0.1, 0.15]:
        for sig in [0.02, 0.05]:
            shapes.append(np.exp(-(step_centers-d)**2/(2*sig*sig)) + np.exp(-(step_centers+d)**2/(2*sig*sig)))
    return shapes

def _polish(h, t0):
    """Polish a starting point using L-BFGS-B with tight tolerances."""
    best_h = h.copy()
    best_f = _f_from_h(best_h)
    bounds = [(0.0, 10.0)] * N
    try:
        res = minimize(_objective, best_h, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12})
        x = np.maximum(res.x, 0.0)
        mx = np.max(x)
        if mx > 0:
            h2 = x / mx
            f2 = _f_from_h(h2)
            if f2 > best_f:
                best_f = f2
                best_h = h2
    except Exception:
        pass
    return best_h, best_f

def construct():
    best_h = np.ones(N)
    best_f = _f_from_h(best_h)
    t0 = time.time()
    
    starts = _param_shapes()
    
    # Add random initializations with various sparsity patterns
    rng = np.random.default_rng(42)
    for _ in range(20):
        starts.append(rng.random(N))
    for _ in range(10):
        # Sparse random
        s = np.zeros(N)
        k = rng.integers(1, 6)
        idx = rng.choice(N, size=k, replace=False)
        s[idx] = rng.random(k) + 0.1
        starts.append(s)
    for _ in range(5):
        # Very sparse: single bump
        s = np.zeros(N)
        c = rng.integers(5, N-5)
        w = rng.integers(1, 4)
        s[c-w:c+w+1] = np.clip(1 - np.abs(np.arange(-w, w+1))/w, 0, None)
        starts.append(s)
    
    # Evaluate all starting points
    for s in starts:
        if time.time() - t0 > DEADLINE:
            break
        s = np.maximum(s, 0.0)
        mx = np.max(s)
        if mx <= 0:
            continue
        h = s / mx
        f = _f_from_h(h)
        if f > best_f:
            best_f = f
            best_h = h
    
    # Polish the best few starting points
    # Re-evaluate to get top candidates
    candidates = []
    for s in starts:
        if time.time() - t0 > DEADLINE:
            break
        s = np.maximum(s, 0.0)
        mx = np.max(s)
        if mx <= 0:
            continue
        h = s / mx
        f = _f_from_h(h)
        candidates.append((f, h))
    candidates.sort(key=lambda x: -x[0])
    
    # Polish top 15 candidates
    for f, h in candidates[:15]:
        if time.time() - t0 > DEADLINE:
            break
        h2, f2 = _polish(h, t0)
        if f2 > best_f:
            best_f = f2
            best_h = h2
    
    # Also try a few more random restarts if time permits
    if time.time() - t0 < DEADLINE - 20:
        for _ in range(10):
            if time.time() - t0 > DEADLINE:
                break
            s = rng.random(N)
            s = np.maximum(s, 0.0)
            mx = np.max(s)
            if mx <= 0:
                continue
            h = s / mx
            h2, f2 = _polish(h, t0)
            if f2 > best_f:
                best_f = f2
                best_h = h2
    
    # Final: try coordinate-descent-style refinement on best
    if time.time() - t0 < DEADLINE - 5:
        h = best_h.copy()
        for _ in range(3):
            if time.time() - t0 > DEADLINE:
                break
            # Small perturbations
            for _ in range(20):
                if time.time() - t0 > DEADLINE:
                    break
                h_pert = h.copy()
                idx = rng.integers(0, N)
                h_pert[idx] = max(0, h_pert[idx] + rng.normal(0, 0.05))
                mx = np.max(h_pert)
                if mx > 0:
                    h_pert = h_pert / mx
                    f_pert = _f_from_h(h_pert)
                    if f_pert > best_f:
                        best_f = f_pert
                        best_h = h_pert
                        h = best_h.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
