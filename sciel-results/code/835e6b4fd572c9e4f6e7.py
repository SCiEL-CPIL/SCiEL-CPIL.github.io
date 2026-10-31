import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)

    def update_best(h, score):
        nonlocal best_h, best_score
        if score > best_score:
            best_score = score
            best_h = h.copy()

    # Parametric search for good starting points
    candidates = []

    # 1. Gaussian
    x = np.arange(N) - N // 2
    for sigma in np.linspace(0.5, 10.0, 50):
        h = np.exp(-0.5 * (x/sigma)**2)
        candidates.append(h)
        if time.time() > deadline - 30.0:
            break

    # 2. Cauchy-like
    for gamma in np.linspace(0.5, 10.0, 50):
        h = 1.0 / (1.0 + (x/gamma)**2)
        candidates.append(h)
        if time.time() > deadline - 30.0:
            break

    # 3. Uniform blocks (just in case)
    for k in range(2, N):
        h = np.zeros(N)
        h[N//2 - k//2 : N//2 - k//2 + k] = 1.0
        candidates.append(h)

    # Evaluate candidates
    for h in candidates:
        sc = compute_ratio(h)
        update_best(h, sc)

    # Local Optimization on Symmetric Unimodal Subspace
    # Variables: first half of the vector (symmetric)
    # We ensure non-negativity and monotonicity towards center?
    # Actually, just non-negativity and symmetry is enough to start.
    # Unimodality might be enforced by the landscape or we can just search symmetric.
    
    def symmetric_ratio(params):
        # params are the first N//2 + 1 elements (if N is odd, N//2+1; if even, N//2)
        # Let's use N//2 + 1 to cover the center for even N?
        # N=50. Center is 25. Indices 0..49.
        # Symmetric: h[i] = h[49-i].
        # Variables: h[0]...h[24] (25 vars).
        # h[25] is the center? No, 50 is even.
        # Indices 0..24 and 25..49.
        # h[24] = h[25].
        # So we need 25 variables.
        half = N // 2
        h_half = np.maximum(params, 0.0)
        h = np.zeros(N)
        h[:half] = h_half
        h[half:] = h_half[::-1]
        # Ensure sum is 1 for ratio calculation? compute_ratio normalizes.
        sc = compute_ratio(h)
        return -sc

    # Initial guesses for local opt
    starts = []
    # From best candidate
    if best_score > 0:
        h_best = best_h
        h_best = h_best / np.sum(h_best)
        starts.append(h_best[:N//2])
    
    # Random symmetric starts
    for _ in range(20):
        if time.time() > deadline - 10.0:
            break
        h_rand = np.random.exponential(1.0, N//2)
        starts.append(h_rand)

    for start in starts:
        if time.time() > deadline - 5.0:
            break
        
        # Use L-BFGS-B
        try:
            res = minimize(symmetric_ratio, start, method='L-BFGS-B', options={'maxiter': 500})
            if res.success or res.fun < -best_score:
                h_opt = np.zeros(N)
                h_half = np.maximum(res.x, 0.0)
                h_opt[:N//2] = h_half
                h_opt[N//2:] = h_half[::-1]
                sc = compute_ratio(h_opt)
                update_best(h_opt, sc)
        except Exception:
            pass

    return list(best_h)
