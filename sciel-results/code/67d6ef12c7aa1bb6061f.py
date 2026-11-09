"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _overlap_scores(h):
    """Compute overlap at each circular shift k: sum_i h[i]*(1-h[(i-k)%N])"""
    h = np.asarray(h)
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    return overlaps

def _max_overlap(h):
    return _overlap_scores(h).max()

def _soft_max_overlap(h, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    ov = _overlap_scores(h)
    m = ov.max()
    return m + np.log(np.sum(np.exp(beta * (ov - m)))) / beta

def _objective(params, beta=50.0):
    """Objective: soft-max overlap + penalty for out-of-bounds."""
    h = _params_to_h(params)
    # penalty for leaving [0,1]
    penalty = np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2)
    return _soft_max_overlap(h, beta) + 100.0 * penalty

def _params_to_h(params):
    """Map free parameters to step heights in [0,1] via sigmoid, then scale to sum=N/2."""
    # Use a linear combination of basis functions
    x = np.arange(N)
    # Basis: constant + cosines + sines
    h_raw = np.zeros(N)
    idx = 0
    # First param: overall level
    level = params[0]
    h_raw += level
    idx = 1
    # Fourier modes
    for mode in range(1, 8):
        a = params[idx]; b = params[idx+1]
        idx += 2
        h_raw += a * np.cos(2 * np.pi * mode * x / N) + b * np.sin(2 * np.pi * mode * x / N)
    # Apply sigmoid to map to (0,1), then affine to ensure sum = N/2
    h_sigmoid = 1.0 / (1.0 + np.exp(-h_raw * 3.0))
    # Scale to make sum = N/2
    s = np.sum(h_sigmoid)
    if s > 1e-12:
        h = h_sigmoid * (N / 2.0) / s
    else:
        h = np.full(N, 0.5)
    return h

def construct():
    t0 = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Random restarts with Fourier parameterization
    np.random.seed(42)
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() - t0 > deadline - 30:
            break
        
        # Initialize params
        n_params = 1 + 2 * 7  # level + 7 Fourier modes
        if r == 0:
            params0 = np.zeros(n_params)
        else:
            params0 = np.random.randn(n_params) * 0.5
        
        try:
            res = minimize(
                _objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
                tol=1e-10
            )
            h_candidate = _params_to_h(res.x)
            # Clip and renormalize
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            # Adjust to exact sum
            diff = N/2.0 - np.sum(h_candidate)
            h_candidate += diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            # Final exact sum fix
            diff = N/2.0 - np.sum(h_candidate)
            h_candidate += diff / N
            
            sc = _max_overlap(h_candidate)
            if sc < best_score:
                best_score = sc
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Local polish: direct optimization on h with box constraints
    if time.time() - t0 < deadline - 10:
        try:
            def obj_direct(h):
                return _soft_max_overlap(h, beta=100.0)
            
            def jac_direct(h):
                # Numerical gradient
                eps = 1e-6
                g = np.zeros_like(h)
                for i in range(N):
                    hp = h.copy(); hp[i] += eps
                    hm = h.copy(); hm[i] -= eps
                    g[i] = (obj_direct(hp) - obj_direct(hm)) / (2*eps)
                return g
            
            # Use SLSQP with equality constraint for sum
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N
            
            res2 = minimize(
                obj_direct,
                best_h.copy(),
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            h_polished = np.clip(res2.x, 0.0, 1.0)
            diff = N/2.0 - np.sum(h_polished)
            h_polished += diff / N
            h_polished = np.clip(h_polished, 0.0, 1.0)
            diff = N/2.0 - np.sum(h_polished)
            h_polished += diff / N
            
            sc2 = _max_overlap(h_polished)
            if sc2 < best_score:
                best_score = sc2
                best_h = h_polished.copy()
        except Exception:
            pass
    
    # Final exact sum correction
    diff = N/2.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0.0, 1.0)
    # One more exact fix
    diff = N/2.0 - np.sum(best_h)
    best_h += diff / N
    
    return best_h
# EVOLVE-BLOCK-END
