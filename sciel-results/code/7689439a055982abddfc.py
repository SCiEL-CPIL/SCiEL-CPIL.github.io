"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _overlap_values(h):
    """Compute overlap values for all shifts k=1..N-1."""
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(1, N):
        overlaps[k] = np.dot(h[:N-k], 1.0 - h[k:])
    return overlaps

def _max_overlap(h):
    """Compute the maximum overlap value."""
    return np.max(_overlap_values(h))

def _smooth_max(values, mu=0.5):
    """Smooth approximation of max using log-sum-exp."""
    shifted = values / mu
    m = np.max(shifted)
    return mu * (m + np.log(np.sum(np.exp(shifted - m))))

def _smooth_max_grad(values, mu=0.5):
    """Gradient of smooth max w.r.t. values."""
    shifted = values / mu
    m = np.max(shifted)
    weights = np.exp(shifted - m) / np.sum(np.exp(shifted - m))
    return weights / mu

def _h_from_coeffs(coeffs, n_modes):
    """
    Convert spectral coefficients to height sequence.
    coeffs[0] = DC offset, coeffs[1] = a1 (cos), coeffs[2] = b1 (sin), ...
    h[i] = coeffs[0] + sum_{k=1}^{n_modes} (coeffs[2k-1]*cos(2*pi*k*i/N) + coeffs[2k]*sin(2*pi*k*i/N))
    """
    N = len(coeffs) // 2 + 1 if n_modes is None else N_global
    # Actually let's be more explicit
    n = N_global
    h = np.full(n, coeffs[0])
    for k in range(1, n_modes + 1):
        a = coeffs[2*k - 1]
        b = coeffs[2*k]
        h += a * np.cos(2 * np.pi * k * np.arange(n) / n)
        h += b * np.sin(2 * np.pi * k * np.arange(n) / n)
    return h

def _fix_sum(h, target_sum):
    """Adjust h to have exact sum while staying in [0,1]."""
    h = np.clip(h, 0.0, 1.0)
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    # Distribute correction
    h += diff / len(h)
    h = np.clip(h, 0.0, 1.0)
    # Iterative correction
    for _ in range(20):
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-14:
            break
        # Adjust elements that have room
        can_inc = h < 1.0 - 1e-10
        can_dec = h > 1e-10
        if diff > 0:
            if np.any(can_inc):
                mask = can_inc
                room = 1.0 - h[mask]
                total_room = np.sum(room)
                if total_room > 0:
                    adj = np.minimum(diff / total_room, 1.0)
                    h[mask] += room * adj
        else:
            if np.any(can_dec):
                mask = can_dec
                room = h[mask]
                total_room = np.sum(room)
                if total_room > 0:
                    adj = np.minimum(-diff / total_room, 1.0)
                    h[mask] -= room * adj
    return h

def _score_from_coeffs(coeffs, n_modes):
    """Compute score (max overlap) from spectral coefficients."""
    h = _h_from_coeffs(coeffs, n_modes)
    h = _fix_sum(h, N_global / 2.0)
    return _max_overlap(h), h

def construct():
    """
    Construct step heights using spectral basis optimization.
    """
    global N_global
    N_global = N
    deadline = time.time() + 140  # 10-second safety margin
    
    # Start with a reasonable number of modes
    best_score = np.inf
    best_h = np.full(N, 0.5)
    
    # Try different numbers of modes
    for n_modes in [3, 5, 8, 12]:
        if time.time() > deadline - 20:
            break
        
        # Number of coefficients: 1 (DC) + 2*n_modes (cos and sin for each mode)
        n_coeffs = 1 + 2 * n_modes
        
        # Initial guess: DC = 0.5, others small random
        np.random.seed(42 + n_modes)
        x0 = np.zeros(n_coeffs)
        x0[0] = 0.5  # DC component ensures sum ≈ N/2
        x0[1:] = np.random.randn(n_coeffs - 1) * 0.1
        
        # Bounds for coefficients
        # DC: [0, 1], others: [-0.5, 0.5] (conservative)
        bounds = [(0.0, 1.0)] + [(-0.5, 0.5)] * (n_coeffs - 1)
        
        # Objective function
        def objective(x):
            h = _h_from_coeffs(x, n_modes)
            h = _fix_sum(h, N_global / 2.0)
            ov = _overlap_values(h)
            return _smooth_max(ov, mu=0.3)
        
        # Try differential evolution first (global search)
        try:
            de_res = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=50,
                tol=1e-6,
                seed=42,
                polish=True,
                workers=1
            )
            if de_res.fun < best_score:
                h = _h_from_coeffs(de_res.x, n_modes)
                h = _fix_sum(h, N_global / 2.0)
                score = _max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 10:
            break
        
        # Refine with L-BFGS-B
        try:
            # Start from best DE result or x0
            x_start = de_res.x if 'de_res' in dir() else x0
            
            def lbfgs_objective(x):
                h = _h_from_coeffs(x, n_modes)
                h = _fix_sum(h, N_global / 2.0)
                ov = _overlap_values(h)
                return _smooth_max(ov, mu=0.1)
            
            res = minimize(
                lbfgs_objective,
                x_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.fun < best_score:
                h = _h_from_coeffs(res.x, n_modes)
                h = _fix_sum(h, N_global / 2.0)
                score = _max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
        except Exception:
            pass
    
    # Final refinement: local search on full h vector
    if time.time() < deadline - 5 and best_score < np.inf:
        h_current = best_h.copy()
        
        def full_objective(h):
            ov = _overlap_values(h)
            return _smooth_max(ov, mu=0.05)
        
        # Use SLSQP for final polish
        try:
            def constraint_sum(x):
                return np.sum(x) - N_global / 2.0
            
            res = minimize(
                full_objective,
                h_current,
                method='SLSQP',
                bounds=[(0, 1)] * N_global,
                constraints=[{'type': 'eq', 'fun': constraint_sum}],
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if res.fun < best_score:
                h = _fix_sum(res.x, N_global / 2.0)
                score = _max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
        except Exception:
            pass
    
    # Ensure valid output
    best_h = _fix_sum(best_h, N_global / 2.0)
    return best_h

# EVOLVE-BLOCK-END
