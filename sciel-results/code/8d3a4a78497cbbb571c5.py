"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _softmin(x, t=0.01):
    """Smooth approximation of min(x) using log-sum-exp."""
    x = np.asarray(x)
    # For numerical stability
    xm = np.min(x)
    return xm - t * np.log(np.mean(np.exp((x - xm) / t)))

def _objective(params, N, t):
    """Objective to minimize: -softmin of all pairwise half-distances and boundary clearances."""
    c = params.reshape(N, 3)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Get upper triangle (i < j)
    iu = np.triu_indices(N, k=1)
    pd = dists[iu] / 2.0
    
    # Boundary clearances: 1/2 - |c_ik| for each coordinate
    clearances = 0.5 - np.abs(c)
    # All boundary clearances
    bc = clearances.flatten()
    
    # Combine all values
    all_vals = np.concatenate([pd, bc])
    
    return -_softmin(all_vals, t)

def _score_centres(c):
    """Compute the actual (non-smooth) score for a set of centres."""
    c = np.asarray(c)
    if c.shape != (N, 3):
        return -1
    if not np.all(np.isfinite(c)):
        return -1
    # Check inside cube
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return -1
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pd = dists[iu] / 2.0
    # Boundary clearances
    bc = (0.5 - np.abs(c)).flatten()
    r = min(np.min(pd), np.min(bc))
    return r

def _fibonacci_lattice(N):
    """Generate N points in [-0.5, 0.5]^3 using a 3D quasi-random/Fibonacci-like approach."""
    # Use a 3D Fibonacci lattice: distribute points on a grid with golden-ratio offsets
    # Actually, let's use a simpler approach: stratified sampling with jitter
    # Or use the 3D Fibonacci spiral mapped to cube
    
    # Better: use a 3D quasi-random sequence (Halton-like or just well-distributed)
    # For simplicity and good distribution, use a 3D grid with golden ratio offsets
    
    # Let's use a 3D Fibonacci lattice approach:
    # Place points at (i*phi mod 1, j*phi^2 mod 1, k*phi^3 mod 1) type scheme
    # Simpler: use a 3D Halton sequence
    
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((N, 3))
    for i in range(N):
        idx = i + 1
        pts[i, 0] = halton(idx, 2) - 0.5
        pts[i, 1] = halton(idx, 3) - 0.5
        pts[i, 2] = halton(idx, 5) - 0.5
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_c = None
    best_score = -1.0
    
    # Initial guess: Halton sequence
    c0 = _fibonacci_lattice(N)
    best_c = c0.copy()
    best_score = _score_centres(best_c)
    
    # Try multiple restarts with different initializations and temperature
    attempts = 0
    max_attempts = 30
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Create perturbed initialization
        if attempts == 1:
            c_init = best_c.copy()
        else:
            # Perturb the best solution or use fresh random
            if np.random.random() < 0.5 and best_c is not None:
                c_init = best_c + np.random.randn(N, 3) * 0.05
            else:
                c_init = _fibonacci_lattice(N) + np.random.randn(N, 3) * 0.02
            # Clamp to cube
            c_init = np.clip(c_init, -0.49, 0.49)
        
        # Use L-BFGS-B with bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Start with larger t (smoother) and anneal to smaller t
        for t in [0.05, 0.02, 0.005, 0.001]:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    _objective,
                    c_init.flatten(),
                    args=(N, t),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                c_new = res.x.reshape(N, 3)
                # Ensure validity
                c_new = np.clip(c_new, -0.5, 0.5)
                s = _score_centres(c_new)
                if s > best_score:
                    best_score = s
                    best_c = c_new.copy()
                c_init = c_new  # Continue from this point for next t
            except Exception:
                break
        
        # If we've found a good solution, do a final polish with very small t
        if time.time() > deadline - 20.0:
            break
    
    # Final polish: one more optimization with very small t on the best solution
    if best_c is not None and time.time() < deadline - 5.0:
        try:
            bounds = [(-0.5, 0.5)] * (N * 3)
            res = minimize(
                _objective,
                best_c.flatten(),
                args=(N, 0.0001),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            s = _score_centres(c_new)
            if s > best_score:
                best_score = s
                best_c = c_new.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_c is None:
        best_c = _fibonacci_lattice(N)
    
    # Final validation and clamping
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
