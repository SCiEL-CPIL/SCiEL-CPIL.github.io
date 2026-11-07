"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 135.0  # safety margin below 150s kill
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def objective(h):
    """Compute the raw value (lower is better)."""
    # Ensure h is exactly in [0,1] and sum is N/2 for evaluation accuracy
    # The evaluator checks constraints, so we return the projected h
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def project_sum(h):
    """
    Project h onto the hyperplane sum(h) = N/2 while respecting [0,1] bounds.
    Uses a simple iterative water-filling / clipping approach which is fast
    and robust for N=100.
    """
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    for _ in range(50):
        s = h.sum()
        diff = target - s
        if abs(diff) < 1e-12:
            break
        # Adjust only elements that have room to move in the direction of diff
        if diff > 0:
            # Need to increase sum. Add to elements < 1.
            mask = h < 1.0
            if not np.any(mask):
                break
            n_active = np.sum(mask)
            step = diff / n_active
            h[mask] += step
        else:
            # Need to decrease sum. Subtract from elements > 0.
            mask = h > 0.0
            if not np.any(mask):
                break
            n_active = np.sum(mask)
            step = diff / n_active
            h[mask] -= step
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    """
    Strategy: Coordinate Descent with Local Refinement.
    
    Unlike the previous PSO approach which jumps around globally, this method
    refines a good initial guess using local gradient-free line searches along
    coordinate axes. Since the objective is the max of quadratic forms, it is
    smooth almost everywhere. Coordinate descent is effective for such separable-ish
    structures if we handle the coupling via the projection.
    
    We will also use a few random restarts with local optimization (Nelder-Mead)
    to escape potential local minima, but the bulk of the time is spent in
    efficient local refinement.
    """
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # 1.0
    
    # Initial set of candidates
    # 1. Uniform 0.5
    # 2. Random uniform projected
    # 3. Alternating 0/1 pattern (if N even, sum=50) -> h = [1,0,1,0...]
    #    Let's check value for alternating 1,0,1,0...
    #    h = [1,0,1,0...], 1-h = [0,1,0,1...]
    #    correlate(h, 1-h) at lag 0: sum(1*0 + 0*1 + ...) = 0
    #    lag 1: sum(h_i (1-h_{i+1})) = sum(1*1 + 0*0 + ...) = 50 (for N=100)
    #    value = 50 / 100 * 2 = 1.0. Same as uniform.
    #    What about blocks?
    
    candidates = []
    
    # Candidate 1: Uniform
    candidates.append(np.full(N, 0.5))
    
    # Candidate 2: Random
    np.random.seed(42)
    h_rand = np.random.uniform(0, 1, N)
    candidates.append(project_sum(h_rand))
    
    # Candidate 3: Two-level structure (0 and 1)
    # We want to maximize min_k sum h_i h_{i+k}.
    # If h is binary, sum h_i h_{i+k} is the number of overlaps of 1s.
    # This is related to difference sets.
    # For N=100, K=50 ones.
    # Let's try a few random binary patterns.
    for _ in range(10):
        h_bin = np.zeros(N)
        indices = np.random.choice(N, 50, replace=False)
        h_bin[indices] = 1.0
        candidates.append(h_bin)
        
    # Candidate 4: Smooth sinusoidal-like variations
    # h_i = 0.5 + A * sin(2*pi*i/N + phase)
    # Clip and project.
    for A in [0.1, 0.2, 0.4]:
        for phase in [0, np.pi/2, np.pi]:
            i = np.arange(N)
            h_sine = 0.5 + A * np.sin(2 * np.pi * i / N + phase)
            candidates.append(project_sum(h_sine))

    # Evaluate initial candidates
    for h_c in candidates:
        h_c = project_sum(h_c)
        val = objective(h_c)
        if val < best_val:
            best_val = val
            best_h = h_c.copy()

    # Local Refinement Loop
    # We use a simple coordinate descent step:
    # For each i, try to change h_i by small deltas, keep the best feasible change.
    # To speed this up, we can batch small perturbations.
    
    # However, pure coordinate descent on 100 vars might be slow if done naively.
    # Let's use scipy.optimize.minimize with Nelder-Mead on the projected space.
    # Nelder-Mead is robust to non-smoothness (max function) and doesn't need gradients.
    
    def obj_for_scipy(x):
        h_p = project_sum(x)
        return objective(h_p)

    # Run Nelder-Mead from the best candidate
    if remaining_time() > 15:
        res = minimize(
            obj_for_scipy,
            best_h,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-9,
                'fatol': 1e-9,
                'adaptive': True
            }
        )
        h_res = project_sum(res.x)
        val_res = objective(h_res)
        
        if val_res < best_val:
            best_val = val_res
            best_h = h_res.copy()
    
    # If time remains, try a second restart from a different random point
    if remaining_time() > 15:
        # Generate a new random start, biased towards the current best
        h_start = best_h + np.random.normal(0, 0.1, N)
        h_start = project_sum(h_start)
        
        res2 = minimize(
            obj_for_scipy,
            h_start,
            method='Nelder-Mead',
            options={
                'maxiter': 1000,
                'xatol': 1e-9,
                'fatol': 1e-9,
                'adaptive': True
            }
        )
        h_res2 = project_sum(res2.x)
        val_res2 = objective(h_res2)
        
        if val_res2 < best_val:
            best_val = val_res2
            best_h = h_res2.copy()

    # Final sanity check and projection
    best_h = project_sum(best_h)
    assert np.all(np.isfinite(best_h)), "Non-finite values in result"
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Out of bounds"
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h


# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
