"""Rewritten program for Erdos minimum overlap using spectral optimization with smooth refinement."""

import time
import numpy as np
from scipy.optimize import minimize

N = 100

def _linear_overlap_max(h):
    """Calculate max linear overlap for shifts k=1..N-1."""
    max_ov = 0.0
    for k in range(1, N):
        # overlap = sum_{i=k}^{N-1} h[i] * (1 - h[i-k])
        ov = np.sum(h[k:] * (1.0 - h[:-k]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _analytic_overlap_max(m, A, N):
    """
    Estimate max linear overlap for h_i = 0.5 + A * cos(2*pi*m*i/N).
    Uses closed-form sums for speed during search.
    """
    if A == 0:
        # Constant case
        return (N - 1) / 4.0
        
    theta_step = 2 * np.pi * m / N
    max_ov = 0.0
    
    # Precompute some terms if possible, but N is small.
    for k in range(1, N):
        # Term 1: 0.25 * (N-k)
        t1 = 0.25 * (N - k)
        
        # Term 2: -0.5 * A * sum_{j=0}^{N-1-k} cos(2*pi*m*j/N)
        # Sum of cos is Re( sum (e^{i theta})^j ) = Re( (1 - e^{i theta (N-k)}) / (1 - e^{i theta}) )
        # Let M = N - k
        M = N - k
        if M == 0:
            t2 = 0
        else:
            # Handle theta close to 0 (m=0 or m=N, not in loop) or m=N/2
            z = np.exp(1j * theta_step)
            if np.isclose(z, 1.0):
                sum_cos = M
            else:
                # Geometric series sum
                sum_z = (1 - z**M) / (1 - z)
                sum_cos = np.real(sum_z)
            t2 = -0.5 * A * sum_cos
            
        # Term 3: 0.5 * A * sum_{i=k}^{N-1} cos(2*pi*m*i/N)
        # This is sum_{i=k}^{N-1} cos(theta_step * i)
        # = Re( e^{i theta k} * (1 - e^{i theta (N-k)}) / (1 - e^{i theta}) )
        if np.isclose(z, 1.0):
            sum_cos_2 = M
        else:
            sum_z_2 = (1 - z**M) / (1 - z)
            sum_cos_2 = np.real(z**k * sum_z_2)
        t3 = 0.5 * A * sum_cos_2
        
        # Term 4: -A^2 * sum_{i=k}^{N-1} cos(theta_i) cos(theta_{i-k})
        # = -A^2 * 0.5 * [ (N-k) * cos(2*pi*m*k/N) + sum_{i=k}^{N-1} cos(2*pi*m*(2i-k)/N) ]
        
        # Part 4a: (N-k) * cos(2*pi*m*k/N)
        t4a = M * np.cos(2 * np.pi * m * k / N)
        
        # Part 4b: sum_{i=k}^{N-1} cos(2*pi*m*(2i-k)/N)
        # Let phi_i = 2*pi*m*(2i-k)/N.
        # Step is 4*pi*m/N.
        # Initial angle for i=k: 2*pi*m*(2k-k)/N = 2*pi*m*k/N.
        # Number of terms: M.
        step_2 = 4 * np.pi * m / N
        z_2 = np.exp(1j * step_2)
        if np.isclose(z_2, 1.0):
            sum_cos_4b = M
        else:
            sum_z_4b = (1 - z_2**M) / (1 - z_2)
            # Start angle is theta_start = 2*pi*m*k/N
            sum_cos_4b = np.real(np.exp(1j * 2 * np.pi * m * k / N) * sum_z_4b)
            
        t4 = -0.5 * A * A * (t4a + sum_cos_4b)
        
        ov = t1 + t2 + t3 + t4
        if ov > max_ov:
            max_ov = ov
            
    return max_ov

def _soft_max_overlap(h, beta=20.0):
    """
    Smooth surrogate for max overlap using log-sum-exp.
    This allows for gradient-based optimization.
    """
    overlaps = np.zeros(N)
    for k in range(1, N):
        overlaps[k] = np.sum(h[k:] * (1.0 - h[:-k]))
    
    # Log-sum-exp for numerical stability
    m = np.max(overlaps)
    if m < 1e-10:
        return 0.0
    return m + np.log(np.sum(np.exp((overlaps - m) * beta))) / beta

def _project_to_feasible(h):
    """
    Project h to the feasible set:
    1. h in [0, 1]
    2. sum(h) = N/2
    Uses a simple iterative clipping and scaling method.
    """
    h = np.array(h, dtype=float)
    for _ in range(20):
        # Clip to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        
        # Check sum
        current_sum = np.sum(h)
        target_sum = N / 2.0
        
        if abs(current_sum - target_sum) < 1e-10:
            break
            
        # If sum is too high, we need to decrease elements.
        # If sum is too low, we need to increase elements.
        # We adjust the elements that have the most "room" in the direction of adjustment.
        if current_sum > target_sum:
            # Need to decrease. Elements closer to 1 have more room to decrease.
            # We can simply scale down, but that might push some to 0 if they are already small.
            # Better: subtract proportional to (1 - h) ? Or just scale?
            # Let's try a simple scaling first, then clip.
            h = h * (target_sum / current_sum)
        else:
            # Need to increase. Elements closer to 0 have more room to increase.
            h = h * (target_sum / current_sum)
            
    # Final clip
    h = np.clip(h, 0.0, 1.0)
    
    # Final precise adjustment to ensure sum is exactly N/2 within tolerance
    # This might break bounds slightly, so we do it carefully.
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-9:
        # Distribute the difference among elements that have space
        # If diff > 0, add to elements with h < 1
        # If diff < 0, subtract from elements with h > 0
        if diff > 0:
            # Find indices where h < 1
            mask = h < 1.0
            if np.sum(mask) > 0:
                # Distribute proportionally to available space (1 - h)
                space = 1.0 - h
                space[~mask] = 0
                total_space = np.sum(space)
                if total_space > 0:
                    h += diff * (space / total_space)
        else:
            # Find indices where h > 0
            mask = h > 0.0
            if np.sum(mask) > 0:
                # Distribute proportionally to current value
                # Or just distribute equally among non-zero?
                # Proportional is smoother
                vals = h
                vals[~mask] = 0
                total_val = np.sum(vals)
                if total_val > 0:
                    h += diff * (vals / total_val)
                    
    # Final clip to ensure strict bounds
    h = np.clip(h, 0.0, 1.0)
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Baseline: constant 0.5
    best_h = np.full(N, 0.5)
    best_score = _linear_overlap_max(best_h)
    
    # Spectral search: h_i = 0.5 + A * cos(2*pi*m*i/N + phase)
    # We can assume phase=0 without loss of generality for the max overlap profile?
    # Shifting the sequence in space just shifts the k-index of the overlap, 
    # but since we take MAX over all k, the phase doesn't change the max value.
    # So we only need to search m and A.
    
    # Search frequencies
    best_m = 0
    best_A = 0.0
    
    for m in range(1, N // 2 + 1):
        if time.time() > deadline - 20.0:
            break
            
        # For each m, find optimal A
        # A is in [0, 0.5]
        # We can do a coarse grid search for A first
        best_A_for_m = 0.0
        best_score_for_m = best_score
        
        # Coarse search for A
        A_coarse = np.linspace(0.0, 0.5, 50)
        for A in A_coarse:
            if time.time() > deadline - 15.0:
                break
            score = _analytic_overlap_max(m, A, N)
            if score < best_score_for_m:
                best_score_for_m = score
                best_A_for_m = A
        
        # Fine search around best_A_for_m
        if best_A_for_m > 0:
            A_fine = np.linspace(max(0, best_A_for_m - 0.01), min(0.5, best_A_for_m + 0.01), 50)
            for A in A_fine:
                if time.time() > deadline - 10.0:
                    break
                score = _analytic_overlap_max(m, A, N)
                if score < best_score_for_m:
                    best_score_for_m = score
                    best_A_for_m = A
        
        # If this m is better than global best, update
        if best_score_for_m < best_score:
            best_score = best_score_for_m
            best_m = m
            best_A = best_A_for_m
            i = np.arange(N)
            best_h = 0.5 + best_A_for_m * np.cos(2 * np.pi * m * i / N)
            # Ensure bounds and sum constraint
            best_h = np.clip(best_h, 0.0, 1.0)
            best_h = _project_to_feasible(best_h)

    # Refine with local search using smooth surrogate
    if best_m > 0 and time.time() < deadline - 5.0:
        # Define objective for scipy
        def objective(params):
            h = _project_to_feasible(params)
            return _soft_max_overlap(h, beta=30.0)
        
        # Initial point
        x0 = best_h.copy()
        
        try:
            # Use L-BFGS-B or similar
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success:
                h_opt = _project_to_feasible(res.x)
                score_opt = _linear_overlap_max(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt
        except Exception:
            pass

    # Final random polish
    if time.time() < deadline - 2.0:
        for trial in range(20):
            if time.time() > deadline - 2.0:
                break
            h = best_h + np.random.normal(0, 0.02, N)
            h = _project_to_feasible(h)
            score = _linear_overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()

    return best_h

if __name__ == "__main__":
    construct()
