import numpy as np
import time
from scipy.optimize import minimize

N = 100

def calculate_max_overlap(h):
    """Calculate maximum linear overlap of h with shifted 1-h."""
    max_ov = 0.0
    for k in range(N):
        if k == 0:
            ov = np.sum(h * (1 - h))
        else:
            ov = np.sum(h[k:] * (1 - h[:-k]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def soft_max_overlaps(h, beta=20.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    overlaps = np.zeros(N)
    for k in range(N):
        if k == 0:
            overlaps[k] = np.sum(h * (1 - h))
        else:
            overlaps[k] = np.sum(h[k:] * (1 - h[:-k]))
    # Log-sum-exp for numerical stability
    m = np.max(overlaps)
    if m < 1e-10:
        return 0.0
    return m + np.log(np.sum(np.exp((overlaps - m) * beta))) / beta

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

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Base solution
    h_base = np.full(N, 0.5)
    base_score = calculate_max_overlap(h_base)
    
    best_h = h_base.copy()
    best_score = base_score
    
    # --- Phase 1: Spectral Search (Analytic) ---
    # Search for the best single sinusoidal perturbation h_i = 0.5 + A * cos(2*pi*m*i/N)
    # This provides a strong, structured starting point.
    
    best_m = 1
    best_A = 0.0
    best_analytic_score = base_score
    
    for m in range(1, N // 2 + 1):
        if time.time() > deadline - 30.0:
            break
            
        # Coarse search for A
        best_A_for_m = 0.0
        best_score_for_m = best_analytic_score
        
        A_coarse = np.linspace(0.0, 0.5, 50)
        for A in A_coarse:
            score = _analytic_overlap_max(m, A, N)
            if score < best_score_for_m:
                best_score_for_m = score
                best_A_for_m = A
        
        # Fine search around best_A_for_m
        if best_A_for_m > 0:
            A_fine = np.linspace(max(0, best_A_for_m - 0.01), min(0.5, best_A_for_m + 0.01), 50)
            for A in A_fine:
                score = _analytic_overlap_max(m, A, N)
                if score < best_score_for_m:
                    best_score_for_m = score
                    best_A_for_m = A
        
        if best_score_for_m < best_analytic_score:
            best_analytic_score = best_score_for_m
            best_m = m
            best_A = best_A_for_m
            
    # Construct the initial candidate from the best spectral parameters
    i = np.arange(N)
    h_spectral = 0.5 + best_A * np.cos(2 * np.pi * best_m * i / N)
    h_spectral = np.clip(h_spectral, 0.0, 1.0)
    
    # Correct sum to exactly N/2
    diff = N / 2.0 - np.sum(h_spectral)
    if abs(diff) > 1e-12:
        # Simple proportional correction
        h_spectral = h_spectral + diff / N
        h_spectral = np.clip(h_spectral, 0.0, 1.0)
        # Re-correct if clipping broke the sum
        diff2 = N / 2.0 - np.sum(h_spectral)
        if abs(diff2) > 1e-12:
             idx = np.argmin(np.abs(h_spectral - 0.5))
             h_spectral[idx] += diff2
             h_spectral = np.clip(h_spectral, 0.0, 1.0)
             
    spectral_score = calculate_max_overlap(h_spectral)
    if spectral_score < best_score:
        best_h = h_spectral
        best_score = spectral_score

    # --- Phase 2: Local Optimization (Numerical) ---
    # Use the spectral solution as a starting point for a more general optimization.
    # We will optimize the full vector h directly, but use a parameterization that
    # allows small deviations from the sinusoidal shape if beneficial, or just
    # use L-BFGS-B on the vector h with bounds.
    
    # Given N=100, optimizing 100 variables is feasible.
    # Objective: soft_max_overlaps
    # Constraints: bounds [0, 1] and sum == N/2.
    # L-BFGS-B handles bounds. Sum constraint can be handled by projecting or
    # by optimizing N-1 variables and setting the last to satisfy the sum.
    # Let's use the projection method: optimize h, then project to sum=N/2.
    # However, L-BFGS-B doesn't support equality constraints directly in a simple way.
    # Alternative: Parameterize h = 0.5 + delta, where sum(delta)=0.
    # Or simply use a penalty method or just accept small sum errors if we
    # correct at the end? No, the task requires exact sum.
    
    # Let's stick to the vector optimization with a correction step inside the objective
    # or use a different optimizer.
    # Actually, scipy.optimize.minimize with method='SLSQP' supports equality constraints.
    # But SLSQP might be slow for 100 variables.
    
    # Let's try optimizing the vector h with L-BFGS-B and a penalty for sum deviation.
    # Or better, since we have a good initial guess, we can do a few iterations of
    # projected gradient descent or just trust the spectral solution if it's good.
    
    # Let's try a local refinement using L-BFGS-B on the vector h.
    # We will define an objective that penalizes sum deviation heavily.
    
    def objective_vector(h_flat):
        h = h_flat.reshape(N)
        # Penalty for sum deviation
        sum_dev = np.sum(h) - N / 2.0
        penalty = 1000.0 * sum_dev**2
        # Penalty for bounds (should be handled by L-BFGS-B bounds, but just in case)
        bound_penalty = 0.0
        if np.min(h) < 0: bound_penalty += 1000.0 * (np.min(h)**2)
        if np.max(h) > 1: bound_penalty += 1000.0 * (np.max(h)-1)**2
        
        val = soft_max_overlaps(h, beta=20.0) + penalty + bound_penalty
        return val
    
    bounds = [(0.0, 1.0)] * N
    x0 = best_h.copy()
    
    try:
        res = minimize(objective_vector, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.success:
            h_opt = res.x
            # Final sum correction
            diff = N / 2.0 - np.sum(h_opt)
            if abs(diff) > 1e-12:
                # Distribute correction
                weights = np.abs(h_opt - 0.5) + 1e-6
                weights /= np.sum(weights)
                h_opt = h_opt + diff * weights
                h_opt = np.clip(h_opt, 0.0, 1.0)
                # Re-correct
                diff2 = N / 2.0 - np.sum(h_opt)
                if abs(diff2) > 1e-12:
                    idx = np.argmin(np.abs(h_opt - 0.5))
                    h_opt[idx] += diff2
                    h_opt = np.clip(h_opt, 0.0, 1.0)
                    
            if np.min(h_opt) >= 0 and np.max(h_opt) <= 1:
                opt_score = calculate_max_overlap(h_opt)
                if opt_score < best_score:
                    best_h = h_opt
                    best_score = opt_score
    except Exception:
        pass

    # --- Phase 3: Final Polish ---
    # Small random perturbations to escape local minima or refine further.
    if time.time() < deadline - 5.0:
        for trial in range(50):
            if time.time() > deadline - 2.0:
                break
            noise = np.random.normal(0, 0.01, N)
            h_trial = best_h + noise
            h_trial = np.clip(h_trial, 0.0, 1.0)
            
            # Correct sum
            diff = N / 2.0 - np.sum(h_trial)
            if abs(diff) > 1e-12:
                weights = np.abs(h_trial - 0.5) + 1e-6
                weights /= np.sum(weights)
                h_trial = h_trial + diff * weights
                h_trial = np.clip(h_trial, 0.0, 1.0)
                diff2 = N / 2.0 - np.sum(h_trial)
                if abs(diff2) > 1e-12:
                    idx = np.argmin(np.abs(h_trial - 0.5))
                    h_trial[idx] += diff2
                    h_trial = np.clip(h_trial, 0.0, 1.0)
                    
            if np.min(h_trial) >= 0 and np.max(h_trial) <= 1:
                trial_score = calculate_max_overlap(h_trial)
                if trial_score < best_score:
                    best_h = h_trial
                    best_score = trial_score

    return best_h
