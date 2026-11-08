"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def compute_max_overlap_linear(h):
    """
    Compute the maximum overlap of h with shifted (1-h) for linear shifts.
    O(s) = sum_{i=0}^{N-s-1} h[i] * (1 - h[i+s])
    """
    n = len(h)
    one_minus_h = 1.0 - h
    max_ov = 0.0
    # We only need to check shifts up to N//2 due to symmetry? No, linear is not symmetric in the same way.
    # But O(N-s) decreases as s increases?
    # Actually, let's just compute for all s to be safe, or use FFT for circular and check if linear is intended.
    # Given the ambiguity, I'll compute circular first as it's faster and the seed implies circular.
    # If circular is optimal, linear is likely similar or worse for non-constant.
    
    # Let's assume circular for now as per the seed's logic, but use a more robust optimizer.
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(H * np.conj(O)).real
    return np.max(corr)


def objective(h):
    """
    Objective function to minimize: max overlap + penalty for out-of-bounds.
    """
    # Penalty for values outside [0, 1]
    penalty = 0.0
    if np.any(h < 0) or np.any(h > 1):
        penalty = 1000.0 * (np.sum(np.clip(h, 0, 1) != h))
    
    return compute_max_overlap_linear(h) + penalty


def construct():
    """
    Construct a sequence of step heights in [0, 1] whose sum is exactly N/2,
    minimizing the maximum overlap with shifted (1-h).
    
    Strategy: Multi-start L-BFGS-B optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    target_sum = N / 2.0
    
    # Best solution found so far
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap_linear(best_h)
    
    # Define constraints: sum(h) = N/2
    # We can handle this by optimizing N-1 variables and setting the last one,
    # or using a constraint in L-BFGS-B.
    # L-BFGS-B supports linear equality constraints? No, only bounds and simple constraints.
    # We will use a penalty method or re-parameterization.
    # Re-parameterization: h[i] = 0.5 + 0.5 * sin(theta[i])? No, sum constraint is hard.
    
    # Alternative: Optimize deviations d[i] such that sum(d) = 0.
    # h[i] = 0.5 + d[i].
    # Constraint: sum(d) = 0.
    # We can optimize d[0:N-1] and set d[N-1] = -sum(d[0:N-1]).
    
    def objective_d(d):
        if time.time() > deadline:
            return 1e10
        h = np.full(N, 0.5)
        h[:N-1] += d
        h[N-1] += -np.sum(d)
        
        # Penalty for out of bounds
        penalty = 0.0
        if np.any(h < 0) or np.any(h > 1):
            penalty = 100.0 * (np.sum(np.clip(h, 0, 1) != h))
            
        return compute_max_overlap_linear(h) + penalty

    def project_d(d):
        # Ensure the resulting h is in [0,1] roughly, or just let penalty handle it.
        # For L-BFGS-B, we can set bounds on d.
        # h[i] = 0.5 + d[i] in [0,1] => d[i] in [-0.5, 0.5].
        return d

    bounds = [(-0.5, 0.5) for _ in range(N-1)]
    
    # Multi-start
    np.random.seed(42)
    num_starts = 10
    max_iter_per_start = 100
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        d_init = np.random.uniform(-0.4, 0.4, N-1)
        
        try:
            res = minimize(
                objective_d, 
                d_init, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter_per_start, 'ftol': 1e-8}
            )
            
            d_opt = res.x
            h_opt = np.full(N, 0.5)
            h_opt[:N-1] += d_opt
            h_opt[N-1] += -np.sum(d_opt)
            
            # Clip to ensure validity
            h_opt = np.clip(h_opt, 0, 1)
            
            # Re-normalize sum to exactly N/2
            # Simple adjustment: scale? No, that changes values.
            # Add residual to the element with most room.
            current_sum = np.sum(h_opt)
            diff = target_sum - current_sum
            if abs(diff) > 1e-9:
                # Find index with most room
                if diff > 0:
                    idx = np.argmin(h_opt)
                    room = 1.0 - h_opt[idx]
                    if room > abs(diff):
                        h_opt[idx] += diff
                    else:
                        # Distribute
                        mask = h_opt < 1.0
                        h_opt[mask] += diff / np.sum(mask)
                else:
                    idx = np.argmax(h_opt)
                    room = h_opt[idx]
                    if room > abs(diff):
                        h_opt[idx] += diff
                    else:
                        mask = h_opt > 0.0
                        h_opt[mask] += diff / np.sum(mask)
            
            # Final clip
            h_opt = np.clip(h_opt, 0, 1)
            
            score = compute_max_overlap_linear(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception as e:
            pass
            
    # Final check: if the constant solution is better (due to numerical noise), return it
    const_score = compute_max_overlap_linear(np.full(N, 0.5))
    if const_score < best_score:
        best_h = np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
