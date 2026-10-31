"""Erdos minimum overlap - optimized step heights to minimize max overlap."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    # Prevent overflow in exp
    x_shifted = x - x_max
    return x_max + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * x_shifted)))

def _discrete_overlap_max(h, n_shifts=50):
    """
    Fast approximation of the maximum overlap.
    The overlap for a shift s is sum_i h[i] * (1 - h[i + k]) * width, where k depends on s.
    For step functions, the overlap function is piecewise linear.
    The maximum is likely to occur at shifts that align the steps, i.e., s = m * (2/N).
    We evaluate the overlap at these discrete shifts.
    """
    step_w = 2.0 / N
    # Shifts s = k * step_w for k in range(-N, N)
    # Overlap for shift k steps: sum_{i} h[i] * (1 - h[i+k]) * step_w
    # Note: h is defined on [0, 2]. h[x] is 0 outside.
    # h[i+k] is valid if 0 <= i+k < N.
    
    max_ov = 0.0
    # We only need to check shifts that bring significant mass into overlap.
    # Actually, checking all N*2 shifts is O(N^2) which is 10,000 ops. Very fast.
    # Let's compute all pairwise overlaps? No, just the max.
    
    # Vectorized approach for a specific shift k:
    # h_shifted = roll(h, k)
    # overlap = sum(h * (1 - h_shifted) * step_w)
    # But we need to handle boundaries (zero padding).
    
    # Let's iterate over possible integer shifts k from -N to N
    # This covers all "aligned" shifts. The max of the continuous overlap
    # for step functions occurs at these boundaries because the function is linear between them.
    
    h_padded = np.pad(h, (N, N), mode='constant', constant_values=0)
    
    for k in range(-N, N + 1):
        # h(x+s) corresponds to h shifted by -k steps in index?
        # If s = k * step_w, then h(x+s) at step i is h[i+k].
        # We want integral of h(x) * (1 - h(x+s)).
        # h(x) is h[i]. h(x+s) is h[i+k].
        # Product is h[i] * (1 - h[i+k]).
        # Sum over i where both are non-zero? No, h is 0 outside, so product is 0.
        
        # Extract h[i] for i in 0..N-1
        # Extract h[i+k] for i in 0..N-1. If i+k out of bounds, value is 0.
        
        h1 = h
        # h2 is h shifted by -k (so h2[i] = h[i+k])
        # If k > 0, h2[0:k] are from negative indices (0), h2[k:N+k] from h[0:N].
        # If k < 0, h2[-k:] are from indices > N-1 (0).
        
        h2 = np.zeros(N)
        if k >= 0:
            h2[k:] = h[:N-k]
        else:
            h2[:N+k] = h[-k:]
            
        ov = np.sum(h1 * (1.0 - h2)) * step_w
        if ov > max_ov:
            max_ov = ov
            
    return max_ov

def _true_overlap_max(h, n_eval=100):
    """
    More accurate check: sample shifts uniformly and compute overlap.
    This is safer than just integer shifts if the optimal solution has non-integer
    alignment (though for step functions, linear segments imply max at vertices).
    However, the "vertices" of the overlap function C(s) are at s = (i-j)*step_w.
    So integer shifts are sufficient for the *maximum* value.
    Let's stick to _discrete_overlap_max for speed and accuracy on step functions.
    """
    return _discrete_overlap_max(h)

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    # Initial guess: The "Golden Ratio" or similar block structure often works well.
    # From inspiration: A = (3 - sqrt(5))/2 ~ 0.382, B = 1 - A ~ 0.618
    # Or simply alternating 0 and 1? No, sum must be N/2.
    # Let's try a block of A and a block of B.
    
    A = (3.0 - np.sqrt(5.0)) / 2.0
    B = 1.0 - A
    
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    
    # Ensure exact sum
    current_sum = np.sum(h0)
    diff = N / 2.0 - current_sum
    h0 += diff / N
    h0 = np.clip(h0, 0, 1)
    
    best_h = h0.copy()
    best_score = _true_overlap_max(best_h)
    
    # Optimization Objective
    # We want to minimize the max overlap.
    # Surrogate: smooth_max of local overlaps h_i(1-h_{i+1})?
    # The discrete overlap for shift 1 step is sum h_i(1-h_{i+1}) step_w.
    # The max over shifts is the max of these sums? No, the sum changes.
    # Actually, C(k) = sum h_i (1 - h_{i+k}) step_w.
    # We want min_k_max C(k).
    # This is a min-max problem.
    # Surrogate: smooth_max over k of C(k).
    
    def objective(h):
        # Compute C(k) for all k
        # Vectorized:
        # C(k) = step_w * sum_i h[i] - h[i]h[i+k]
        # Sum h[i] is constant (N/2).
        # So minimizing C(k) is equivalent to maximizing sum h[i]h[i+k] for the worst case k?
        # No, C(k) = S - H(k), where S = step_w * sum h = 1.0 (integral is 1).
        # Wait, integral of h is 1. So sum h[i] * step_w = 1 => sum h[i] = N/2.
        # C(k) = sum h[i](1-h[i+k]) step_w = step_w * (sum h[i] - sum h[i]h[i+k])
        # C(k) = 1.0 - step_w * sum h[i]h[i+k]
        # Minimizing max C(k) is equivalent to maximizing min_k (sum h[i]h[i+k]).
        
        # Let's compute autocorrelation-like terms.
        # For each k, compute dot product of h and shifted h.
        
        # To be efficient, we can compute the max overlap directly in the objective.
        # But smooth_max needs a vector of values.
        # Let's compute C(k) for all k.
        
        # h is length N.
        # We need sum h[i]h[i+k].
        
        # Using FFT or just loops? N=100, loops are fine.
        # But we need it to be differentiable for SLSQP.
        
        # Let's use the local approximation:
        # The "worst" overlap is often related to local variations.
        # Let's try minimizing the smooth max of the local terms h_i(1-h_{i+1}) and h_i(1-h_{i-1})?
        
        # Actually, let's just use the discrete C(k) values.
        # There are 2N+1 values.
        
        h_padded = np.pad(h, (N, N), mode='constant', constant_values=0)
        
        c_vals = np.zeros(2 * N + 1)
        for idx, k in enumerate(range(-N, N + 1)):
            if k >= 0:
                h2 = np.zeros(N)
                h2[k:] = h[:N-k]
            else:
                h2 = np.zeros(N)
                h2[:N+k] = h[-k:]
            # C(k) = 1.0 - step_w * sum(h * h2)
            # sum(h * h2) is dot product
            dot = np.dot(h, h2)
            c_vals[idx] = 1.0 - (2.0 / N) * dot
            
        # We want to minimize the max of c_vals.
        # Use smooth max.
        return _smooth_max(c_vals, alpha=50.0)

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]
    bounds = [(0.0, 1.0)] * N
    
    # Run optimization
    try:
        # Use SLSQP for constraint handling
        result = minimize(
            objective,
            h0.copy(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
        )
        
        if result.success or result.fun < best_score:
            h_opt = result.x.copy()
            # Enforce exact sum
            diff = N / 2.0 - np.sum(h_opt)
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            
            # Verify with true score
            score_opt = _true_overlap_max(h_opt)
            if score_opt < best_score:
                best_h = h_opt.copy()
                best_score = score_opt
                
    except Exception:
        pass
    
    # Local refinement / Polish
    # If time permits, try a few random restarts or perturbations
    
    if time.time() < deadline - 10:
        np.random.seed(42)
        for i in range(10):
            if time.time() > deadline - 5:
                break
            
            # Perturb best_h
            h_init = best_h.copy()
            # Add small noise
            noise = np.random.normal(0, 0.05, N)
            h_init += noise
            h_init = np.clip(h_init, 0, 1)
            # Fix sum
            diff = N / 2.0 - np.sum(h_init)
            h_init += diff / N
            h_init = np.clip(h_init, 0, 1)
            
            try:
                result = minimize(
                    objective,
                    h_init,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
                )
                
                if result.fun < best_score:
                    h_opt = result.x.copy()
                    diff = N / 2.0 - np.sum(h_opt)
                    h_opt += diff / N
                    h_opt = np.clip(h_opt, 0, 1)
                    
                    score_opt = _true_overlap_max(h_opt)
                    if score_opt < best_score:
                        best_h = h_opt.copy()
                        best_score = score_opt
                        
            except Exception:
                continue

    # Final check
    # Ensure exact sum one last time
    diff = N / 2.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
