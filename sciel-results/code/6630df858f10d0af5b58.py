import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 110
    
    # Coarse grid for filtering
    M_coarse = 2048
    thetas_coarse = np.linspace(0, 2 * np.pi, M_coarse, endpoint=False)
    powers = np.arange(N)
    Z_coarse = np.exp(1j * thetas_coarse[:, None] * powers[None, :])
    
    # Fine grid for peak location identification in refinement
    M_fine = 5000
    thetas_fine = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
    Z_fine = np.exp(1j * thetas_fine[:, None] * powers[None, :])
    
    # Precompute for fast evaluation: use real and imaginary parts
    # For a given theta, P(z) = sum c_k * exp(i*k*theta)
    # We can precompute cos(k*theta) and sin(k*theta) arrays
    
    # Search space: fix c0=1, c1=1.
    # Free bits: c2...c23 (22 bits)
    total_combinations = 1 << 22
    chunk_size = 100000
    
    # List to hold (score, coeffs)
    all_candidates = []
    
    # Fast evaluation: precompute trig tables for golden section
    # We'll use a finer precomputed table for evaluation
    M_eval = 100000
    thetas_eval = np.linspace(0, 2 * np.pi, M_eval, endpoint=False)
    # Precompute cos and sin for all powers at all eval points
    # This is too large (100000 * 24), so we'll compute on the fly but efficiently
    
    def eval_at_theta(theta, c):
        """Evaluate |P(exp(i*theta))| for coefficient vector c."""
        # Vectorized: compute exp(i*theta*powers) and dot with c
        z = np.exp(1j * theta * powers)
        return np.abs(np.dot(c, z))
    
    def evaluate_refined(c):
        """Evaluate sup norm with golden section refinement. Returns max |P(z)| on |z|=1."""
        # 1. Evaluate on fine grid
        vals = Z_fine @ c
        abs_vals = np.abs(vals)
        
        # Find top 10 peaks
        if M_fine > 10:
            part_idx = np.argpartition(abs_vals, -10)[-10:]
            peak_indices = part_idx[np.argsort(abs_vals[part_idx])[::-1]]
        else:
            peak_indices = np.argsort(abs_vals)[::-1][:10]
            
        max_val = 0.0
        step = 2 * np.pi / M_fine
        phi = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
        
        for idx in peak_indices:
            theta_center = thetas_fine[idx]
            # Search in a window around the peak
            a = theta_center - 1.5 * step
            b = theta_center + 1.5 * step
            
            # Ensure a < b (they should be, but handle edge cases)
            if a > b:
                a, b = b, a
            
            c1 = b - phi * (b - a)
            c2 = a + phi * (b - a)
            f1 = eval_at_theta(c1, c)
            f2 = eval_at_theta(c2, c)
            
            # Golden section iteration
            for _ in range(40):
                if f1 < f2:
                    b = c2
                    c2 = c1
                    f2 = f1
                    c1 = b - phi * (b - a)
                    f1 = eval_at_theta(c1, c)
                else:
                    a = c1
                    c1 = c2
                    f1 = f2
                    c2 = a + phi * (b - a)
                    f2 = eval_at_theta(c2, c)
                    
            theta_max = (a + b) / 2
            val_max = eval_at_theta(theta_max, c)
            
            if val_max > max_val:
                max_val = val_max
                
        return max_val

    for start in range(0, total_combinations, chunk_size):
        if time.time() > deadline - 40:
            break
            
        end = min(start + chunk_size, total_combinations)
        count = end - start
        
        indices = np.arange(start, end)
        
        shifts = np.arange(22)
        bits = (indices[:, None] >> shifts[None, :]) & 1
        c_rest = 1 - 2 * bits
        c01 = np.ones((count, 2))
        coeffs = np.hstack((c01, c_rest))
        
        vals = coeffs @ Z_coarse.T
        max_norms = np.max(np.abs(vals), axis=1)
        
        K = 100
        if count > K:
            part_idx = np.argpartition(max_norms, K)[:K]
            top_scores = max_norms[part_idx]
            top_coeffs = coeffs[part_idx]
        else:
            top_scores = max_norms
            top_coeffs = coeffs
            
        for i in range(len(top_scores)):
            all_candidates.append((top_scores[i], top_coeffs[i]))
            
    all_candidates.sort(key=lambda x: x[0])
    
    # Take top 80 for refinement (increased from 50)
    top_refine = all_candidates[:80]
    
    best_c = None
    best_score = np.inf
    
    for rank, (score, c) in enumerate(top_refine):
        if time.time() > deadline - 5:
            break
            
        fine_score = evaluate_refined(c)
        
        if fine_score < best_score:
            best_score = fine_score
            best_c = c.copy()
            
            # Local search for top 10 candidates
            if rank < 10:
                improved = True
                while improved:
                    improved = False
                    for i in range(2, N):
                        c_trial = c.copy()
                        c_trial[i] *= -1
                        score_trial = evaluate_refined(c_trial)
                        if score_trial < best_score - 1e-9:
                            best_score = score_trial
                            best_c = c_trial.copy()
                            c = c_trial
                            improved = True
                            break
                            
    if best_c is None:
        best_c = np.ones(N, dtype=int)
    else:
        best_c = best_c.astype(int)
        
    return [int(x) for x in best_c]
