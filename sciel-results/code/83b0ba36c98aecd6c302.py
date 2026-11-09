import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 110
    
    # Coarse grid for filtering
    # Increased from 1024 to 2048 for better initial filtering
    M_coarse = 2048
    thetas_coarse = np.linspace(0, 2 * np.pi, M_coarse, endpoint=False)
    # Precompute Z_coarse for efficient dot products
    # Z_coarse shape: (M_coarse, N)
    # We need coeffs @ Z_coarse.T
    powers = np.arange(N)
    Z_coarse = np.exp(1j * thetas_coarse[:, None] * powers[None, :])
    
    # Fine grid for peak location identification in refinement
    M_fine = 5000
    thetas_fine = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
    Z_fine = np.exp(1j * thetas_fine[:, None] * powers[None, :])
    
    # Search space: fix c0=1, c1=1.
    # Free bits: c2...c23 (22 bits)
    total_combinations = 1 << 22
    chunk_size = 100000 # Increase chunk size for better vectorization efficiency
    
    # List to hold (score, coeffs)
    all_candidates = []
    
    # Helper function to evaluate sup norm with golden section refinement
    def evaluate_refined(c):
        # c is a numpy array of shape (N,) with values +/- 1
        # 1. Evaluate on fine grid
        vals = Z_fine @ c
        abs_vals = np.abs(vals)
        
        # Find indices of local maxima
        # We look for peaks. A simple way is to find global max and neighbors.
        # To be robust, find top few peaks on the fine grid.
        # Let's take the top 10 peaks on the fine grid.
        if M_fine > 10:
            # Find indices of largest values
            # argpartition is faster than full sort
            # We want the indices of the 10 largest values
            part_idx = np.argpartition(abs_vals, -10)[-10:]
            peak_indices = part_idx[np.argsort(abs_vals[part_idx])[::-1]]
        else:
            peak_indices = np.argsort(abs_vals)[::-1][:10]
            
        max_val = 0.0
        # Golden section search around each peak
        # The fine grid step is 2*pi/M_fine
        step = 2 * np.pi / M_fine
        
        for idx in peak_indices:
            # Define a window around the peak
            # The peak is at theta_idx. Neighbors are idx-1, idx+1.
            # We search in [theta_idx - step, theta_idx + step]
            # But to be safe, maybe a slightly wider window?
            # The function is smooth. The curvature is bounded.
            # Let's use [theta_idx - 1.5*step, theta_idx + 1.5*step]
            theta_center = thetas_fine[idx]
            a = (theta_center - 1.5 * step) % (2 * np.pi)
            b = (theta_center + 1.5 * step) % (2 * np.pi)
            
            # Handle wrap-around? 
            # Since we are on a circle, if a > b, we have a wrap.
            # Golden section search assumes a < b.
            # To simplify, we can shift the domain or handle the interval carefully.
            # Alternatively, since the peak is local, and step is small,
            # we can just search in [theta_center - 1.5*step, theta_center + 1.5*step]
            # and evaluate P(theta) directly, not relying on the grid array.
            
            def f(theta):
                # Evaluate P(z) at z = exp(i theta)
                # c @ exp(i theta * powers)
                z = np.exp(1j * theta * powers)
                val = np.sum(c * z)
                return np.abs(val)
            
            # Golden section search
            phi = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
            
            # Ensure a < b
            # If wrap around, we might have issues. 
            # Let's just define the interval linearly.
            # If a > b, add 2*pi to b.
            if a > b:
                b += 2 * np.pi
            
            c1 = b - phi * (b - a)
            c2 = a + phi * (b - a)
            f1 = f(c1)
            f2 = f(c2)
            
            # Iterate
            for _ in range(50):
                if f1 < f2:
                    b = c2
                    c2 = c1
                    f2 = f1
                    c1 = b - phi * (b - a)
                    f1 = f(c1)
                else:
                    a = c1
                    c1 = c2
                    f1 = f2
                    c2 = a + phi * (b - a)
                    f2 = f(c2)
                    
            # The maximum is at the midpoint of the final interval
            theta_max = (a + b) / 2
            val_max = f(theta_max)
            
            if val_max > max_val:
                max_val = val_max
                
        return max_val

    for start in range(0, total_combinations, chunk_size):
        if time.time() > deadline - 40: # Leave 40s for refinement
            break
            
        end = min(start + chunk_size, total_combinations)
        count = end - start
        
        # Generate indices
        indices = np.arange(start, end)
        
        # Generate bits for c2...c23
        shifts = np.arange(22)
        bits = (indices[:, None] >> shifts[None, :]) & 1
        
        # Convert to +1/-1
        c_rest = 1 - 2 * bits
        
        # Prepend c0=1, c1=1
        c01 = np.ones((count, 2))
        coeffs = np.hstack((c01, c_rest))
        
        # Evaluate on coarse grid
        vals = coeffs @ Z_coarse.T
        max_norms = np.max(np.abs(vals), axis=1)
        
        # Find top K (smallest norms) in this chunk
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
            
    # If we ran out of time, we might have few candidates.
    # Sort all_candidates by coarse score (ascending)
    all_candidates.sort(key=lambda x: x[0])
    
    # Take top 50 for refinement
    top_refine = all_candidates[:50]
    
    best_c = None
    best_score = np.inf
    
    for score, c in top_refine:
        if time.time() > deadline - 5:
            break
            
        # Evaluate with refined method
        fine_score = evaluate_refined(c)
        
        # Local Search: Try flipping each bit
        # If time permits, do a simple local search for the top few
        if fine_score < best_score:
            best_score = fine_score
            best_c = c.copy()
            
            # Simple local search: try flipping each bit i in 2..23
            # If it improves, keep it.
            # This is O(N) evaluations per candidate.
            # 50 candidates * 24 bits * (cost of evaluate_refined)
            # evaluate_refined is ~50 golden steps * 24 exps. ~1200 ops.
            # 50 * 24 * 1200 = 1.4M ops. Should be fast in Python if vectorized?
            # No, evaluate_refined is a loop.
            # Let's limit local search to the very top 5 candidates.
            
            if len(top_refine) < 5 or top_refine.index((score, c)) < 5:
                improved = True
                while improved:
                    improved = False
                    for i in range(2, N):
                        c_trial = c.copy()
                        c_trial[i] *= -1
                        score_trial = evaluate_refined(c_trial)
                        if score_trial < best_score - 1e-9:
                            best_score = score_trial
                            best_c = c_trial
                            improved = True
                            break # Restart loop as best_c changed
                            
    # Fallback if no candidates or error
    if best_c is None:
        best_c = np.ones(N, dtype=int)
    else:
        best_c = best_c.astype(int)
        
    return [int(x) for x in best_c]
