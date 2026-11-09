"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    N = 24
    
    # Precompute evaluation grid for fast sup-norm estimation
    # Use a moderately dense grid for fast local search steps
    num_points_fast = 8192
    theta_fast = np.linspace(0, 2 * np.pi, num_points_fast, endpoint=False)
    k_indices = np.arange(N)
    Z_fast = np.exp(1j * np.outer(k_indices, theta_fast))  # (N, num_points_fast)
    
    # High resolution grid for final verification / peak identification if needed
    num_points_high = 65536
    theta_high = np.linspace(0, 2 * np.pi, num_points_high, endpoint=False)
    Z_high = np.exp(1j * np.outer(k_indices, theta_high))
    
    def sup_norm_fast(coeffs):
        """Compute sup norm of P(z) on the unit circle using fast grid."""
        # coeffs: shape (N,)
        # P(z_j) = sum_k coeffs[k] * z_j^k = coeffs @ Z_fast[:, j]
        vals = Z_fast.T @ coeffs  # shape (num_points_fast,)
        return np.max(np.abs(vals))
    
    def refine_peak(coeffs, peak_theta, num_fine=200):
        """Refine the peak value near peak_theta using golden-section-like search."""
        best_val = 0.0
        best_theta = peak_theta
        
        # Multi-scale refinement
        scales = [1e-3, 1e-4, 1e-5, 1e-6]
        current_center = peak_theta
        current_radius = 1e-3
        
        for scale in scales:
            t_range = np.linspace(current_center - scale, current_center + scale, num_fine)
            z_fine = np.exp(1j * t_range)
            # Evaluate P at these points using Horner's method for speed
            # P(z) = c_0 + c_1*z + c_2*z^2 + ... + c_{N-1}*z^{N-1}
            # Build powers incrementally
            powers = np.ones((num_fine, N), dtype=complex)
            powers[:, 0] = 1.0
            for k in range(1, N):
                powers[:, k] = powers[:, k-1] * z_fine
            vals_fine = powers @ coeffs
            abs_fine = np.abs(vals_fine)
            local_max_idx = np.argmax(abs_fine)
            if abs_fine[local_max_idx] > best_val:
                best_val = abs_fine[local_max_idx]
                best_theta = t_range[local_max_idx]
            current_center = best_theta
            current_radius = scale * 0.5
        
        return best_val, best_theta
    
    def evaluate_refined(coeffs):
        """Evaluate score with peak refinement using high-res grid for peak finding."""
        vals = Z_high.T @ coeffs
        abs_vals = np.abs(vals)
        
        # Find top 3 peaks to refine, as the global max might be close to a local max
        # that is slightly lower on the grid but higher in reality, or vice versa.
        # However, usually the global max is the highest on a dense grid.
        # Let's just refine the highest one, and maybe the second highest if close.
        
        # Sort indices by value
        # np.argsort is expensive on 65536 elements. 
        # Just take the max.
        max_idx = np.argmax(abs_vals)
        max_theta = theta_high[max_idx]
        
        # Refine this peak
        refined_val, _ = refine_peak(coeffs, max_theta)
        
        # Check if there are other significant peaks
        # We can check the max value excluding the neighborhood of max_idx
        # But for speed, let's just refine the top 2 distinct peaks if possible.
        
        # Simple heuristic: refine the highest peak.
        # If we have time, we could refine the top 3.
        # Let's stick to top 1 for now to save time in the loop, 
        # but ensure refine_peak is accurate.
        
        return refined_val / np.sqrt(N)
    
    def evaluate_fast(coeffs):
        """Fast evaluation without refinement (for initial screening)."""
        return sup_norm_fast(coeffs) / np.sqrt(N)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Golay-Rudin-Shapiro polynomial
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    gsr = np.array(p[:N])
    candidates.append(gsr)
    
    # 2. Exhaustive search over self-reciprocal polynomials (c_k = c_{N-1-k})
    # This reduces 24 bits to 12 bits. 2^12 = 4096 candidates.
    # We evaluate all of them to find the best in this subfamily.
    half_N = N // 2
    best_sr_score = float('inf')
    best_sr_coeffs = None
    
    # Precompute the Z matrix part for self-reciprocal?
    # Actually, just use the existing fast evaluator.
    # 4096 evaluations * (N * num_points) might be slow if num_points is large.
    # Z.T @ coeffs is a matrix-vector product.
    # Let's batch them or just loop. 4096 * 65536 * 24 ops is ~ 6.5 * 10^9 ops.
    # This might be too slow in Python loop.
    # Let's use a smaller grid for this enumeration or vectorize.
    
    # Vectorized enumeration:
    # Create a matrix of all 4096 self-reciprocal polynomials.
    # Shape: (4096, 24)
    # Bits 0..11 determine the polynomial.
    # coeffs[0..11] are the bits. coeffs[12..23] are the reverse.
    
    # Generate all 12-bit patterns
    patterns = np.array([list(map(int, np.binary_repr(i, width=half_N))) for i in range(2**half_N)], dtype=float)
    # Convert 0/1 to -1/1
    patterns = 2 * patterns - 1
    
    # Construct full polynomials
    # First half: patterns
    # Second half: reverse of patterns
    full_sr = np.hstack([patterns, patterns[:, ::-1]])
    
    # Evaluate all at once?
    # Z.T is (num_points, N). full_sr is (4096, N).
    # We want (4096, num_points) values.
    # vals = full_sr @ Z  -> (4096, N) @ (N, num_points) = (4096, num_points)
    # This is a large matrix multiplication: 4096 * 65536 complex numbers.
    # 4096 * 65536 * 16 bytes = 4 GB. Too much memory.
    
    # We must loop or use a smaller grid for this step.
    # Let's use a smaller grid (e.g., 4096 points) for the enumeration screening.
    num_points_screen = 4096
    theta_screen = np.linspace(0, 2 * np.pi, num_points_screen, endpoint=False)
    k_indices_screen = np.arange(N)
    Z_screen = np.exp(1j * np.outer(k_indices_screen, theta_screen))
    
    # Batch size to avoid memory issues
    batch_size = 512
    for i in range(0, 2**half_N, batch_size):
        batch = full_sr[i:i+batch_size]
        # batch @ Z_screen -> (batch_size, num_points_screen)
        vals = batch @ Z_screen
        max_vals = np.max(np.abs(vals), axis=1)
        scores = max_vals / np.sqrt(N)
        
        # Find best in this batch
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_sr_score:
            best_sr_score = scores[min_idx]
            best_sr_coeffs = batch[min_idx].copy()
            
    if best_sr_coeffs is not None:
        candidates.append(best_sr_coeffs)
        
    # Also add the GRS and its sign flip / reverse variants
    candidates.append(-gsr)
    candidates.append(gsr[::-1])
    candidates.append(-gsr[::-1])
    
    # 3. Several random candidates with different seeds
    np.random.seed(42)
    for _ in range(100): # Increased random candidates
        r = np.array([1 if np.random.rand() < 0.5 else -1 for _ in range(N)])
        candidates.append(r)
    
    best_score = float('inf')
    best_coeffs = None
    
    # First pass: fast screening to find good candidates
    screened = []
    for cand in candidates:
        if time.time() > deadline - 15.0:
            break
        s = evaluate_fast(cand)
        screened.append((s, cand))
    
    # Sort by fast score and take top ones for detailed refinement
    screened.sort(key=lambda x: x[0])
    top_candidates = [cand for _, cand in screened[:20]] # Increased top candidates
    
    # Local Search (Hill Climbing) on top candidates
    for i, cand in enumerate(top_candidates):
        if time.time() > deadline - 5.0:
            break
            
        current = cand.copy().astype(float)
        current_score = evaluate_fast(current) # Start with fast score for comparison
        
        improved = True
        max_iterations = 100 # Increased iterations due to faster eval
        iteration = 0
        while improved and time.time() < deadline - 5.0 and iteration < max_iterations:
            improved = False
            iteration += 1
            # Try flipping each bit in random order
            indices = np.random.permutation(N)
            for k in indices:
                if time.time() > deadline - 5.0:
                    break
                # Flip bit k
                old_val = current[k]
                current[k] = -old_val
                new_score = evaluate_fast(current)  # Use fast eval during search
                
                if new_score < current_score - 1e-12: # Tighter threshold for fast eval
                    current_score = new_score
                    improved = True
                else:
                    current[k] = old_val
            
        # Final refined evaluation
        final_score = evaluate_refined(current)
        if final_score < best_score:
            best_score = final_score
            best_coeffs = current.copy()
    
    # If we have time, do more random restarts from the best found
    if best_coeffs is not None and time.time() < deadline - 10.0:
        # Increase restarts since eval is faster
        for restart in range(50): 
            if time.time() > deadline - 5.0:
                break
            # Start from a perturbation of the best
            current = best_coeffs.copy().astype(float)
            # Flip 1-3 random bits for finer exploration
            num_flips = np.random.randint(1, 4)
            flip_indices = np.random.choice(N, num_flips, replace=False)
            for k in flip_indices:
                current[k] = -current[k]
            
            current_score = evaluate_fast(current)
            improved = True
            max_iterations = 50
            iteration = 0
            while improved and time.time() < deadline - 5.0 and iteration < max_iterations:
                improved = False
                iteration += 1
                indices = np.random.permutation(N)
                for k in indices:
                    if time.time() > deadline - 5.0:
                        break
                    old_val = current[k]
                    current[k] = -old_val
                    new_score = evaluate_fast(current)
                    
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        improved = True
                    else:
                        current[k] = old_val
                
                # Check if better than current best
                if current_score < best_score - 1e-10:
                    refined = evaluate_refined(current)
                    if refined < best_score:
                        best_score = refined
                        best_coeffs = current.copy()
    
    # If best_coeffs is None (shouldn't happen), return GRS
    if best_coeffs is None:
        return [int(x) for x in gsr]
        
    return [int(x) for x in best_coeffs]
# EVOLVE-BLOCK-END
