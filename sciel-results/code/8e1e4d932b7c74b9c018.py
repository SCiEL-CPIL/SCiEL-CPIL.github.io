"""
Strategy: Deterministic Vectorized Ratchet Local Search.

1. Generate a diverse set of initial candidates using deterministic patterns 
   (alternating, blocks, Goldbach-like structures) and a fixed-seed PRNG.
2. For each candidate, perform an exhaustive local search using a "Ratchet" mechanism:
   - Evaluate the sup-norm of the current polynomial.
   - Evaluate the sup-norm of ALL possible 1-bit flips simultaneously using vectorization.
   - Evaluate the sup-norm of ALL possible 2-bit flips simultaneously using vectorization.
   - Move to the neighbor with the lowest sup-norm if it is an improvement.
   - Repeat until no improvement is found or time limit is reached.
3. This approach is structurally different because it is deterministic, 
   exhaustively explores the local neighborhood in a single vectorized batch, 
   and avoids the stochastic tunneling of SA which can get stuck or waste time.
"""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    N = 24
    n_samples = 4096  # Higher resolution for better peak detection
    
    # Pre-compute the evaluation grid
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    z = np.exp(1j * theta)
    
    # Z_powers[k] = z^k for k=0..N-1
    # Shape: (N, n_samples)
    Z_powers = np.zeros((N, n_samples), dtype=complex)
    Z_powers[0] = 1.0
    for k in range(1, N):
        Z_powers[k] = Z_powers[k-1] * z

    def compute_sup_norm(c):
        """
        Compute the sup norm of P(z) = sum c_k z^k.
        c: array of shape (N,) with values +1/-1.
        Returns the maximum absolute value on the grid.
        """
        # vals = c @ Z_powers
        # c shape (N,), Z_powers shape (N, M) -> vals shape (M,)
        vals = np.dot(c, Z_powers)
        return np.max(np.abs(vals))

    def refine_peak(c, peak_idx, z_grid, Z_powers, n_samples):
        """
        Refine the peak at peak_idx using golden section search.
        Returns the refined maximum value.
        """
        # Get the interval around peak_idx
        # The peak is between theta[peak_idx-1] and theta[peak_idx+1]
        step = 2 * np.pi / n_samples
        a = theta[(peak_idx - 1) % n_samples]
        b = theta[(peak_idx + 1) % n_samples]
        if a > b:
            b += 2 * np.pi
        
        # Local evaluation function
        def P_abs(t):
            z_val = np.exp(1j * t)
            val = 0.0 + 0.0j
            zp = 1.0 + 0.0j
            for k in range(N):
                val += c[k] * zp
                zp *= z_val
            return np.abs(val)

        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        inv_gr = 1 / gr
        x1 = a + (b - a) * inv_gr
        x2 = a + (b - a) * gr
        f1 = P_abs(x1)
        f2 = P_abs(x2)
        
        # Iterate until interval is small enough
        for _ in range(40):
            if f1 < f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + (b - a) * gr
                f2 = P_abs(x2)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + (b - a) * inv_gr
                f1 = P_abs(x1)
                
            if b - a < 1e-11:
                break
                
        return P_abs((x1 + x2) / 2)

    def get_best_neighbor(c, current_val):
        """
        Find the best 1-bit or 2-bit flip.
        Returns (new_coeffs, new_val) if improvement found, else (None, None).
        """
        # 1. Evaluate all 1-bit flips
        # c shape (N,)
        # For each j, new_c = c.copy(); new_c[j] *= -1
        # P_new = P_old - 2*c[j]*z^j
        # We can compute P_old once, then subtract contributions.
        
        vals_old = np.dot(c, Z_powers) # Shape (n_samples,)
        
        best_val = current_val
        best_c = None
        
        # 1-bit flips
        # We want to find j that minimizes max(|vals_old - 2*c[j]*Z_powers[j]|)
        # Vectorize: 
        # deltas_1[j, k] = -2 * c[j] * Z_powers[j, k]
        # Shape of deltas_1: (N, n_samples)
        # c_col: (N, 1)
        c_col = c[:, np.newaxis]
        deltas_1 = -2.0 * c_col * Z_powers  # (N, n_samples)
        
        # new_vals_1[j, k] = vals_old[k] + deltas_1[j, k]
        # Broadcasting: vals_old (1, M) + deltas_1 (N, M) -> (N, M)
        new_vals_1 = vals_old[np.newaxis, :] + deltas_1
        abs_new_1 = np.abs(new_vals_1)
        max_new_1 = np.max(abs_new_1, axis=1) # Shape (N,)
        
        min_idx_1 = np.argmin(max_new_1)
        min_val_1 = max_new_1[min_idx_1]
        
        if min_val_1 < best_val:
            best_val = min_val_1
            best_c = c.copy()
            best_c[min_idx_1] *= -1
            # We should refine this peak to be accurate
            # Find peak index for this candidate
            # For efficiency in the loop, we might skip refinement for every single step,
            # but the "Ratchet" needs accurate comparisons.
            # Let's refine the peak of the best candidate.
            # To find the peak index, we can just take the argmax of abs_new_1[min_idx_1]
            peak_idx = np.argmax(abs_new_1[min_idx_1])
            refined_val = refine_peak(best_c, peak_idx, z, Z_powers, n_samples)
            if refined_val < best_val:
                best_val = refined_val
            # Note: best_c is already the candidate. The refined_val is the true score.
        
        # 2. Evaluate all 2-bit flips
        # This is more expensive. N*(N-1)/2 = 276 candidates.
        # We can vectorize this too.
        # For pair (j, k), new_c has j and k flipped.
        # P_new = P_old - 2*c[j]*z^j - 2*c[k]*z^k
        
        # We can construct the deltas for all pairs.
        # However, memory might be an issue if we store (276, 4096) complex numbers.
        # 276 * 4096 * 16 bytes ~ 18 MB. This is acceptable.
        
        # Let's iterate over j and vectorize over k > j to save memory/time overhead of full matrix construction?
        # Or just do it in chunks?
        # Actually, 276 is small. Let's just loop over j and vectorize over k.
        
        for j in range(N):
            # For fixed j, consider all k > j
            k_indices = np.arange(j+1, N)
            if len(k_indices) == 0: continue
            
            # c[k_indices] shape (num_k,)
            # Z_powers[k_indices] shape (num_k, n_samples)
            # deltas_k = -2 * c[k_indices][:, np.newaxis] * Z_powers[k_indices]
            # Shape: (num_k, n_samples)
            
            # Base delta for j: -2 * c[j] * Z_powers[j]
            # Shape: (n_samples,)
            delta_j = -2.0 * c[j] * Z_powers[j]
            
            # new_vals = vals_old + delta_j + deltas_k
            # vals_old shape (1, M), delta_j shape (1, M), deltas_k shape (num_k, M)
            
            deltas_k = -2.0 * c[k_indices][:, np.newaxis] * Z_powers[k_indices]
            
            new_vals_2 = vals_old[np.newaxis, :] + delta_j[np.newaxis, :] + deltas_k
            abs_new_2 = np.abs(new_vals_2)
            max_new_2 = np.max(abs_new_2, axis=1) # Shape (num_k,)
            
            min_idx_in_k = np.argmin(max_new_2)
            min_val_2 = max_new_2[min_idx_in_k]
            
            if min_val_2 < best_val:
                k = k_indices[min_idx_in_k]
                best_val = min_val_2
                best_c = c.copy()
                best_c[j] *= -1
                best_c[k] *= -1
                
                # Refine peak
                peak_idx = np.argmax(abs_new_2[min_idx_in_k])
                refined_val = refine_peak(best_c, peak_idx, z, Z_powers, n_samples)
                if refined_val < best_val:
                    best_val = refined_val
        
        return best_c, best_val

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Alternating
    c_alt = np.array([1, -1] * 12)
    candidates.append(c_alt)
    
    # 2. Blocks of 2
    c_block2 = np.array([1, 1, -1, -1] * 6)
    candidates.append(c_block2)
    
    # 3. Blocks of 4
    c_block4 = np.array([1, 1, 1, 1, -1, -1, -1, -1] * 3)
    candidates.append(c_block4)
    
    # 4. Random with fixed seed
    rng = np.random.RandomState(42)
    for _ in range(20):
        c_rand = rng.choice([-1, 1], size=N)
        candidates.append(c_rand)
        
    # 5. Specific known good patterns (from Odlyzko tables if I recall any, or just heuristic)
    # The seed program had one. Let's include it.
    c_seed = np.array([1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])
    candidates.append(c_seed)
    
    best_overall_c = None
    best_overall_val = np.inf
    
    for c0 in candidates:
        if time.time() > deadline - 5.0:
            break
            
        c = c0.copy()
        # Initial evaluation with refinement
        vals = np.dot(c, Z_powers)
        peak_idx = np.argmax(np.abs(vals))
        val = refine_peak(c, peak_idx, z, Z_powers, n_samples)
        
        if val < best_overall_val:
            best_overall_val = val
            best_overall_c = c.copy()
            
        # Ratchet Loop
        # We perform local search until no improvement is found.
        # To prevent getting stuck in a very deep local minimum that is not global,
        # we could restart with a perturbation, but the "Ratchet" is greedy.
        # The diversity of seeds handles the multi-start aspect.
        
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                break
                
            new_c, new_val = get_best_neighbor(c, val)
            
            if new_c is not None and new_val < val:
                c = new_c
                val = new_val
                improved = True
                if val < best_overall_val:
                    best_overall_val = val
                    best_overall_c = c.copy()
            else:
                improved = False
                
        # If we have time, we can do a "kick" (random flip) and continue ratcheting
        # But the deterministic nature is the main selling point.
        # Let's add a simple kick mechanism to escape local minima for the best candidate found so far?
        # No, let's stick to the multi-start deterministic ratchet.

    if best_overall_c is None:
        # Fallback
        return [1] * N
        
    return best_overall_c.tolist()
# EVOLVE-BLOCK-END
