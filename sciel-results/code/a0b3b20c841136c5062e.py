"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Initial Candidate: Truncated Golay-Rudin-Shapiro
    # This is a known good starting point for low peak-to-average ratio polynomials.
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    
    # 2. Fast Sup-Norm Evaluation
    # We use a dense grid for initial search, then refine.
    # For speed, we'll use a moderate grid size for the main loop.
    n_points = 2048
    angles = 2 * np.pi * np.arange(n_points) / n_points
    zs = np.exp(1j * angles)
    
    # Pre-compute the power matrix for the fixed grid
    # z_j^k for j in 0..N-1, k in 0..23
    # Shape: (N, 24)
    powers = np.ones((n_points, 24), dtype=complex)
    for k in range(1, 24):
        powers[:, k] = powers[:, k-1] * zs
        
    def fast_score(coeffs):
        vals = powers @ coeffs
        return np.max(np.abs(vals))
    
    def full_score(coeffs):
        # Higher accuracy score using denser grid + golden section refinement
        # This is expensive, so we only use it for final verification or occasional checks.
        n_dense = 20000
        angles_dense = 2 * np.pi * np.arange(n_dense) / n_dense
        zs_dense = np.exp(1j * angles_dense)
        
        # Evaluate on dense grid
        powers_dense = np.ones((n_dense, 24), dtype=complex)
        for k in range(1, 24):
            powers_dense[:, k] = powers_dense[:, k-1] * zs_dense
            
        vals = powers_dense @ coeffs
        abs_vals = np.abs(vals)
        
        # Find top peaks and refine
        # Get indices of top 10 peaks
        top_k = 10
        indices = np.argsort(abs_vals)[-top_k:][::-1]
        
        max_val = 0
        for idx in indices:
            # Golden section refinement around angles_dense[idx]
            theta_center = angles_dense[idx]
            # Search in a small window
            window = 2 * np.pi / n_dense
            a = theta_center - window
            b = theta_center + window
            
            # Golden ratio
            gr = (math.sqrt(5) - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            def f(theta):
                z = np.exp(1j * theta)
                # Compute P(z) = sum c_k z^k
                # Using Horner's rule or direct sum
                # Direct sum is fine for 24 terms
                val = 0
                zk = 1
                for k in range(24):
                    val += coeffs[k] * zk
                    zk *= z
                return abs(val)
                
            fc = f(c)
            fd = f(d)
            
            for _ in range(50):
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = f(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = f(d)
                    
            max_val = max(max_val, fc, fd)
            
        return max_val / math.sqrt(24)

    best_val = fast_score(best_coeffs) / math.sqrt(24)
    
    # 3. Deterministic "Charge Repulsion" Heuristic
    # Strategy: 
    # 1. Evaluate P(z) on the grid.
    # 2. Identify points where |P(z)| is large (peaks).
    # 3. For each coefficient c_k, calculate the "force" or "repulsion" it contributes to these peaks.
    # 4. Flip the coefficient that most reduces the total "energy" (sum of squared magnitudes at peaks).
    
    current_coeffs = best_coeffs.copy()
    current_val = fast_score(current_coeffs) / math.sqrt(24)
    
    # We will iterate, trying to flip each coefficient if it improves the score.
    # To be more efficient, we can look at the derivative or gradient.
    # But for binary coefficients, we can just try flipping each one.
    # 24 coefficients, 2048 points. Flipping one coefficient changes the value at point j by:
    # delta_j = -2 * c_k * z_j^k
    # New |P(z_j)| = |P(z_j) - 2*c_k*z_j^k|
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Evaluate current polynomial
        vals = powers @ current_coeffs
        abs_vals = np.abs(vals)
        
        # We want to minimize the max. 
        # A greedy approach: find the current max peak.
        # Try to flip a coefficient to reduce this peak.
        # However, reducing one peak might increase another.
        # So we minimize the sum of squared magnitudes of the top M peaks?
        # Or just the global max?
        
        # Let's try to minimize the global max by flipping the coefficient 
        # that best reduces the value at the current maximum peak.
        
        max_idx = np.argmax(abs_vals)
        max_val_local = abs_vals[max_idx]
        
        # Try flipping each coefficient
        best_flip_idx = -1
        best_new_max = max_val_local
        
        # Pre-compute z^k for the max point
        z_max = zs[max_idx]
        z_powers_max = np.ones(24, dtype=complex)
        for k in range(1, 24):
            z_powers_max[k] = z_powers_max[k-1] * z_max
            
        for k in range(24):
            # If we flip c_k, the change at z_max is -2 * c_k * z_max^k
            # New value at z_max: V_new = V_old - 2 * c_k * z_max^k
            # We want to minimize |V_new|
            # But we also care about other points.
            
            # Quick check: what is the new value at the max peak?
            v_old = vals[max_idx]
            v_new = v_old - 2 * current_coeffs[k] * z_powers_max[k]
            
            # If this doesn't reduce the peak significantly, skip?
            # Actually, we need to check the whole polynomial to ensure we don't create a larger peak elsewhere.
            # This is expensive (24 * 2048 operations per iteration).
            # 24 * 2048 = ~50k ops. Very fast.
            
            new_coeffs = current_coeffs.copy()
            new_coeffs[k] = -new_coeffs[k]
            new_vals = powers @ new_coeffs
            new_abs_vals = np.abs(new_vals)
            new_max = np.max(new_abs_vals)
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_flip_idx = k
                
        if best_flip_idx != -1:
            current_coeffs[best_flip_idx] = -current_coeffs[best_flip_idx]
            current_val = best_new_max / math.sqrt(24)
            if current_val < best_val:
                best_val = current_val
                best_coeffs = current_coeffs.copy()
            improved = True
        else:
            # If no single flip reduces the max, try a random flip to escape local minima?
            # Or try flipping pairs?
            # For now, let's try a random restart with a slightly perturbed initial state
            # or just stop if we can't improve greedily.
            
            # Let's try a "kick": flip a random coefficient even if it increases the score,
            # then try to recover.
            if time.time() < deadline - 10:
                idx = np.random.randint(0, 24)
                current_coeffs[idx] = -current_coeffs[idx]
                current_val = fast_score(current_coeffs) / math.sqrt(24)
                improved = True # Allow more iterations to recover
            else:
                break

    # Final verification with high-accuracy score
    # If time permits, run full_score on the best candidate
    if time.time() < deadline - 5:
        final_score = full_score(best_coeffs)
        # If the high-accuracy score is worse than expected, it means our grid was misleading.
        # But we have no alternative. We return the best_coeffs found by the fast heuristic.
        
    return best_coeffs.astype(int).tolist()
# EVOLVE-BLOCK-END
