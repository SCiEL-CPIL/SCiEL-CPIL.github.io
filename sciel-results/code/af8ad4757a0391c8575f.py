import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s buffer before 120s kill
    
    N = 24
    
    # The known optimal polynomial for 24 coefficients has sup-norm/sqrt(24) ≈ 1.2583758
    # This corresponds to a specific sequence. We start with the GRS sequence and 
    # also include the known optimal structure if we can identify it.
    # The optimal polynomials for length 24 are listed in Odlyzko's tables.
    # One known optimal sequence (self-reciprocal) for n=24 is:
    # 1 1 1 -1 -1 -1 1 1 1 -1 -1 1 1 -1 -1 1 1 -1 1 -1 1 1 -1 1
    # Let's verify this structure or just use a strong initial guess.
    # Actually, the GRS sequence is: 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, ...
    # GRS for n=24:
    # p0 = 1
    # p1 = 1
    # p2 = 1
    # p3 = -1
    # p4 = -1
    # p5 = -1
    # p6 = 1
    # p7 = 1
    # p8 = 1
    # p9 = -1
    # p10 = -1
    # p11 = 1
    # p12 = 1
    # p13 = -1
    # p14 = -1
    # p15 = 1
    # p16 = 1
    # p17 = -1
    # p18 = 1
    # p19 = -1
    # p20 = 1
    # p21 = 1
    # p22 = -1
    # p23 = 1
    grs = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1])
    
    # Let's define a fast evaluator.
    # For speed in local search, we use a dense FFT grid.
    # M = 2^15 = 32768 is likely sufficient for finding peaks, then refine.
    # But for the final score, we need high precision.
    # The task says "dense sampling and golden-section refinement ... accuracy about 1e-9".
    # We should mimic this evaluation to ensure our score matches the evaluator's.
    
    M_grid = 65536  # 2^16
    
    def evaluate_sup_norm(c):
        """
        Compute max |P(z)| for |z|=1.
        c: array of length N with +/-1
        """
        # Pad to M_grid
        c_pad = np.zeros(M_grid, dtype=complex)
        c_pad[:N] = c
        # P(e^{i 2pi j/M}) = M * ifft(c_pad)_j
        vals = M_grid * np.fft.ifft(c_pad)
        abs_vals = np.abs(vals)
        
        # Find top peaks
        # The max is likely at one of the top few grid points.
        # We take top 3 to be safe.
        top_indices = np.argsort(abs_vals)[-3:]
        
        max_val = 0.0
        grid_spacing = 2 * np.pi / M_grid
        
        for idx in top_indices:
            theta_center = 2 * np.pi * idx / M_grid
            # Search in a small window around the peak
            # 2*grid_spacing is ~ 2 * 1e-4 radians
            a = theta_center - 2 * grid_spacing
            b = theta_center + 2 * grid_spacing
            
            def f(theta):
                # Direct evaluation for precision
                # P(z) = sum c_k z^k
                # z = exp(i theta)
                # Use np.dot for speed
                # Precompute powers? No, theta changes.
                # N=24 is small.
                angles = np.arange(N) * theta
                z_powers = np.exp(1j * angles)
                P_val = np.dot(c, z_powers)
                return np.abs(P_val)
                
            # Golden section search
            gr = (np.sqrt(5) - 1) / 2
            c_pt = a + (1 - gr) * (b - a)
            d_pt = a + gr * (b - a)
            
            fc = f(c_pt)
            fd = f(d_pt)
            
            # 20 iterations is enough for 1e-9 accuracy on a window of ~1e-3
            for _ in range(20):
                if fc < fd:
                    a = c_pt
                    c_pt = d_pt
                    fc = fd
                    d_pt = a + gr * (b - a)
                    fd = f(d_pt)
                else:
                    b = d_pt
                    d_pt = c_pt
                    fd = fc
                    c_pt = a + (1 - gr) * (b - a)
                    fc = f(c_pt)
                    
                if b - a < 1e-10:
                    break
                    
            mid = (a + b) / 2
            fm = f(mid)
            local_max = max(fc, fd, fm)
            if local_max > max_val:
                max_val = local_max
                
        return max_val

    best_coeffs = grs.copy()
    best_score = evaluate_sup_norm(best_coeffs)
    
    # Local Search
    # To speed up, we can use a faster surrogate for the local search steps,
    # but the final evaluation must be accurate.
    # Let's use the same evaluate_sup_norm but maybe reduce M_grid for search?
    # No, let's stick to one evaluator to avoid bias.
    # Optimization: Precompute ifft of basis vectors?
    # ifft of c_pad is linear.
    # ifft(c_pad) = sum c_k * ifft(e_k_pad)
    # Let B be a matrix where column k is ifft(e_k_pad).
    # Then vals = M * B @ c.
    # This is a matrix-vector product. B is M x N.
    # M=65536, N=24. B is 65536x24 complex.
    # Memory: 65536 * 24 * 16 bytes = ~25 MB. Acceptable.
    # Time: Matrix-vector product is fast.
    
    # Precompute B
    B = np.zeros((M_grid, N), dtype=complex)
    for k in range(N):
        e_k = np.zeros(M_grid, dtype=complex)
        e_k[k] = 1
        B[:, k] = np.fft.ifft(e_k)
        
    # Actually, ifft of a unit vector at k is just 1/M * exp(-2pi i j k / M).
    # So B[j, k] = 1/M * exp(-2pi i j k / M).
    # We can compute this directly without calling ifft N times.
    j = np.arange(M_grid)
    k_idx = np.arange(N)
    # B[j, k] = 1/M * exp(-2pi i j k / M)
    # This is a Vandermonde-like matrix.
    # Let's just use the direct formula for speed in precomputation.
    # exp(-2pi i j k / M)
    # j is (M,), k is (N,)
    # outer product
    # B = (1/M) * exp(-2pi i / M * np.outer(j, k))
    # This creates a (M, N) array.
    # M=65536, N=24.
    # This is fast.
    B = (1.0 / M_grid) * np.exp(-2j * np.pi / M_grid * np.outer(np.arange(M_grid), np.arange(N)))
    
    def evaluate_sup_norm_fast(c):
        # c is real array of +/-1
        # vals = M * B @ c
        # Note: B was defined as ifft(e_k), so ifft(c_pad) = B @ c.
        # So vals = M * (B @ c)
        vals = M_grid * (B @ c)
        abs_vals = np.abs(vals)
        top_indices = np.argsort(abs_vals)[-3:]
        
        max_val = 0.0
        grid_spacing = 2 * np.pi / M_grid
        
        for idx in top_indices:
            theta_center = 2 * np.pi * idx / M_grid
            a = theta_center - 2 * grid_spacing
            b = theta_center + 2 * grid_spacing
            
            def f(theta):
                angles = np.arange(N) * theta
                z_powers = np.exp(1j * angles)
                P_val = np.dot(c, z_powers)
                return np.abs(P_val)
                
            gr = (np.sqrt(5) - 1) / 2
            c_pt = a + (1 - gr) * (b - a)
            d_pt = a + gr * (b - a)
            
            fc = f(c_pt)
            fd = f(d_pt)
            
            for _ in range(20):
                if fc < fd:
                    a = c_pt
                    c_pt = d_pt
                    fc = fd
                    d_pt = a + gr * (b - a)
                    fd = f(d_pt)
                else:
                    b = d_pt
                    d_pt = c_pt
                    fd = fc
                    c_pt = a + (1 - gr) * (b - a)
                    fc = f(c_pt)
                    
                if b - a < 1e-10:
                    break
                    
            mid = (a + b) / 2
            fm = f(mid)
            local_max = max(fc, fd, fm)
            if local_max > max_val:
                max_val = local_max
                
        return max_val

    # Re-evaluate best with fast method to ensure consistency
    best_score = evaluate_sup_norm_fast(best_coeffs)
    
    def local_search(coeffs):
        current = coeffs.copy()
        current_score = evaluate_sup_norm_fast(current)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try 1-flips
            best_single_flip = None
            best_single_score = current_score
            for k in range(N):
                new_c = current.copy()
                new_c[k] *= -1
                score = evaluate_sup_norm_fast(new_c)
                if score < best_single_score:
                    best_single_score = score
                    best_single_flip = new_c
            
            if best_single_flip is not None:
                current = best_single_flip
                current_score = best_single_score
                improved = True
                continue
                
            # Try 2-flips
            best_pair_flip = None
            best_pair_score = current_score
            for i in range(N):
                for j in range(i+1, N):
                    new_c = current.copy()
                    new_c[i] *= -1
                    new_c[j] *= -1
                    score = evaluate_sup_norm_fast(new_c)
                    if score < best_pair_score:
                        best_pair_score = score
                        best_pair_flip = new_c
                        
            if best_pair_flip is not None:
                current = best_pair_flip
                current_score = best_pair_score
                improved = True
                continue
                
        return current, current_score

    # Iterated Local Search
    # Start with GRS
    # Also try a few random starts
    starts = [best_coeffs]
    for _ in range(5):
        random_coeffs = np.random.choice([-1, 1], N)
        starts.append(random_coeffs)
        
    for start_c in starts:
        if time.time() > deadline - 10:
            break
        local_best, local_score = local_search(start_c)
        if local_score < best_score:
            best_score = local_score
            best_coeffs = local_best
            
    # Continue with ILS from best
    while time.time() < deadline - 5:
        num_flips = 3 + np.random.randint(0, 3) # 3-5 flips
        indices = np.random.choice(N, num_flips, replace=False)
        perturbed = best_coeffs.copy()
        perturbed[indices] *= -1
        
        local_best, local_score = local_search(perturbed)
        
        if local_score < best_score:
            best_score = local_score
            best_coeffs = local_best
            
    return [int(x) for x in best_coeffs]
