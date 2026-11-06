import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0  # Keep margin from 120s
    
    # Independent region: top-left 16x16
    half_size = size // 2
    
    def symmetrize(quadrant):
        """Expand 16x16 quadrant to 32x32 symmetric cell."""
        cell = np.zeros((size, size), dtype=np.int8)
        for i in range(half_size):
            for j in range(half_size):
                val = quadrant[i, j]
                cell[i, j] = val
                cell[i, size - 1 - j] = val
                cell[size - 1 - i, j] = val
                cell[size - 1 - i, size - 1 - j] = val
        return cell

    def compute_fourier_coeffs(cell):
        """Compute epsilon^{-1}(G) for G in [-16, 16]."""
        # cell is 32x32. 1 is dielectric (eps=12), 0 is air (eps=1).
        # We need eps_inv.
        eps_inv = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        
        # G vectors range
        G_range = np.arange(-16, 17)
        # We need coeffs for Gx, Gy in this range
        coeffs = np.zeros((len(G_range), len(G_range)), dtype=np.complex128)
        
        # Use FFT for speed? 
        # eps_inv(G) = (1/N^2) sum eps_inv(i,j) exp(-i 2pi (n i / N + m j / N))
        # This is just the DFT of eps_inv.
        # We need the central part of the DFT.
        # N=32.
        # DFT gives frequencies 0..31. We need -16..16.
        # np.fft.fft2 gives 0..31.
        # freqs = [0, 1, ..., 15, -16, -15, ..., -1]
        # We want -16..16.
        # Index mapping:
        # k in -16..16 maps to fft index:
        # if k >= 0: idx = k
        # if k < 0: idx = k + 32
        
        fft_eps_inv = np.fft.fft2(eps_inv) / (size * size)
        
        for i, gx in enumerate(G_range):
            for j, gy in enumerate(G_range):
                idx_x = gx if gx >= 0 else gx + size
                idx_y = gy if gy >= 0 else gy + size
                coeffs[i, j] = fft_eps_inv[idx_x, idx_y]
                
        return coeffs, G_range

    def evaluate_cell(cell):
        """Compute the gap-midgap ratio for the given cell."""
        # Check time
        if time.time() - start_time > time_limit - 10:
            return -1.0  # Timeout indicator
            
        coeffs, G_range = compute_fourier_coeffs(cell)
        
        # Plane waves: 17x17 grid centered at 0
        # Indices -8 to 8
        pw_indices = np.arange(-8, 9)
        G_vectors = []
        for gx in pw_indices:
            for gy in pw_indices:
                G_vectors.append((gx, gy))
                
        N_pw = len(G_vectors)
        
        # Precompute G^2 for each PW
        G_sq = np.array([(gx**2 + gy**2) for gx, gy in G_vectors], dtype=np.float64)
        
        # We need to evaluate at several k-points.
        # The prompt mentions 30 wave vectors.
        # Irreducible BZ of square lattice: 0 <= ky <= kx, kx, ky in [0, pi/a]
        # Let's sample a few points: Gamma (0,0), M (pi,0), X (pi/2, pi/2) ? 
        # Square lattice BZ: square from -pi to pi.
        # Irreducible: 1/8 wedge.
        # Points: (0,0), (pi,0), (pi/2, pi/2).
        # Let's use a dense set of points to approximate min/max.
        
        # k-points in units of pi/size? No, k is in 1/length.
        # G = 2pi n / size.
        # k should be in same units.
        # Let's define k in units of 2pi/size.
        # Gamma: (0,0)
        # M: (1,0) -> k = 2pi/32 * 1? No.
        # BZ boundary is at k = pi/size? No, lattice constant a=size.
        # BZ is -pi/a to pi/a.
        # In units of 2pi/a (which is G unit), BZ is -0.5 to 0.5.
        # Let's use k in units of G (2pi/a).
        # Gamma: (0,0)
        # M: (0.5, 0)
        # X: (0.5, 0.5)
        
        k_points = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.25, 0.0),
            (0.25, 0.25),
            (0.5, 0.25),
            (0.1, 0.0),
            (0.3, 0.1),
            (0.4, 0.2)
        ]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build matrix
            # M[k,l] = (k+G_k)^2 * eps_inv(G_k - G_l)
            # Note: k is in units of G. So k+G_k is just vector addition of indices?
            # G_k is (gx, gy). k is (kx, ky).
            # (k+G_k)^2 = (kx+gx)^2 + (ky+gy)^2
            
            M = np.zeros((N_pw, N_pw), dtype=np.complex128)
            
            for i, (gx_i, gy_i) in enumerate(G_vectors):
                kx_total = kx + gx_i
                ky_total = ky + gy_i
                k_sq = kx_total**2 + ky_total**2
                
                for j, (gx_j, gy_j) in enumerate(G_vectors):
                    # G_diff = G_i - G_j
                    dx = gx_i - gx_j
                    dy = gy_i - gy_j
                    
                    # Get coeff
                    # coeffs is indexed by G_range [-16, 16]
                    # dx, dy are in [-16, 16]
                    idx_x = dx + 16
                    idx_y = dy + 16
                    eps_inv_g = coeffs[idx_x, idx_y]
                    
                    M[i, j] = k_sq * eps_inv_g
            
            # Solve eigenvalues
            # M is Hermitian?
            # eps_inv(G) is real? 
            # eps_inv(r) is real and symmetric (cell is symmetric).
            # So eps_inv(G) is real.
            # M[i,j] = (k+G_i)^2 eps_inv(G_i-G_j).
            # Is M Hermitian?
            # M[j,i] = (k+G_j)^2 eps_inv(G_j-G_i).
            # (k+G_i)^2 != (k+G_j)^2 generally.
            # So M is NOT Hermitian.
            # We must use general eigenvalue solver.
            
            try:
                eigvals = np.linalg.eigvals(M)
            except Exception:
                return -1.0
                
            # Filter real, positive eigenvalues
            eigvals_real = [v.real for v in eigvals if abs(v.imag) < 1e-6 and v.real > 0]
            if len(eigvals_real) < 5:
                continue
                
            eigvals_real.sort()
            # 4th and 5th bands (1-based) -> indices 3 and 4
            w4 = eigvals_real[3]
            w5 = eigvals_real[4]
            
            # w is omega^2. We need omega.
            w4_omega = np.sqrt(w4)
            w5_omega = np.sqrt(w5)
            
            max_omega_4 = max(max_omega_4, w4_omega)
            min_omega_5 = min(min_omega_5, w5_omega)
            
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        ratio = 2.0 * gap / midgap
        return ratio

    # Initial Candidates
    candidates = []
    
    # 1. Seed: Rod
    q1 = np.zeros((16, 16), dtype=np.int8)
    for i in range(16):
        for j in range(16):
            # Center of 32x32 is (15.5, 15.5)
            # In 16x16 quadrant, center is (7.5, 7.5)?
            # i=0..15 maps to 0..15 and 31..16.
            # Pixel (i,j) in quadrant corresponds to (i,j), (i, 31-j), (31-i, j), (31-i, 31-j).
            # Distance from center (15.5, 15.5) for pixel (i,j) is sqrt((i-15.5)^2 + (j-15.5)^2)
            # Wait, pixel (0,0) is corner. Pixel (15,15) is near center.
            # Center of cell is between (15,15), (15,16), (16,15), (16,16).
            # Let's use (15.5, 15.5).
            dist_sq = (i - 15.5)**2 + (j - 15.5)**2
            if dist_sq < 0.2**2 * (32**2): # Radius 0.2 * 32 pixels?
                # Seed used radius 0.2 lattice constants. Lattice constant = 32 pixels.
                # So radius = 0.2 * 32 = 6.4 pixels.
                if dist_sq < 6.4**2:
                    q1[i, j] = 1
    candidates.append(q1)
    
    # 2. Inverted Rod (Dielectric background, Air hole)
    q2 = 1 - q1
    candidates.append(q2)
    
    # 3. Cross
    q3 = np.zeros((16, 16), dtype=np.int8)
    for i in range(16):
        for j in range(16):
            # Vertical bar: j=7,8 (near center)
            # Horizontal bar: i=7,8
            # Center is 15.5.
            # j=7 -> dist 8.5. j=8 -> dist 7.5.
            # Let's make a thick cross.
            if abs(j - 7.5) < 2.0 or abs(i - 7.5) < 2.0:
                q3[i, j] = 1
    candidates.append(q3)
    
    # 4. Random
    q4 = (np.random.rand(16, 16) > 0.5).astype(np.int8)
    candidates.append(q4)
    
    best_score = -1.0
    best_q = q1.copy()
    
    # Evaluate initial candidates
    for q in candidates:
        cell = symmetrize(q)
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_q = q.copy()
            
    # Hill Climbing
    max_iterations = 2000
    no_improve_count = 0
    patience = 50
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Flip a random pixel
        i = np.random.randint(0, 16)
        j = np.random.randint(0, 16)
        
        q_new = best_q.copy()
        q_new[i, j] = 1 - q_new[i, j]
        
        cell = symmetrize(q_new)
        score = evaluate_cell(cell)
        
        if score > best_score:
            best_score = score
            best_q = q_new
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > patience:
                # Try a bigger jump: flip 2 pixels
                i1 = np.random.randint(0, 16)
                j1 = np.random.randint(0, 16)
                i2 = np.random.randint(0, 16)
                j2 = np.random.randint(0, 16)
                
                q_jump = best_q.copy()
                q_jump[i1, j1] = 1 - q_jump[i1, j1]
                q_jump[i2, j2] = 1 - q_jump[i2, j2]
                
                cell = symmetrize(q_jump)
                score = evaluate_cell(cell)
                
                if score > best_score:
                    best_score = score
                    best_q = q_jump
                    no_improve_count = 0
                else:
                    no_improve_count = 0 # Reset patience after jump attempt

    # Final check
    if time.time() - start_time > time_limit - 1:
         # Return best known
         pass
         
    final_cell = symmetrize(best_q)
    
    # Convert to list of lists
    return final_cell.tolist()
