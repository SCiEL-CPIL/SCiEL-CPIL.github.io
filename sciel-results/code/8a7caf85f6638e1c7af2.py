import numpy as np
import time
import itertools

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Symmetry: D4 group about center
    # Independent region: x >= 0, y >= 0, y <= x in centered coords
    # Center is (size/2 - 0.5, size/2 - 0.5) = (15.5, 15.5) for size=32
    # x = i - 15.5, y = j - 15.5
    # x >= 0 => i >= 16
    # y >= 0 => j >= 16
    # y <= x => j <= i
    # So i in [16, 31], j in [16, i]
    
    # List of independent pixel indices (i, j)
    indep_indices = []
    for i in range(16, size):
        for j in range(16, i + 1):
            indep_indices.append((i, j))
            
    num_indep = len(indep_indices)
    
    # Initialize with a square hole in dielectric background (inverted rod)
    # Dielectric is 1, Air is 0.
    # Square hole: center is air.
    # Let's start with a diamond or square shape.
    # For i,j in indep, if (i-15.5)^2 + (j-15.5)^2 < r^2 then Air (0) else Dielectric (1)
    # Let's try a square hole of size 8x8 (radius 4)
    # Or a cross.
    # Start with a simple circle of radius 4.
    r_init = 4.0
    initial_state = []
    for i, j in indep_indices:
        x = i - 15.5
        y = j - 15.5
        if x*x + y*y < r_init*r_init:
            initial_state.append(0) # Air
        else:
            initial_state.append(1) # Dielectric
            
    best_state = initial_state[:]
    best_score = -np.inf
    
    # Fast PWE Solver Parameters
    # We need to compute the gap between band 4 and 5.
    # Use 25 plane waves for speed.
    # Plane waves G satisfying |G| < G_max
    # For 25 modes, we can take G vectors with |G| <= 2 * (2*pi/size)?
    # Let's generate G vectors.
    # Lattice constant a = 1.
    # G = (2*pi/a) * (n, m)
    # We want about 25 vectors.
    # |G| = 2*pi * sqrt(n^2 + m^2)
    # Let's take sqrt(n^2+m^2) <= 2.
    # (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1) -> 9 vectors
    # sqrt(5) ~ 2.23. (2,1), (1,2)...
    # Let's just take all (n,m) with n^2+m^2 <= 4 (radius 2).
    # n,m in {-2, -1, 0, 1, 2}
    # Count:
    # 0: 1
    # 1: 4 (1,0,0,1,-1,0,0,-1) -> 4
    # 2: 4 (2,0,0,2,-2,0,0,-2) -> 4
    # 5: 8 (1,2,2,1,-1,2,-2,1,1,-2,2,-1,-1,-2,-2,-1) -> 8
    # 8: 4 (2,2,-2,2,-2,-2,2,-2) -> 4
    # Total: 1+4+4+8+4 = 21 vectors. Close enough to 25.
    
    G_vectors = []
    for n in range(-2, 3):
        for m in range(-2, 3):
            if n*n + m*m <= 4:
                G_vectors.append((2*np.pi*n, 2*np.pi*m))
                
    num_G = len(G_vectors)
    
    # K points: M (pi, 0) and X (pi, pi) are critical for square lattice gaps.
    # Also Gamma (0,0).
    k_points = [
        (0, 0),
        (np.pi, 0), # M
        (np.pi, np.pi), # X
        (np.pi/2, np.pi/2) # Intermediate
    ]
    
    def get_full_cell(state):
        cell = np.zeros((size, size), dtype=int)
        for idx, (i, j) in enumerate(indep_indices):
            val = state[idx]
            # Apply symmetries
            # (i, j)
            cell[i, j] = val
            # (j, i)
            cell[j, i] = val
            # (size-1-i, j) -> x neg
            cell[size-1-i, j] = val
            # (i, size-1-j) -> y neg
            cell[i, size-1-j] = val
            # (size-1-i, size-1-j) -> both neg
            cell[size-1-i, size-1-j] = val
            # (j, size-1-i) -> swap and neg x?
            # Let's verify D4 symmetries.
            # D4 includes:
            # 1. Identity
            # 2. 90 deg rot: (i, j) -> (size-1-j, i)
            # 3. 180 deg rot: (i, j) -> (size-1-i, size-1-j)
            # 4. 270 deg rot: (i, j) -> (j, size-1-i)
            # 5. Refl x: (i, j) -> (size-1-i, j)
            # 6. Refl y: (i, j) -> (i, size-1-j)
            # 7. Refl x=y: (i, j) -> (j, i)
            # 8. Refl x=-y: (i, j) -> (size-1-j, size-1-i)
            
            # We defined indep as x>=0, y>=0, y<=x.
            # i >= 16, j >= 16, j <= i.
            
            # Generate all 8 symmetric positions
            pos1 = (i, j)
            pos2 = (j, i)
            pos3 = (size-1-i, j)
            pos4 = (i, size-1-j)
            pos5 = (size-1-i, size-1-j)
            pos6 = (size-1-j, i)
            pos7 = (j, size-1-i)
            pos8 = (size-1-j, size-1-i)
            
            for p in [pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]:
                cell[p[0], p[1]] = val
                
        return cell

    def compute_fourier_coeffs(cell):
        # cell is 0/1. eps = 1 if 0, 12 if 1.
        # eps_inv = 1/eps
        eps = np.where(cell == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        
        # FFT of eps_inv
        # Need to shift to center?
        # The pixel (0,0) is at (-15.5, -15.5).
        # Standard FFT assumes (0,0) is at origin.
        # We need to shift the array so that the center is at (0,0) in the FFT grid?
        # Or just use the raw FFT and account for the phase?
        # The Fourier coefficient c_G = (1/N^2) sum_{x,y} eps_inv(x,y) exp(-i G . r)
        # Using numpy.fft.fft2, the output is ordered [0, 1, ..., N-1, -N/2, ...]
        # The input array is indexed 0..N-1.
        # We need to map pixel (i,j) to position (i-15.5, j-15.5).
        # This is a shift of the array.
        # np.fft.fft2 expects the array to be aligned with the grid.
        # If we just use fft2(eps_inv), it computes sum eps_inv(i,j) exp(-i 2pi (n i + m j)/N).
        # This corresponds to G = (2pi n/N, 2pi m/N).
        # But our G vectors are (2pi n, 2pi m) with lattice constant 1.
        # The pixel size is 1/32.
        # So the discrete sum approximates the integral.
        # c_G = (1/32^2) sum_{i,j} eps_inv(i,j) exp(-i (2pi n (i-15.5)/32 + 2pi m (j-15.5)/32))
        # = exp(i 2pi (n+m) 15.5/32) * (1/32^2) sum eps_inv(i,j) exp(-i 2pi (n i + m j)/32)
        # The term in the sum is exactly the FFT coefficient.
        # So we need to multiply the FFT result by a phase factor.
        
        fft_eps_inv = np.fft.fft2(eps_inv)
        
        # We need to extract coefficients for our G_vectors.
        # G = (2pi n, 2pi m).
        # In the FFT grid, the frequency index k corresponds to 2pi k / N.
        # So n_fft = n * N / (2pi) * (2pi / 1)? No.
        # G_x = 2pi n.
        # FFT freq = 2pi k / N.
        # So 2pi n = 2pi k / N => k = n * N.
        # Wait, the lattice constant is 1. The pixel size is 1/N.
        # The reciprocal lattice vector is 2pi / 1 = 2pi.
        # The FFT frequencies are 2pi k / N.
        # So to get G = 2pi n, we need k = n * N.
        # But n is an integer. k must be an integer in 0..N-1.
        # If n=1, k=32. This is out of bounds for FFT (0..31).
        # This means the FFT resolution is not fine enough to directly index n=1?
        # No, the FFT computes coefficients for G = 2pi k / N.
        # We need G = 2pi n.
        # So we need the coefficient at index k = n * N?
        # This implies we need to interpolate or use a larger grid?
        # No, the standard PWE uses the Fourier series of the periodic function.
        # The coefficients are c_n = (1/A) integral eps(r) exp(-i 2pi n r / a) dA.
        # With pixels, c_n = (1/N^2) sum_{i,j} eps(i,j) exp(-i 2pi n (i-0.5)/N)?
        # If we use fft2, we get coefficients for frequencies 2pi k / N.
        # If we want the coefficient for G = 2pi n, we need the frequency 2pi n.
        # This is the k-th harmonic where k = n * N?
        # No, the fundamental frequency of the FFT grid is 2pi / N.
        # The G vector 2pi n is the n-th harmonic of the lattice.
        # In the FFT of a sampled function, the coefficient at index k corresponds to frequency 2pi k / N.
        # So to get the coefficient for G = 2pi n, we look at index k = n * N?
        # This is only possible if n*N < N, i.e., n=0.
        # This suggests that the FFT of the pixel grid directly gives the coefficients for the discrete Fourier series on the pixel grid, not the continuous one.
        # However, for PWE, we need the continuous Fourier coefficients.
        # We can approximate them by:
        # c_G = (1/N^2) sum_{i,j} eps_inv(i,j) exp(-i G . (r_i - r_0))
        # where r_i is the pixel center.
        # We can compute this directly for each G in G_vectors.
        
        coeffs = np.zeros(num_G, dtype=complex)
        # Precompute pixel centers
        x_coords = np.arange(size) - (size/2 - 0.5)
        y_coords = np.arange(size) - (size/2 - 0.5)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        for idx, G in enumerate(G_vectors):
            Gx, Gy = G
            phase = -1j * (Gx * X + Gy * Y)
            coeffs[idx] = np.sum(eps_inv * np.exp(phase)) / (size * size)
            
        return coeffs

    def solve_pwe(kx, ky, coeffs):
        # Build matrix M
        # M_ij = delta_ij |k+G_i|^2 + sum_k c_k (k+G_i) . (k+G_j - k) ?
        # Correct TM PWE matrix:
        # M_ij = (1/c^2) [ delta_ij |k+G_i|^2 + sum_k c_k (k+G_i) . (k+G_j - k) ]
        # Wait, the term is (k+G_i) . (k+G_j - k)?
        # Let's use the form:
        # M_ij = delta_ij |k+G_i|^2 + sum_k c_k (k+G_i) . (k+G_j - k)
        # Actually, the standard form is:
        # M_ij = delta_ij |k+G_i|^2 + sum_k c_k (k+G_i) . (k+G_j - k)
        # Let's check the dimension.
        # The coupling term is sum_k c_k (k+G_i) . (k+G_j - k).
        # Note that (k+G_j - k) = G_j.
        # So the term is sum_k c_k (k+G_i) . G_j.
        # This seems asymmetric.
        
        # Correct TM PWE:
        # H_ij = delta_ij |k+G_i|^2 + sum_k c_k (k+G_i) . (k+G_j - k)
        # Actually, the vector is (k+G_i) and (k+G_j).
        # The term is sum_k c_k (k+G_i) . (k+G_j - k)?
        # Let's use the known result:
        # H_ij = delta_ij |k+G_i|^2 + sum_k c_k (k+G_i) . (k+G_j - k)
        # I will use:
        # H_ij = delta_ij |k+G_i|^2 + sum_k c_k (k+G_i) . (k+G_j - k)
        
        M = np.zeros((num_G, num_G), dtype=complex)
        k_vec = np.array([kx, ky])
        
        for i in range(num_G):
            kGi = k_vec + np.array(G_vectors[i])
            for j in range(num_G):
                kGj = k_vec + np.array(G_vectors[j])
                term = 0
                for l in range(num_G):
                    Gl = np.array(G_vectors[l])
                    # Term: c_l * (k+G_i) . (k+G_j - G_l) ?
                    # No, the sum is over G_l.
                    # The term is c_l * (k+G_i) . (k+G_j - G_l)
                    # Wait, the standard formula is:
                    # H_ij = delta_ij |k+G_i|^2 + sum_l c_l (k+G_i) . (k+G_j - G_l)
                    # Let's verify.
                    # If l=0, G_l=0. Term is c_0 (k+G_i) . (k+G_j).
                    # This looks plausible.
                    
                    term += coeffs[l] * np.dot(kGi, kGj - Gl)
                    
                if i == j:
                    M[i, j] = np.dot(kGi, kGi) + term
                else:
                    M[i, j] = term
                    
        # Solve eigenvalue problem
        # M is Hermitian?
        # Check if M is Hermitian.
        # M_ij = delta_ij |kGi|^2 + sum_l c_l (kGi) . (kGj - Gl)
        # M_ji = delta_ji |kGj|^2 + sum_l c_l (kGj) . (kGi - Gl)
        # If c_l is real (which it is for symmetric eps), and the dot product is symmetric?
        # (kGi) . (kGj - Gl) vs (kGj) . (kGi - Gl)
        # These are not necessarily equal.
        # However, the matrix should be Hermitian.
        # Let's symmetrize: M = (M + M.conj().T) / 2
        
        M = (M + M.conj().T) / 2
        
        # Get eigenvalues
        eigvals = np.linalg.eigvalsh(M)
        return eigvals

    def fast_score(state):
        cell = get_full_cell(state)
        coeffs = compute_fourier_coeffs(cell)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            eigvals = solve_pwe(kx, ky, coeffs)
            # eigvals are omega^2
            omegas = np.sqrt(eigvals)
            # Sort
            omegas = np.sort(omegas)
            
            # We need bands 4 and 5 (indices 3 and 4)
            if len(omegas) >= 5:
                omega_4 = omegas[3]
                omega_5 = omegas[4]
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
                
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -np.inf
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2
        if midgap == 0:
            return 0
        return gap / midgap

    # Optimization Loop
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    current_state = best_state[:]
    current_score = fast_score(current_state)
    if current_score > best_score:
        best_score = current_score
        best_state = current_state[:]
        
    while time.time() - start_time < time_limit:
        # Hill climbing
        improved = False
        for idx in range(num_indep):
            if time.time() - start_time > time_limit:
                break
                
            # Flip pixel
            new_state = current_state[:]
            new_state[idx] = 1 - new_state[idx]
            
            score = fast_score(new_state)
            
            if score > current_score:
                current_state = new_state
                current_score = score
                improved = True
                
                if score > best_score:
                    best_score = score
                    best_state = current_state[:]
                    
        if not improved:
            # Random restart
            # Perturb current state
            for _ in range(10):
                idx = np.random.randint(num_indep)
                current_state[idx] = 1 - current_state[idx]
                
            current_score = fast_score(current_state)
            if current_score > best_score:
                best_score = current_score
                best_state = current_state[:]

    # Convert best_state to nested list
    best_cell = get_full_cell(best_state)
    return best_cell.tolist()

