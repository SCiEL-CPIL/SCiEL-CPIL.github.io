import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell maximizing the TM band gap 
    between the 4th and 5th bands using a 4-fold symmetric shape optimization.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    timeout = 220.0 # Safety margin before 230s kill
    
    # Pre-compute grid coordinates
    # Coordinates centered at 0.5, 0.5 for the 32x32 grid
    # x, y in [0, 1)
    x_coords = (np.arange(size) + 0.5) / size
    y_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Distance from center (0.5, 0.5)
    dx = X - 0.5
    dy = Y - 0.5
    R = np.sqrt(dx**2 + dy**2)
    Theta = np.arctan2(dy, dx)
    
    # Surrogate Solver Setup
    # We use a reduced plane wave expansion for speed.
    # The evaluator uses 289 PWs (approx 17x17). We use 81 (9x9) or 49 (7x7).
    # Let's try 9x9 = 81 PWs.
    N_pw = 9 
    # G vectors for square lattice
    G = []
    for i in range(-N_pw//2, N_pw//2 + 1):
        for j in range(-N_pw//2, N_pw//2 + 1):
            G.append((i, j))
    G = np.array(G)
    N_G = len(G)
    
    # Fourier coefficients of epsilon
    # epsilon(r) = eps_low + (eps_high - eps_low) * mask(r)
    # We need the Fourier series of the mask.
    # Instead of computing Fourier coeffs of the specific mask for each candidate (expensive),
    # we can pre-compute the basis functions e^{-i G . r} on the grid?
    # Actually, the matrix element is <G|1/eps|G'>.
    # For a binary material, 1/eps(r) is also binary.
    # We can approximate 1/eps(r) by its Fourier series.
    
    # To make it faster, we will calculate the Fourier coefficients of the mask 
    # and 1/eps mask directly from the pixel grid for each candidate.
    # This is O(N_px * N_G) which is 1024 * 81 ~ 80k ops per candidate. Very fast.
    
    # Key K-points in the irreducible BZ
    # Gamma (0,0), X (0.5, 0), M (0.5, 0.5)
    # And some interior points to catch the band edges which might not be at high symmetry
    # for 4th/5th bands.
    # Common band edges for TM4/TM5 in square lattice are often near X or M or interior.
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.1, 0.4), # Random interior
        (0.4, 0.1), # Random interior
    ]
    # Scale k-points by 2*pi? The plane wave expansion usually uses k in units of 2*pi/a.
    # The G vectors are in units of 2*pi/a.
    # So k_points are in units of 2*pi/a.
    
    def calculate_gap(mask):
        """
        Calculates the gap ratio using the surrogate solver.
        mask: 32x32 array of 0/1.
        Returns: gap_ratio
        """
        # Compute Fourier coefficients of 1/epsilon
        # 1/eps(r) = 1/eps_low * (1 - mask) + 1/eps_high * mask
        # Let's compute the FT of the binary mask and 1/eps mask.
        
        # Using DFT approximation on the 32x32 grid
        # Note: The grid is 32x32. The G vectors we chose are for a 9x9 expansion.
        # The DFT of the 32x32 grid gives coefficients for G = (m/32, n/32) in 2pi units?
        # No, standard DFT gives frequencies 0..31.
        # For a periodic function on [0,1), the Fourier series is sum c_n e^{2pi i n x}.
        # The G vectors in the PW expansion are integer multiples of the reciprocal lattice vector.
        # So we need the coefficients for integer n.
        # We can compute the DFT of the mask, and then pick the coefficients corresponding to 
        # the integer frequencies -4..4 (for 9x9).
        
        # Compute 2D FFT of mask
        fft_mask = np.fft.fft2(mask)
        fft_mask = np.fft.fftshift(fft_mask) # Center zero frequency
        
        # Compute 1/eps mask
        inv_eps_mask = np.zeros_like(mask, dtype=float)
        inv_eps_mask[mask == 1] = 1.0 / eps_high
        inv_eps_mask[mask == 0] = 1.0 / eps_low
        
        fft_inv_eps = np.fft.fft2(inv_eps_mask)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        
        # Normalize FFT to get Fourier coefficients
        # The relation between DFT and Fourier series coefficients:
        # c_n = (1/N^2) * FFT_n
        # But we need to map the 32x32 FFT indices to the integer G vectors.
        
        # Indices for G in 9x9 range: -4 to 4.
        # In the shifted FFT array (size 32x32), index 16 is 0.
        # Index 16 + i corresponds to frequency i.
        
        # Extract coefficients for G in our list
        # G list contains (i, j) where i, j in -4..4
        # We need to map (i, j) to FFT index (16+i, 16+j)
        
        # However, the DFT of a 32x32 grid gives coefficients for frequencies -16..15.
        # Our G vectors are -4..4, which are within this range.
        
        # Let's build the matrix H (Hamiltonian) for each k-point.
        # H_{G, G'} = (k + G) dot (k + G') * delta_{G, G'}  (for TM, the operator is (k+G)^2 / eps)
        # Wait, the TM eigenvalue problem is:
        # (k+G)^2 * a_{G'} / eps(G-G') ... no.
        # The standard PW expansion for TM:
        # (k+G)^2 * a_{G'} = omega^2 * sum_{G'} (1/eps(G-G')) a_{G'}
        # So the matrix is M_{G, G'} = (1/eps(G-G'))
        # And the eigenvalues are lambda = omega^2 / (k+G)^2 ? No.
        
        # Let's stick to the generalized eigenvalue problem:
        # A v = lambda B v
        # Where A is diagonal with entries (k+G)^2
        # And B is the matrix with entries 1/eps(G-G')
        
        # Actually, it's simpler:
        # sum_{G'} (1/eps(G-G')) a_{G'} = (omega/(2pi))^2 * (1/(k+G)^2) a_G ?
        # No, the equation is:
        # (k+G)^2 a_G = omega^2 sum_{G'} (1/eps(G-G')) a_{G'}
        # So:
        # sum_{G'} [ delta_{G,G'} (k+G)^2 - omega^2 (1/eps(G-G')) ] a_{G'} = 0
        
        # This is a generalized eigenvalue problem:
        # D a = omega^2 M a
        # Where D is diagonal with (k+G)^2
        # And M is the matrix of 1/eps Fourier coefficients.
        
        # Let's compute M first.
        # M_{G, G'} = 1/eps(G - G')
        # We have the FFT of 1/eps.
        # The coefficient for difference G - G' is c_{G-G'}.
        
        # Let's extract the needed coefficients from fft_inv_eps.
        # c_{n, m} = fft_inv_eps[16+n, 16+m] / (32*32)
        
        # We need M for all pairs in our G list.
        # G list is 9x9=81 vectors.
        # M will be 81x81.
        
        # Pre-compute indices for G in the FFT grid
        # G is (i, j). FFT index is (16+i, 16+j).
        
        # Build M
        M = np.zeros((N_G, N_G), dtype=complex)
        for idx1, G1 in enumerate(G):
            for idx2, G2 in enumerate(G):
                dG = G1 - G2
                # Get coefficient c_{dG}
                # dG is (dx, dy).
                # FFT index: 16 + dx, 16 + dy
                # Note: fftshift moves 0 to center.
                # Original FFT index for freq f is (f + 16) % 32.
                # With fftshift, index i corresponds to freq i-16.
                # So freq f is at index f+16.
                # Since dG is small (-4..4), index is 12..20.
                c = fft_inv_eps[16 + int(dG[0]), 16 + int(dG[1])] / (size * size)
                M[idx1, idx2] = c
        
        # Now solve for each k-point
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for k in k_points:
            kx, ky = k
            # Build D matrix (diagonal)
            D_vals = np.array([ (kx + gx)**2 + (ky + gy)**2 for gx, gy in G ])
            
            # Solve generalized eigenvalue problem:
            # D a = omega^2 M a
            # This is equivalent to standard eigenvalue problem if we invert M?
            # Or use scipy.linalg.eigh(M_inv, D)? No, eigh(A, B) solves A v = lambda B v.
            # Here A = D, B = M.
            # We want omega^2.
            # D a = omega^2 M a  =>  M^{-1} D a = omega^2 a
            # So we can compute M_inv = np.linalg.inv(M) and then eig(M_inv @ D).
            # M is Hermitian positive definite (since 1/eps > 0).
            
            try:
                M_inv = np.linalg.inv(M)
                # D is diagonal, so M_inv @ D is just M_inv scaled by columns?
                # No, (M_inv @ D)_{ij} = M_inv_{ij} * D_j
                A_eff = M_inv * D_vals[None, :] # Broadcasting: scale columns of M_inv
                # A_eff should be Hermitian?
                # M is Hermitian. D is real diagonal.
                # M_inv is Hermitian.
                # M_inv @ D is NOT Hermitian in general.
                # Wait. The problem is D a = lambda M a.
                # This is a symmetric generalized eigenvalue problem if D and M are symmetric.
                # We can use scipy.linalg.eigh(D, M).
                
                import scipy.linalg as sla
                eigvals, eigvecs = sla.eigh(D_vals, M)
                
                # eigvals are lambda = omega^2
                # Sort them
                eigvals = np.sort(eigvals)
                
                # We need the 4th and 5th bands.
                # Note: The bands are indexed 1-based in the problem description?
                # "fourth and fifth TM bands".
                # Usually, band 1 is the lowest.
                # So we take eigvals[3] and eigvals[4] (0-indexed).
                
                if len(eigvals) >= 5:
                    w4_sq = eigvals[3]
                    w5_sq = eigvals[4]
                    
                    # Ensure positive
                    if w4_sq > 0 and w5_sq > 0:
                        w4 = np.sqrt(w4_sq)
                        w5 = np.sqrt(w5_sq)
                        
                        max_omega_4 = max(max_omega_4, w4)
                        min_omega_5 = min(min_omega_5, w5)
            except Exception:
                pass
        
        if max_omega_4 == 0.0 or min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    best_score = -1.0
    best_mask = None
    
    # Grid search parameters
    # R0: base radius
    # eps: modulation depth
    # Shape: R(Theta) = R0 + eps * cos(4*Theta)
    
    R0_range = np.linspace(0.15, 0.35, 10)
    eps_range = np.linspace(-0.1, 0.1, 10)
    
    # Initial guess: The baseline circle
    # Let's start with a few points
    
    for R0 in R0_range:
        for eps in eps_range:
            if time.time() - start_time > timeout:
                break
                
            # Generate mask
            R_target = R0 + eps * np.cos(4 * Theta)
            mask = (R < R_target).astype(int)
            
            # Enforce symmetry? 
            # The formula R < R0 + eps*cos(4*Theta) is inherently 4-fold symmetric.
            # However, pixelization might break exact symmetry if not careful.
            # The evaluator requires: cell == cell[::-1] row-wise and column-wise and cell == transpose.
            # This means symmetry about the center (16,16) in 0-indexed, or (0.5,0.5) in continuous.
            # Our continuous shape is symmetric. The pixelization:
            # x = (i+0.5)/32 - 0.5.
            # The grid is symmetric about 0.5.
            # So the mask should be symmetric.
            # Just to be safe, we can enforce it.
            
            # Enforce 4-fold symmetry and reflections
            # 1. Reflection X: mask[i, j] == mask[i, size-1-j]
            # 2. Reflection Y: mask[i, j] == mask[size-1-i, j]
            # 3. Transpose: mask[i, j] == mask[j, i]
            
            # Since we generated from a radial function with 4-fold symmetry, 
            # it should satisfy these. But let's enforce to be safe.
            # We'll take the majority vote or just ensure consistency.
            # Actually, if the continuous function is symmetric, the pixel centers 
            # are symmetric, so the mask should be symmetric.
            
            score = calculate_gap(mask)
            
            if score > best_score:
                best_score = score
                best_mask = mask.copy()
                
            if time.time() - start_time > timeout:
                break

    # If no valid mask found, return baseline
    if best_mask is None:
        best_mask = (R < 0.2).astype(int)

    # Convert to list of lists
    return best_mask.tolist()
