"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fast_pwe_gap(cell, eps_low=1.0, eps_high=12.0, n_pw=17, n_kpoints=10):
    """
    Fast PWE surrogate using FFT to get Fourier coefficients.
    Returns the gap-midgap ratio for bands 4 and 5.
    """
    N = len(cell)
    eps = np.array(cell, dtype=float)
    eps = eps * (eps_high - eps_low) + eps_low
    
    # Compute Fourier coefficients of 1/eps for TM mode
    # We need the inverse dielectric function for TM
    inv_eps = 1.0 / eps
    
    # Get Fourier coefficients using FFT
    # Note: The PWE basis functions are exp(i (k+G) . r)
    # The matrix elements are <G' | 1/eps | G>
    # We can compute the convolution via FFT in real space or directly.
    # For a 32x32 grid, direct FFT is fast.
    # However, the PWE method sums over G vectors.
    # Let's use the standard PWE matrix construction.
    
    # Generate G vectors
    # Max G magnitude such that |k+G| < cutoff
    # For 32x32, a 17x17 grid of G vectors (from -8 to 8) is reasonable.
    # G = (2*pi/a) * (m, n)
    # Let a=1 for normalization.
    
    m_max = (n_pw - 1) // 2
    G_vectors = []
    for m in range(-m_max, m_max + 1):
        for n in range(-m_max, m_max + 1):
            G_vectors.append((m, n))
            
    G_vectors = np.array(G_vectors)
    n_g = len(G_vectors)
    
    # Fourier coefficients of inv_eps
    # f_G = (1/N^2) sum_{x,y} inv_eps(x,y) exp(-i G . (x,y))
    # This is the DFT.
    # We need f_G for G in G_vectors.
    # np.fft.fft2 computes F_{u,v} = sum_{x,y} f_{x,y} exp(-2pi i (u x / N + v y / N))
    # Our G is (2 pi m, 2 pi n).
    # exp(-i (2 pi m x + 2 pi n y)) matches fft2 with u=m, v=n if we map indices correctly.
    # fft2 output index [u, v] corresponds to frequency u, v.
    # If m is negative, it maps to N+m.
    
    F_inv_eps = np.fft.fft2(inv_eps) / (N * N)
    
    # Map G vectors to FFT indices
    # G = (m, n). Index in F_inv_eps is (m % N, n % N).
    # Since m, n are small (-8 to 8) and N=32, no aliasing issues if we handle negatives.
    f_G = np.zeros(n_g, dtype=complex)
    for idx, (m, n) in enumerate(G_vectors):
        im = m % N
        in_ = n % N
        f_G[idx] = F_inv_eps[im, in_]
        
    # Define k-points in IBZ
    # IBZ for square lattice: 0 <= ky <= kx, 0 <= kx <= pi/a
    # We sample a few points: Gamma, M, X, and some interior.
    k_points = []
    # Gamma (0,0)
    k_points.append((0.0, 0.0))
    # M (pi, pi)
    k_points.append((np.pi, np.pi))
    # X (pi, 0)
    k_points.append((np.pi, 0.0))
    # Interior points along M-X line and others
    # We need to cover the IBZ to find min omega_5 and max omega_4.
    # Let's use a small grid in the IBZ.
    # IBZ is triangle with vertices (0,0), (pi,0), (pi,pi).
    # Wait, standard square lattice IBZ is the square [-pi, pi]x[-pi, pi] reduced by symmetry.
    # Symmetry group D4.
    # Fundamental domain: 0 <= ky <= kx <= pi.
    
    # Let's generate points in this triangle.
    n_grid = 4
    for i in range(n_grid + 1):
        for j in range(n_grid + 1):
            kx = np.pi * i / n_grid
            ky = np.pi * j / n_grid
            if ky <= kx:
                k_points.append((kx, ky))
                
    # Remove duplicates roughly
    unique_k = []
    for kp in k_points:
        is_dup = False
        for uk in unique_k:
            if abs(kp[0]-uk[0]) < 1e-5 and abs(kp[1]-uk[1]) < 1e-5:
                is_dup = True
                break
        if not is_dup:
            unique_k.append(kp)
    k_points = unique_k
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    for kx, ky in k_points:
        # Construct PWE matrix
        # M_{G', G} = (k+G') . (k+G) / (2pi)^2 * f_{G'-G}  (for TM, simplified)
        # Actually, the eigenvalue problem is:
        # sum_G ( (k+G') . (k+G) / (2pi)^2 ) f_{G'-G} E_G = (omega/c)^2 E_{G'}
        # Let's use units where c=1, a=1.
        
        M = np.zeros((n_g, n_g), dtype=complex)
        
        # Precompute k+G vectors
        KG = G_vectors + np.array([kx, ky])
        
        for i_g in range(n_g):
            for j_g in range(n_g):
                # Need f_{G_i - G_j}
                # G_i - G_j = (m_i - m_j, n_i - n_j)
                dm = G_vectors[i_g, 0] - G_vectors[j_g, 0]
                dn = G_vectors[i_g, 1] - G_vectors[j_g, 1]
                
                # Find index of (dm, dn) in G_vectors?
                # No, f_G is defined for G in our set.
                # If G_i - G_j is not in G_vectors, we need its coefficient.
                # But we only have f_G for G in G_vectors.
                # We can compute f_{G_diff} directly from F_inv_eps if G_diff is within range.
                # Since G_vectors is a square grid, the difference is also in a larger square grid.
                # We can just look up F_inv_eps[dm % N, dn % N].
                
                f_diff = F_inv_eps[dm % N, dn % N]
                
                # Dot product (k+G_i) . (k+G_j)
                dot = KG[i_g, 0] * KG[j_g, 0] + KG[i_g, 1] * KG[j_g, 1]
                
                M[i_g, j_g] = dot * f_diff / (2 * np.pi) ** 2
                
        # Solve eigenvalue problem
        # M is Hermitian
        eigvals = np.linalg.eigvalsh(M)
        eigvals = np.sort(eigvals.real)
        
        # We want bands 4 and 5 (indices 3 and 4)
        if len(eigvals) > 4:
            w4 = np.sqrt(eigvals[3])
            w5 = np.sqrt(eigvals[4])
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
    if min_omega_5 == np.inf:
        return 0.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap == 0:
        return 0.0
        
    return gap / midgap

def design(problem):
    """
    Design a 32x32 unit cell with four dielectric rods in the quadrants.
    Optimizes radius and center using a fast PWE surrogate.
    """
    size = problem["pixels"]
    half = size // 2
    
    # Initial parameters
    cx_init = 7.5
    cy_init = 7.5
    r_init = 5.0
    
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0 # 10s margin
    
    best_score = -np.inf
    best_params = (cx_init, cy_init, r_init)
    
    # Local search around initial parameters
    # Step sizes
    delta_r = 0.1
    delta_c = 0.1
    
    # Coarse search first
    for r in np.arange(4.5, 5.6, delta_r):
        if time.time() - start_time > time_limit:
            break
        for cx in np.arange(7.0, 8.1, delta_c):
            if time.time() - start_time > time_limit:
                break
            # Symmetry requires cy = cx if we maintain C4v with identical quadrants?
            # The current code uses cx=cy. Let's stick to cx=cy for simplicity and symmetry.
            cy = cx
            # Generate cell
            quadrant = np.zeros((half, half), dtype=int)
            for i in range(half):
                for j in range(half):
                    x = i + 0.5
                    y = j + 0.5
                    if (x - cx)**2 + (y - cy)**2 < r**2:
                        quadrant[i][j] = 1
                        
            cell = np.zeros((size, size), dtype=int)
            cell[:half, :half] = quadrant
            cell[:half, half:] = quadrant[:, ::-1]
            cell[half:, :half] = quadrant[::-1, :]
            cell[half:, half:] = quadrant[::-1, ::-1]
            
            # Evaluate
            score = _fast_pwe_gap(cell)
            
            if score > best_score:
                best_score = score
                best_params = (cx, cy, r)
                
    # Fine search around best
    cx_best, cy_best, r_best = best_params
    for _ in range(5): # 5 iterations of refinement
        if time.time() - start_time > time_limit:
            break
        delta_r = 0.05
        delta_c = 0.05
        for r in np.arange(r_best - 0.2, r_best + 0.2, delta_r):
            if time.time() - start_time > time_limit:
                break
            for cx in np.arange(cx_best - 0.2, cx_best + 0.2, delta_c):
                if time.time() - start_time > time_limit:
                    break
                cy = cx
                quadrant = np.zeros((half, half), dtype=int)
                for i in range(half):
                    for j in range(half):
                        x = i + 0.5
                        y = j + 0.5
                        if (x - cx)**2 + (y - cy)**2 < r**2:
                            quadrant[i][j] = 1
                            
                cell = np.zeros((size, size), dtype=int)
                cell[:half, :half] = quadrant
                cell[:half, half:] = quadrant[:, ::-1]
                cell[half:, :half] = quadrant[::-1, :]
                cell[half:, half:] = quadrant[::-1, ::-1]
                
                score = _fast_pwe_gap(cell)
                
                if score > best_score:
                    best_score = score
                    best_params = (cx, cy, r)
                    cx_best, cy_best, r_best = cx, cy, r

    # Generate final cell from best parameters
    cx, cy, r = best_params
    quadrant = [[0 for _ in range(half)] for _ in range(half)]
    for i in range(half):
        for j in range(half):
            x = i + 0.5
            y = j + 0.5
            if (x - cx)**2 + (y - cy)**2 < r**2:
                quadrant[i][j] = 1
                
    cell = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(half):
        for j in range(half):
            val = quadrant[i][j]
            cell[i][j] = val
            cell[i][size - 1 - j] = val
            cell[size - 1 - i][j] = val
            cell[size - 1 - i][size - 1 - j] = val
            
    return cell
# EVOLVE-BLOCK-END
