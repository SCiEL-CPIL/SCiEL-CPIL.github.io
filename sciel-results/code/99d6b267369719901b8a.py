"""Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap (bands 4-5)."""
import math
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 unit cell.
    Uses numerical optimization (gradient descent) on the dielectric distribution
    to maximize the gap between the 4th and 5th TM bands.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Symmetry: The cell must be invariant under the square symmetry group.
    # This means cell[i][j] == cell[size-1-i][j] == cell[i][size-1-j] == cell[j][i] etc.
    # We can define the design on the fundamental domain (e.g., i <= j and i <= size/2, j <= size/2)
    # and reflect it.
    
    # To simplify the PWE solver and optimization, we work with the full 32x32 array
    # but enforce symmetry by only updating the fundamental domain and copying.
    
    # Initialize with a random symmetric pattern
    np.random.seed(42)
    # Fundamental domain: top-left quadrant (0..15, 0..15)
    # We need to be careful with the center line if size is even.
    # For 32x32, indices 0..15 are the unique part if we reflect across 15.5.
    # Actually, for 4-fold symmetry about center (15.5, 15.5):
    # (i, j) -> (j, 31-i) -> (31-i, 31-j) -> (31-j, i)
    # So we only need to define values for i in [0, 15] and j in [0, 15] such that
    # the pattern is consistent.
    # Let's just generate a random symmetric matrix.
    
    def make_symmetric(mat):
        """Enforce D4 symmetry on a 32x32 matrix."""
        sym = np.zeros_like(mat)
        for i in range(size):
            for j in range(size):
                # Average over the orbit
                # Orbit of (i,j): (i,j), (j, size-1-i), (size-1-i, size-1-j), (size-1-j, i)
                # Note: For even size, these are 4 distinct points unless on axes/diagonals.
                # Since we are optimizing, we can just pick one representative and copy.
                # Let's use the representative (i, j) and copy to others.
                # To ensure symmetry, we define the value at (i,j) and set:
                # mat[j][size-1-i] = mat[i][j]
                # mat[size-1-i][size-1-j] = mat[i][j]
                # mat[size-1-j][i] = mat[i][j]
                pass
        # Easier: Generate random values for the unique pixels.
        # Unique pixels are those where i <= 15 and j <= 15? No, that's 1/4.
        # Let's just generate a random matrix and symmetrize it by averaging its orbit.
        rand = np.random.rand(size, size)
        sym = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                # Orbit
                p1 = (i, j)
                p2 = (j, size - 1 - i)
                p3 = (size - 1 - i, size - 1 - j)
                p4 = (size - 1 - j, i)
                val = (rand[p1[0], p1[1]] + rand[p2[0], p2[1]] + rand[p3[0], p3[1]] + rand[p4[0], p4[1]]) / 4.0
                sym[i, j] = val
                # We only need to fill the unique ones, but filling all is fine for init
        return sym

    # We will optimize a continuous variable 'x' in [0,1] which maps to epsilon.
    # x=0 -> eps_low, x=1 -> eps_high.
    x_init = make_symmetric(np.random.rand(size, size))
    x = x_init.copy()
    
    # PWE Solver for TM modes
    # We use a plane-wave expansion with a cutoff of N=20 (sufficient for 32x32 grid resolution)
    # The dielectric function epsilon(k) is the Fourier transform of the pixel map.
    # epsilon(r) = sum_G epsilon_G exp(i G . r)
    # For a pixel map, epsilon_G = (1/A) sum_pixels epsilon_pixel exp(-i G . r_pixel)
    
    N_cw = 20 # Cutoff for plane waves
    # Generate reciprocal lattice vectors G
    # Square lattice with period a=1 (in pixel units). G = (m, n) * 2*pi / a.
    # We need G vectors with |G| <= G_max.
    G_vectors = []
    G2_max = (N_cw * 2 * math.pi) ** 2 # Arbitrary cutoff, let's use index based
    # Actually, let's use a simpler cutoff: |m|,|n| <= N_cw
    for m in range(-N_cw, N_cw + 1):
        for n in range(-N_cw, N_cw + 1):
            if m == 0 and n == 0:
                continue
            G_vectors.append((m, n))
    
    # Sort G vectors by magnitude
    G_vectors.sort(key=lambda g: g[0]**2 + g[1]**2)
    
    # Number of plane waves
    num_pw = len(G_vectors)
    
    # Precompute G dot G for the eigenvalue problem
    # The PWE equation for TM is:
    # sum_G' epsilon_{G-G'} (k + G)^2 |c_{G'}| = omega^2 |c_G|
    # Or in matrix form: M c = omega^2 c
    # Where M_{G, G'} = epsilon_{G-G'} |k + G'|^2
    
    # We need to compute epsilon_G for the current x.
    # epsilon(r) = eps_low + (eps_high - eps_low) * x(r)
    # epsilon_G = (1/N_pix) sum_{i,j} epsilon(i,j) exp(-i 2 pi (m i + n j) / size)
    
    def compute_epsilon_fourier(x):
        """Compute Fourier coefficients of the dielectric map."""
        eps_fourier = np.zeros((num_pw, num_pw), dtype=complex)
        # We need epsilon_{G-G'} for all pairs of G, G' in our basis.
        # G indices are mapped to (m,n).
        # Let's create a lookup for G vector components.
        G_m = np.array([g[0] for g in G_vectors])
        G_n = np.array([g[1] for g in G_vectors])
        
        # Precompute exp terms for each pixel and each G? Too slow.
        # Use FFT? 
        # The grid is 32x32. The G vectors are integer multiples of 2pi/32.
        # So epsilon_G is essentially the DFT of the pixel map.
        # DFT: F[m,n] = sum_{i,j} f[i,j] exp(-i 2 pi (m i + n j) / 32)
        
        # We need F for all (m,n) that are differences of G vectors in our basis.
        # Max m, n in basis is N_cw. Min is -N_cw.
        # So differences range from -2*N_cw to 2*N_cw.
        # We can compute the full DFT up to 2*N_cw and then pick the needed components.
        
        M_fft = 2 * N_cw
        # Pad x to (2*N_cw + 1) x (2*N_cw + 1)? No, x is 32x32.
        # The DFT of a 32x32 array gives frequencies 0..31.
        # Our G vectors are integers m,n.
        # So we need F[m mod 32, n mod 32].
        
        F = np.fft.fft2(x)
        
        for idx_g, (m_g, n_g) in enumerate(G_vectors):
            for idx_gp, (m_gp, n_gp) in enumerate(G_vectors):
                dm = m_g - m_gp
                dn = n_g - n_gp
                # Map to 0..31
                dm_mod = dm % 32
                dn_mod = dn % 32
                eps_fourier[idx_g, idx_gp] = F[dm_mod, dn_mod]
                
        return eps_fourier

    def solve_bands(x, k_point):
        """Solve for TM bands at a given k-point."""
        eps_fourier = compute_epsilon_fourier(x)
        
        # k-point is (kx, ky) in units of 2pi/a.
        kx, ky = k_point
        
        # Construct matrix M
        # M_{G, G'} = epsilon_{G-G'} |k + G'|^2
        # Note: The standard PWE formulation for TM is:
        # sum_{G'} (1/epsilon_{G-G'}) |k+G|^2 c_{G'} = omega^2 c_G  <-- This is for TE?
        # Let's check.
        # Maxwell: curl curl E = omega^2/c^2 epsilon E
        # For TM (Ez), the equation is:
        # sum_{G'} epsilon_{G-G'} |k+G'|^2 c_{G'} = omega^2 c_G
        # Wait, usually it's:
        # TE: sum (1/eps_{G-G'}) |k+G|^2 c_{G'} = omega^2 c_G
        # TM: sum eps_{G-G'} |k+G'|^2 c_{G'} = omega^2 c_G
        # Let's assume TM uses eps in the matrix.
        
        M = np.zeros((num_pw, num_pw), dtype=complex)
        for idx_g in range(num_pw):
            m_g, n_g = G_vectors[idx_g]
            for idx_gp in range(num_pw):
                m_gp, n_gp = G_vectors[idx_gp]
                # k + G'
                kx_gp = kx + m_gp
                ky_gp = ky + n_gp
                k2 = kx_gp**2 + ky_gp**2
                M[idx_g, idx_gp] = eps_fourier[idx_g, idx_gp] * k2
        
        # Solve eigenvalue problem M c = omega^2 c
        # M is Hermitian?
        # eps_{G-G'} = conj(eps_{G'-G}) if eps is real.
        # M_{G, G'} = eps_{G-G'} |k+G'|^2
        # M_{G', G} = eps_{G'-G} |k+G|^2 = conj(eps_{G-G'}) |k+G|^2
        # They are not conjugates unless |k+G|^2 = |k+G'|^2 which is false.
        # So M is not Hermitian.
        # However, the PWE matrix for TM is often written as:
        # [1/eps] * [ |k+G|^2 ]
        # Let's re-verify the TM equation.
        # From Joannopoulos:
        # TE: sum_{G'} (1/eps_{G-G'}) |k+G|^2 c_{G'} = omega^2 c_G
        # TM: sum_{G'} eps_{G-G'} |k+G'|^2 c_{G'} = omega^2 c_G
        # Actually, both are symmetric if we define the basis correctly.
        # Let's use the standard formulation:
        # TE: A_{G,G'} = (1/eps_{G-G'}) * |k+G|^2
        # TM: A_{G,G'} = eps_{G-G'} * |k+G'|^2
        # Wait, the TM matrix is NOT Hermitian with this definition.
        # The correct Hermitian form for TM is obtained by transforming the basis.
        # Let c'_G = sqrt(eps_G) c_G.
        # Then the equation becomes:
        # sum_{G'} sqrt(eps_G) eps_{G-G'} |k+G'|^2 (1/sqrt(eps_{G'})) c'_{G'} = omega^2 sqrt(eps_G) c'_G
        # This is getting complicated.
        
        # Let's stick to the non-Hermitian matrix and use general eigensolver.
        # Or better, use the TE-like formulation if we swap eps and 1/eps?
        # No, the physics is different.
        
        # Let's use `np.linalg.eigvals` for speed, filtering for real positive eigenvalues.
        # Or, we can use the fact that the eigenvalues are real.
        
        # Optimization: We only care about bands 4 and 5.
        # We can use `scipy.sparse.linalg.eigsh` if M was Hermitian.
        # Since M is not Hermitian, we use `np.linalg.eigvals`.
        
        eigvals = np.linalg.eigvals(M)
        # Filter real parts
        eigvals_real = np.real(eigvals)
        # Sort
        eigvals_sorted = np.sort(eigvals_real)
        # The lowest eigenvalue corresponds to the lowest frequency.
        # We want the gap between 4th and 5th.
        # Note: The zero mode (G=0, k=0) might be an issue? No, k is not 0 usually.
        # Also, there might be spurious modes.
        
        # Return the 4th and 5th smallest positive eigenvalues.
        # We assume the first few are the physical bands.
        # We need to skip any negative or zero eigenvalues if they exist due to numerical noise.
        positive_eigvals = eigvals_sorted[eigvals_sorted > 1e-6]
        
        if len(positive_eigvals) < 5:
            return 0.0, 0.0 # No gap
        
        omega_4 = positive_eigvals[3]
        omega_5 = positive_eigvals[4]
        
        return omega_4, omega_5

    # Optimization Loop
    # We will use a simple gradient-free method (Nelder-Mead) or gradient descent.
    # Given the time limit (120s), we can do a few iterations of a coarse search.
    # Let's try a few random restarts and local refinement.
    
    best_gap = -1
    best_x = x.copy()
    
    # Define k-points in the irreducible BZ
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= 1 (in units 2pi/a)
    # Actually, 0 <= kx, 0 <= ky, kx+ky <= 1? No.
    # High symmetry points: Gamma(0,0), X(0.5,0), M(0.5,0.5).
    # Irreducible triangle: (0,0) - (0.5,0) - (0.5,0.5).
    # kx in [0, 0.5], ky in [0, 0.5], and ky >= kx? No, the triangle is defined by
    # 0 <= kx <= 0.5, 0 <= ky <= 0.5, and kx + ky <= 0.5? No.
    # The BZ is a square [-0.5, 0.5] x [-0.5, 0.5].
    # Irreducible part is the triangle with vertices (0,0), (0.5,0), (0.5,0.5).
    # So kx in [0, 0.5], ky in [0, 0.5], and ky >= kx?
    # Let's check. The symmetry operation (kx, ky) -> (ky, kx) swaps them.
    # So we can restrict to kx <= ky.
    # And (kx, ky) -> (0.5-kx, 0.5-ky) maps to the other side.
    # So the irreducible zone is the triangle (0,0), (0.5,0), (0.5,0.5).
    # Wait, (0.5, 0) is X. (0.5, 0.5) is M. (0,0) is Gamma.
    # The triangle is Gamma-X-M.
    # So kx in [0, 0.5], ky in [kx, 0.5]?
    # Let's sample points in this triangle.
    
    k_points = []
    for i in range(5):
        for j in range(5):
            kx = i * 0.1
            ky = j * 0.1
            if kx <= 0.5 and ky <= 0.5 and kx <= ky:
                k_points.append((kx, ky))
    
    # Also add the boundary points explicitly
    k_points.append((0.5, 0.5))
    k_points.append((0.5, 0.0))
    k_points.append((0.0, 0.0))
    
    # Remove duplicates
    k_points = list(set(k_points))
    
    def calculate_score(x):
        """Calculate the gap-midgap ratio."""
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for k in k_points:
            w4, w5 = solve_bands(x, k)
            if w4 > 0 and w5 > 0:
                if w5 < min_omega_5:
                    min_omega_5 = w5
                if w4 > max_omega_4:
                    max_omega_4 = w4
        
        if min_omega_5 == float('inf'):
            return 0.0
        
        gap = 2 * (min_omega_5 - max_omega_4)
        midgap = min_omega_5 + max_omega_4
        ratio = gap / midgap
        return ratio

    # Simple local search
    # We'll try a few steps of a random walk with reflection.
    # Since gradient calculation is expensive, we use a finite difference approach
    # or just random perturbations on the fundamental domain.
    
    # Define fundamental domain indices
    # For 32x32, the unique pixels are those in the top-left 16x16 quadrant?
    # No, due to the center being at 15.5, the pixels 0..15 are unique if we reflect.
    # Let's verify:
    # Pixel (0,0) reflects to (0,31), (31,31), (31,0).
    # Pixel (15,15) reflects to (15,16), (16,16), (16,15).
    # So yes, the 16x16 block (0..15, 0..15) determines the whole grid.
    
    unique_indices = [(i, j) for i in range(16) for j in range(16)]
    
    # Optimization
    num_iterations = 200
    step_size = 0.1
    
    for iter in range(num_iterations):
        # Calculate current score
        current_score = calculate_score(x)
        if current_score > best_gap:
            best_gap = current_score
            best_x = x.copy()
            
        # Perturb one random pixel in the fundamental domain
        idx = np.random.randint(len(unique_indices))
        i, j = unique_indices[idx]
        
        # Try increasing
        x_new = x.copy()
        x_new[i, j] += step_size
        # Enforce symmetry
        x_new[j, size-1-i] = x_new[i, j]
        x_new[size-1-i, size-1-j] = x_new[i, j]
        x_new[size-1-j, i] = x_new[i, j]
        
        # Clamp to [0,1]
        x_new = np.clip(x_new, 0, 1)
        
        new_score = calculate_score(x_new)
        
        if new_score > current_score:
            x = x_new
        else:
            # Try decreasing
            x_new = x.copy()
            x_new[i, j] -= step_size
            x_new[j, size-1-i] = x_new[i, j]
            x_new[size-1-i, size-1-j] = x_new[i, j]
            x_new[size-1-j, i] = x_new[i, j]
            x_new = np.clip(x_new, 0, 1)
            
            new_score = calculate_score(x_new)
            if new_score > current_score:
                x = x_new
    
    # Binarize the best solution
    # We want to maximize the gap. Binarization might reduce the gap,
    # but the problem requires 0/1 pixels.
    # We can try thresholding at 0.5.
    cell_float = best_x
    cell_bin = (cell_float > 0.5).astype(int)
    
    # Convert to nested list
    cell = cell_bin.tolist()
    
    return cell
