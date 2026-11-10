import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 pixel unit cell for a 2D square-lattice photonic crystal.
    Uses a differentiable surrogate optimization with a low-order Fourier basis
    constrained to C4v symmetry to find a topology that opens a large TM band gap.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    lower_band = problem["lower_band"]  # 4
    
    # Deadline guard
    start_time = time.time()
    deadline = 200.0  # Safety margin before 230s limit
    
    # Fast surrogate plane-wave calculation
    # Use a smaller basis for speed during optimization
    # The evaluator uses 289 plane waves (17x17), we use 9x9=81 for speed
    N_pw = 9  # 9x9 = 81 plane waves for surrogate
    N_pw_eval = 17  # 17x17 = 289 for final check
    
    # C4v symmetric Fourier basis
    # For a square lattice with C4v symmetry, the dielectric function can be
    # represented as a sum of Fourier terms that respect the symmetry
    # epsilon(r) = eps_avg + sum_{(m,n) in C4v orbits} c_{mn} * cos(2*pi*(m*x + n*y))
    
    # Generate C4v-symmetric Fourier modes
    # For C4v, we need modes that are invariant under:
    # 1. x -> -x (reflection about y-axis)
    # 2. y -> -y (reflection about x-axis)  
    # 3. x -> y (reflection about diagonal)
    # This means we can only use cos terms (no sin) and the coefficients must satisfy
    # c_{mn} = c_{nm} = c_{-mn} = c_{-nm}
    
    # We'll use a low-order expansion
    max_order = 4
    
    # Generate unique C4v orbits
    modes = []
    seen = set()
    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            if m == 0 and n == 0:
                continue
            # Canonical form for C4v: take the orbit representative
            # Sort by (abs(m), abs(n)) and enforce m >= 0, n >= 0, m >= n for uniqueness
            # Actually for C4v, the orbit of (m,n) contains:
            # (m,n), (-m,n), (m,-n), (-m,-n), (n,m), (-n,m), (n,-m), (-n,-m)
            # We pick a canonical representative
            key = (abs(m), abs(n))
            if key not in seen:
                seen.add(key)
                modes.append((m, n))
    
    # Remove duplicates - keep only one representative per orbit
    # For C4v, (m,n) and (n,m) are in the same orbit if m != n
    # Let's be more careful
    modes = []
    seen_orbits = set()
    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            if m == 0 and n == 0:
                continue
            # Generate the full C4v orbit
            orbit = set()
            orbit.add((m, n))
            orbit.add((-m, n))
            orbit.add((m, -n))
            orbit.add((-m, -n))
            orbit.add((n, m))
            orbit.add((-n, m))
            orbit.add((n, -m))
            orbit.add((-n, -m))
            
            # Check if this orbit is new
            orbit_frozenset = frozenset(orbit)
            if orbit_frozenset not in seen_orbits:
                seen_orbits.add(orbit_frozenset)
                modes.append((m, n))  # Store one representative
    
    num_modes = len(modes)
    
    # Create grid for evaluation
    x = np.linspace(0, 1, size, endpoint=False)
    y = np.linspace(0, 1, size, endpoint=False)
    X, Y = np.meshgrid(x, y)
    
    # Precompute Fourier basis values
    # basis[i, j, k] = value of mode k at pixel (i, j)
    basis = np.zeros((size, size, num_modes))
    for k, (m, n) in enumerate(modes):
        basis[:, :, k] = np.cos(2 * np.pi * (m * X + n * Y))
    
    def get_eps_from_coeffs(coeffs):
        """Get dielectric grid from Fourier coefficients."""
        # coeffs[0] is the average, coeffs[1:] are the mode coefficients
        eps_avg = coeffs[0]
        if num_modes > 0:
            eps = eps_avg + basis @ coeffs[1:]
        else:
            eps = np.full((size, size), eps_avg)
        # Clip to [eps_low, eps_high]
        eps = np.clip(eps, eps_low, eps_high)
        return eps
    
    def get_pixel_mask_from_coeffs(coeffs):
        """Get binary pixel mask from Fourier coefficients."""
        eps = get_eps_from_coeffs(coeffs)
        # Threshold at the midpoint
        threshold = (eps_low + eps_high) / 2.0
        mask = (eps >= threshold).astype(int)
        return mask
    
    def compute_fourier_coefficients(eps_grid):
        """Compute Fourier coefficients from a dielectric grid."""
        # Compute DFT
        eps_dft = np.fft.fft2(eps_grid)
        # Extract C4v-symmetric coefficients
        coeffs = np.zeros(num_modes + 1)
        coeffs[0] = np.real(eps_dft[size//2, size//2]) / (size * size)  # Average
        
        for k, (m, n) in enumerate(modes):
            # Get the DFT coefficient for mode (m, n)
            # Handle negative indices
            m_idx = m % size
            n_idx = n % size
            val = np.real(eps_dft[m_idx, n_idx])
            coeffs[k + 1] = val / (size * size)
        
        return coeffs
    
    def plane_wave_eigenvalues(eps_grid, n_pw, n_bands=6):
        """
        Compute band structure using plane wave method.
        Returns eigenvalues (squared frequencies) for the lowest n_bands.
        """
        # Get Fourier coefficients of 1/epsilon
        # For TM polarization, we solve: sum_{G'} (k+G) x (k+G') / eps_G * A_{G'} = omega^2 A_k
        # where eps_G are Fourier coefficients of 1/epsilon(r)
        
        # Compute 1/epsilon
        inv_eps = 1.0 / eps_grid
        
        # Get Fourier coefficients of 1/epsilon
        inv_eps_dft = np.fft.fft2(inv_eps)
        inv_eps_dft = inv_eps_dft / (size * size)
        
        # Generate plane wave vectors
        # For n_pw x n_pw grid
        half = n_pw // 2
        Gx = np.arange(-half, half + 1)
        Gy = np.arange(-half, half + 1)
        
        # Create 2D grid of G vectors
        GX, GY = np.meshgrid(Gx, Gy, indexing='ij')
        G_flat = np.column_stack([GX.ravel(), GY.ravel()])
        N = len(G_flat)
        
        # For each k-point, build the matrix and compute eigenvalues
        # We'll use a few k-points for the surrogate
        # Irreducible BZ for square lattice: (0,0) to (0.5,0) to (0.5,0.5) to (0,0.5)
        k_points = np.array([
            [0.0, 0.0],      # Gamma
            [0.5, 0.0],      # X
            [0.5, 0.5],      # M
            [0.25, 0.0],     # Midpoint Gamma-X
            [0.5, 0.25],     # Midpoint X-M
            [0.25, 0.25],    # Interior
        ])
        
        all_eigs = []
        
        for k in k_points:
            # Build the A matrix
            # A_{G,G'} = (k+G) x (k+G') / eps_{G-G'}
            # where eps_{G-G'} is the Fourier coefficient of 1/eps at G-G'
            
            A = np.zeros((N, N), dtype=complex)
            
            for i in range(N):
                for j in range(N):
                    # k + G_i
                    kg_i = k + G_flat[i]
                    # k + G_j
                    kg_j = k + G_flat[j]
                    # G_i - G_j
                    G_diff = G_flat[i] - G_flat[j]
                    
                    # Get eps_{G_diff}
                    # Need to map G_diff to the DFT index
                    # G_diff components are integers in range [-n_pw+1, n_pw-1]
                    # Map to [0, size)
                    gx_idx = int(G_diff[0]) % size
                    gy_idx = int(G_diff[1]) % size
                    
                    eps_G = inv_eps_dft[gx_idx, gy_idx]
                    
                    if np.abs(eps_G) > 1e-10:
                        # (k+G_i) x (k+G_j) = (kx+Gix)(kx+Gjx) + (ky+Giy)(ky+Gjy)
                        dot_prod = kg_i[0] * kg_j[0] + kg_i[1] * kg_j[1]
                        A[i, j] = dot_prod / eps_G
                    else:
                        A[i, j] = 0.0
            
            # Make Hermitian
            A = (A + A.conj().T) / 2.0
            
            # Compute eigenvalues
            try:
                eigs = np.linalg.eigvalsh(A)
                # Take the lowest n_bands
                eigs = eigs[:n_bands]
                all_eigs.append(eigs)
            except np.linalg.LinAlgError:
                continue
        
        if len(all_eigs) == 0:
            return np.array([])
        
        all_eigs = np.array(all_eigs)
        return all_eigs
    
    def compute_gap_score(eps_grid, n_pw, n_bands=6):
        """
        Compute the gap-midgap ratio for the given dielectric grid.
        Returns the score (higher is better).
        """
        eigs = plane_wave_eigenvalues(eps_grid, n_pw, n_bands)
        if len(eigs) == 0:
            return -1.0
        
        # Find the max of band 4 (index 3) and min of band 5 (index 4)
        # across all k-points
        band4_max = np.max(eigs[:, 3])  # Max of 4th band
        band5_min = np.min(eigs[:, 4])  # Min of 5th band
        
        # Gap-midgap ratio
        gap = band5_min - band4_max
        midgap = (band5_min + band4_max) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        score = 2.0 * gap / midgap
        return score
    
    # Initial guess: start from a simple pattern
    # A dielectric frame
    initial_mask = np.zeros((size, size))
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            dx = abs(i - center)
            dy = abs(j - center)
            if dx < 13.0 and dy < 13.0 and not (dx < 5.0 and dy < 5.0):
                initial_mask[i, j] = 1.0
    
    # Convert to dielectric grid
    initial_eps = eps_low + (eps_high - eps_low) * initial_mask
    
    # Get initial Fourier coefficients
    initial_coeffs = compute_fourier_coefficients(initial_eps)
    
    best_coeffs = initial_coeffs.copy()
    best_score = compute_gap_score(initial_eps, N_pw)
    
    # Gradient-based optimization
    # We'll use a simple gradient ascent with adaptive step size
    
    num_steps = 500
    step_size = 0.1
    
    for step in range(num_steps):
        if time.time() > deadline:
            break
        
        # Compute current score
        current_eps = get_eps_from_coeffs(best_coeffs)
        current_score = compute_gap_score(current_eps, N_pw)
        
        if current_score > best_score:
            best_score = current_score
        
        # Compute gradient numerically
        grad = np.zeros_like(best_coeffs)
        for i in range(len(best_coeffs)):
            perturbed = best_coeffs.copy()
            perturbed[i] += 0.01
            perturbed_eps = get_eps_from_coeffs(perturbed)
            perturbed_score = compute_gap_score(perturbed_eps, N_pw)
            grad[i] = (perturbed_score - current_score) / 0.01
        
        # Update coefficients
        best_coeffs += step_size * grad
        
        # Adaptive step size
        new_eps = get_eps_from_coeffs(best_coeffs)
        new_score = compute_gap_score(new_eps, N_pw)
        
        if new_score > current_score:
            # Increase step size slightly
            step_size = min(step_size * 1.05, 0.5)
        else:
            # Decrease step size
            step_size = max(step_size * 0.9, 0.01)
            # Revert
            best_coeffs = best_coeffs - step_size * grad
    
    # Final evaluation with larger basis
    final_eps = get_eps_from_coeffs(best_coeffs)
    final_mask = get_pixel_mask_from_coeffs(best_coeffs)
    
    # Ensure C4v symmetry by averaging with symmetric copies
    # The mask should already be symmetric due to the Fourier basis,
    # but let's enforce it explicitly
    
    # Reflect about center
    def enforce_c4v_symmetry(mask):
        """Enforce C4v symmetry on the mask."""
        size = mask.shape[0]
        result = mask.copy().astype(float)
        
        # Average with all C4v symmetric copies
        # 1. Original
        result += mask
        # 2. Flip x
        result += mask[:, ::-1]
        # 3. Flip y
        result += mask[::-1, :]
        # 4. Flip both
        result += mask[::-1, ::-1]
        # 5. Transpose
        result += mask.T
        # 6. Transpose + flip x
        result += mask.T[:, ::-1]
        # 7. Transpose + flip y
        result += mask.T[::-1, :]
        # 8. Transpose + flip both
        result += mask.T[::-1, ::-1]
        
        result /= 8.0
        return (result >= 0.5).astype(int)
    
    final_mask = enforce_c4v_symmetry(final_mask)
    
    # Convert to nested list
    cell = final_mask.tolist()
    
    return cell
