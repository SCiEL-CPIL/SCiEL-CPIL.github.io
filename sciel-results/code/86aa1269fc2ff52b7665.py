"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """Optimize dielectric disk radius and position for TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 200.0  # Leave safety margin before 230s kill
    
    def compute_cell(radius, cx, cy):
        """Create a 32x32 cell with a circular dielectric region."""
        cell = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                    cell[i, j] = 1
        # Ensure symmetry: reflect across center
        cell = (cell + cell[::-1, ::-1]) // 2
        return cell.tolist()
    
    def band_gap_score(cell_list):
        """Approximate band gap score using plane-wave method."""
        cell = np.array(cell_list, dtype=float)
        # Compute Fourier coefficients for the dielectric function
        # epsilon(x,y) = 1 + (eps_high - 1) * cell(x,y)
        eps = 1.0 + (eps_high - 1.0) * cell
        
        # Use a reasonable number of plane waves (N=17 gives 289 total)
        N = 17
        # Fourier coefficients
        G = np.fft.fftfreq(N, d=1.0/N)
        Gx, Gy = np.meshgrid(G, G, indexing='ij')
        G2 = Gx**2 + Gy**2
        
        # Compute Fourier coefficients of epsilon
        eps_fourier = np.zeros((N, N), dtype=complex)
        for mx in range(N):
            for my in range(N):
                kx = 2 * np.pi * Gx[mx, my]
                ky = 2 * np.pi * Gy[mx, my]
                # Sample at pixel centers
                phase = np.exp(-1j * (kx * (np.arange(size) + 0.5) / size * 2 * np.pi + 
                                     ky * (np.arange(size) + 0.5) / size * 2 * np.pi))
                eps_fourier[mx, my] = np.sum(eps * phase[:, None] * phase[None, :]) / (size * size)
        
        # Find band structure along high-symmetry directions
        # Test several k-points in the irreducible BZ
        k_points = [
            (0.0, 0.0),      # Gamma
            (0.5, 0.0),      # X
            (0.5, 0.5),      # M
            (0.25, 0.0),
            (0.25, 0.25),
            (0.3, 0.2),
            (0.4, 0.1),
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            # Build the eigenvalue problem for TM polarization
            # (1/G^2) * sum_G' epsilon(-G-G') |G'|^2 c(G') = omega^2 c(G)
            # Simplified: use a smaller basis for speed
            basis_size = 9  # 3x3 grid of G vectors for speed
            G_indices = []
            for mx in range(-1, 2):
                for my in range(-1, 2):
                    G_indices.append((mx, my))
            
            N_basis = len(G_indices)
            A = np.zeros((N_basis, N_basis), dtype=complex)
            
            for i, (gx_i, gy_i) in enumerate(G_indices):
                G2_i = (gx_i + kx * size / 2) ** 2 + (gy_i + ky * size / 2) ** 2
                if G2_i == 0:
                    G2_i = 1e-10  # Avoid division by zero at Gamma
                for j, (gx_j, gy_j) in enumerate(G_indices):
                    dgx = gx_i - gx_j
                    dgy = gy_i - gy_j
                    # Look up Fourier coefficient
                    fx = int(dgx) % N
                    fy = int(dgy) % N
                    eps_f = eps_fourier[fx, fy]
                    A[i, j] = eps_f / G2_i
            
            # Solve eigenvalue problem
            try:
                eigenvalues = np.linalg.eigvalsh(A)
                eigenvalues = np.real(eigenvalues)
                eigenvalues = np.sort(eigenvalues)
                
                if len(eigenvalues) >= 5:
                    max_omega4 = max(max_omega4, np.sqrt(eigenvalues[3]))
                    min_omega5 = min(min_omega5, np.sqrt(eigenvalues[4]))
            except:
                pass
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return -1.0
        
        gap = 2 * (min_omega5 - max_omega4)
        midgap = min_omega5 + max_omega4
        if midgap == 0:
            return -1.0
        return gap / midgap
    
    # Start with the seed design
    best_cell = None
    best_score = -np.inf
    
    # Try various radii
    radii_to_try = [0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.32, 0.35]
    
    for radius in radii_to_try:
        if time.time() - start_time > time_limit - 10:
            break
        
        cell = compute_cell(radius, 0.0, 0.0)
        score = band_gap_score(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # If we found a good cell, refine around it
    if best_cell is not None:
        # Fine-tune the radius
        for dr in [-0.02, -0.01, 0.01, 0.02]:
            if time.time() - start_time > time_limit - 5:
                break
            radius = 0.2 + dr
            cell = compute_cell(radius, 0.0, 0.0)
            score = band_gap_score(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Return the best cell found
    if best_cell is None:
        # Fallback to seed design
        best_cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            best_cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    
    return best_cell
# EVOLVE-BLOCK-END
