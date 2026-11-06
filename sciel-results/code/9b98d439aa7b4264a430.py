"""Design a 32x32 photonic crystal unit cell to maximize TM band 4-5 gap."""
import numpy as np
import time
import math

def design(problem):
    """
    Design a symmetric 32x32 unit cell for a 2D square-lattice photonic crystal
    with epsilon=12 dielectric in air, maximizing the gap between TM bands 4 and 5.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard
    t_start = time.time()
    DEADLINE = 210.0  # safety margin before 230s kill
    
    # Precompute pixel coordinates
    coords = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
    XX, YY = np.meshgrid(coords, coords, indexing='ij')
    
    def make_cell(shape_params):
        """Create a 32x32 cell from shape parameters.
        shape_params: list of (cx, cy, a, b, n) for superellipses, or special format
        Returns a 32x32 numpy array of 0/1.
        Must be symmetric under square group: cell == cell[::-1] rows, cols, and transpose.
        For a 32x32 grid, symmetry about center means:
        - 180-degree rotational symmetry: cell[i][j] == cell[31-i][31-j]
        - Mirror symmetry about x-axis: cell[i][j] == cell[31-i][j]
        - Mirror symmetry about y-axis: cell[i][j] == cell[i][31-j]
        - Transpose symmetry: cell[i][j] == cell[j][i]
        
        Actually, the problem says: cell == cell[::-1] row-wise and column-wise AND cell == its transpose.
        cell[::-1] row-wise means reversing the list of rows: cell[::-1]
        cell[::-1] column-wise means each row reversed: [row[::-1] for row in cell]
        cell == its transpose.
        
        So we need:
        1. cell[i][j] == cell[31-i][j]  (vertical mirror)
        2. cell[i][j] == cell[i][31-j]  (horizontal mirror)  
        3. cell[i][j] == cell[j][i]     (transpose symmetry)
        
        These together imply full D4 symmetry.
        """
        cell = np.zeros((size, size), dtype=int)
        for param in shape_params:
            if len(param) == 5:
                cx, cy, a, b, n = param
                # Superellipse: |x-cx/a|^n + |y-cy/b|^n <= 1
                val = np.abs((XX - cx) / a) ** n + np.abs((YY - cy) / b) ** n
                cell[val <= 1.0] = 1
            elif len(param) == 4:
                # Circle: cx, cy, r, n=2
                cx, cy, r, n = param
                val = (XX - cx) ** 2 + (YY - cy) ** 2
                cell[val <= r ** 2] = 1
        return cell
    
    def enforce_symmetry(cell):
        """Enforce D4 symmetry by averaging over the symmetry group."""
        # The symmetry operations for a 32x32 grid (indices 0..31):
        # Identity: (i,j)
        # 90-degree rotation: (i,j) -> (j, 31-i)
        # 180-degree rotation: (i,j) -> (31-i, 31-j)
        # 270-degree rotation: (i,j) -> (31-j, i)
        # Mirror x: (i,j) -> (i, 31-j)
        # Mirror y: (i,j) -> (31-i, j)
        # Anti-diagonal mirror: (i,j) -> (31-j, 31-i)
        # Main diagonal mirror: (i,j) -> (j, i)
        
        # Actually, let me reconsider. The constraints are:
        # 1. cell == cell[::-1] (reverse rows) -> cell[i][j] == cell[31-i][j]
        # 2. cell == [row[::-1] for row in cell] (reverse each row) -> cell[i][j] == cell[i][31-j]
        # 3. cell == cell.T (transpose) -> cell[i][j] == cell[j][i]
        
        # From 1 and 2: cell[i][j] == cell[31-i][j] and cell[i][j] == cell[i][31-j]
        # From 1 and 3: cell[i][j] == cell[31-i][j] and cell[i][j] == cell[j][i]
        # From 2 and 3: cell[i][j] == cell[i][31-j] and cell[i][j] == cell[j][i]
        
        # Let's just enforce by taking the average over all 8 symmetry operations
        # and thresholding at 0.5.
        
        sym_cells = []
        # Identity
        sym_cells.append(cell)
        # Reverse rows
        sym_cells.append(cell[::-1, :])
        # Reverse columns
        sym_cells.append(cell[:, ::-1])
        # Reverse both
        sym_cells.append(cell[::-1, ::-1])
        # Transpose
        sym_cells.append(cell.T)
        # Transpose + reverse rows
        sym_cells.append(cell.T[::-1, :])
        # Transpose + reverse cols
        sym_cells.append(cell.T[:, ::-1])
        # Transpose + reverse both
        sym_cells.append(cell.T[::-1, ::-1])
        
        avg = np.mean(sym_cells, axis=0)
        return (avg >= 0.5).astype(int)
    
    def make_symmetric_cell(shape_params):
        """Create a symmetric cell from shape parameters."""
        cell = make_cell(shape_params)
        return enforce_symmetry(cell)
    
    # Surrogate gap calculation using a simplified plane-wave method
    def calculate_surrogate_gap(cell, n_pw=17):
        """
        Calculate a surrogate band gap using a simplified plane-wave expansion.
        This is an approximation to speed up the search.
        n_pw: number of plane waves in each direction (should be odd for symmetry)
        """
        # Fourier coefficients of epsilon
        # epsilon(x,y) = eps_low + (eps_high - eps_low) * cell(x,y)
        # We need the Fourier coefficients of the periodic function.
        
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        
        # Compute Fourier coefficients using FFT
        # For a 2D periodic function on a 32x32 grid, the Fourier coefficients are:
        # eps_G = (1/N^2) * sum_{i,j} eps[i,j] * exp(-2*pi*i*(G_x*i + G_y*j)/N)
        
        # We'll use a subset of G-vectors for the surrogate
        # G-vectors: G = (Gx, Gy) where Gx, Gy are integers
        # For TM mode, we need: (G - k)^T * eps_{G-G'} * (G' - k)
        
        # Simplified: use a small number of G-vectors
        # Let's use G-vectors with |Gx|, |Gy| <= n_pw//2
        
        N = size
        half = n_pw // 2
        
        # List of G-vectors
        G_list = []
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        
        # Compute Fourier coefficients
        eps_F = {}
        for gx, gy in G_list:
            # eps_F[(gx,gy)] = (1/N^2) * sum_{i,j} eps[i,j] * exp(-2*pi*i*(gx*i + gy*j)/N)
            s = 0.0
            for i in range(N):
                for j in range(N):
                    s += eps[i, j] * np.exp(-2j * np.pi * (gx * i + gy * j) / N)
            eps_F[(gx, gy)] = s / (N * N)
        
        # For TM mode, the eigenvalue equation is:
        # sum_{G'} (G - k)^T * eps_{G-G'} * (G' - k) * c_{G'} = omega^2 * c_G
        # where eps_{G} = 1/epsilon at G (for TM)
        
        # Actually, for TM polarization in 2D, the equation is:
        # sum_{G'} (G - k) · (G' - k) * (1/epsilon)_{G-G'} * c_{G'} = omega^2 * c_G
        
        # The inverse epsilon Fourier coefficients:
        # 1/epsilon(x) = sum_G (1/epsilon)_G * exp(i*G*x)
        # (1/epsilon)_G = (1/N^2) * sum_{i,j} (1/eps[i,j]) * exp(-2*pi*i*(Gx*i+Gy*j)/N)
        
        inv_eps_F = {}
        for gx, gy in G_list:
            s = 0.0
            for i in range(N):
                for j in range(N):
                    s += (1.0 / eps[i, j]) * np.exp(-2j * np.pi * (gx * i + gy * j) / N)
            inv_eps_F[(gx, gy)] = s / (N * N)
        
        # Now build the eigenvalue matrix for a given k-point
        # k is in units of 2*pi/a where a=1
        # The irreducible BZ for square lattice: 0 <= ky <= kx, kx >= 0
        # Key points: Gamma (0,0), X (pi, 0), M (pi, pi), and points along X-M
        
        # For the surrogate, let's evaluate at a few key k-points
        k_points = [
            (0.0, 0.0),      # Gamma
            (0.5, 0.0),      # X (in units where BZ edge is 0.5)
            (0.5, 0.5),      # M
            (0.25, 0.0),     # along Gamma-X
            (0.5, 0.25),     # along X-M
            (0.25, 0.25),    # along Gamma-M
        ]
        
        # For each k-point, solve the eigenvalue problem
        # The matrix is: M_{G,G'} = (G-k)·(G'-k) * inv_eps_F[G-G']
        
        # We need G-G' to be in our G_list for all G, G' in G_list
        # Let's use a smaller set to keep it fast
        
        # Actually, let's use a much simpler surrogate: just look at the
        # Fourier components at specific G-vectors that matter for the gap.
        
        # The gap between bands 4 and 5 in TM for a square lattice PC is
        # primarily controlled by the Fourier components at G = (1,0), (0,1), (1,1), etc.
        
        # Let's compute the key Fourier components of 1/epsilon
        key_Gs = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2), (2, 1), (1, 2)]
        
        # For the surrogate score, use the magnitude of these Fourier components
        # as a proxy for the gap strength
        score = 0.0
        for gx, gy in key_Gs:
            if (gx, gy) in inv_eps_F:
                score += abs(inv_eps_F[(gx, gy)])
        
        return score
    
    # A better surrogate: actually compute the band structure at a few k-points
    def calculate_band_structure(cell, k_points_list, n_pw=9):
        """
        Compute TM bands at given k-points using plane-wave method.
        Returns list of (k_point, sorted_eigenvalues)
        """
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps = 1.0 / eps
        
        N = size
        half = n_pw // 2
        
        # G-vectors
        G_list = []
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        
        nG = len(G_list)
        
        # Precompute inv_eps Fourier coefficients
        inv_eps_F = {}
        for gx, gy in G_list:
            s = 0.0
            for i in range(N):
                for j in range(N):
                    s += inv_eps[i, j] * np.exp(-2j * np.pi * (gx * i + gy * j) / N)
            inv_eps_F[(gx, gy)] = s / (N * N)
        
        results = []
        for kx, ky in k_points_list:
            # Build eigenvalue matrix
            # M_{G,G'} = (G-k)·(G'-k) * inv_eps_F[G-G']
            M = np.zeros((nG, nG), dtype=complex)
            
            for ig, (gx, gy) in enumerate(G_list):
                Gkx = gx - kx
                Gky = gy - ky
                for igp, (gxp, gyp) in enumerate(G_list):
                    Gpkx = gxp - kx
                    Gpky = gyp - ky
                    dGx = gx - gxp
                    dGy = gy - gyp
                    if (dGx, dGy) in inv_eps_F:
                        M[ig, igp] = (Gkx * Gpkx + Gky * Gpky) * inv_eps_F[(dGx, dGy)]
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals.real)
            results.append((kx, ky, eigvals))
        
        return results
    
    # Key k-points in the irreducible BZ (in units where BZ is [0, 0.5]x[0, 0.5])
    # Irreducible BZ: 0 <= ky <= kx <= 0.5
    k_points_list = [
        (0.0, 0.0),      # Gamma
        (0.1, 0.0),
        (0.2, 0.0),
        (0.3, 0.0),
        (0.4, 0.0),
        (0.5, 0.0),      # X
        (0.5, 0.1),
        (0.5, 0.2),
        (0.5, 0.3),
        (0.5, 0.4),
        (0.5, 0.5),      # M
        (0.3, 0.1),
        (0.4, 0.2),
        (0.3, 0.2),
        (0.2, 0.1),
        (0.4, 0.3),
    ]
    
    def calculate_gap(cell):
        """Calculate the band 4-5 gap ratio using the plane-wave method."""
        results = calculate_band_structure(cell, k_points_list, n_pw=7)
        
        # For each k-point, we have eigenvalues. We need to track bands 4 and 5.
        # The bands are ordered by frequency (omega = sqrt(eigenvalue)).
        # Band 4 is the 4th lowest, band 5 is the 5th lowest.
        
        # We need to find max(omega_4) and min(omega_5) over all k-points.
        # omega_4 = sqrt(eigval[3]) (0-indexed)
        # omega_5 = sqrt(eigval[4])
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky, eigvals in results:
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(max(0, eigvals[3]))
                omega_5 = np.sqrt(max(0, eigvals[4]))
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap_ratio
    
    # Initial candidate: centered circular rod
    # For a 32x32 grid, center is at (0,0) in our coordinate system
    # Radius in lattice units: 0.2 means radius = 0.2 * 32 pixels = 6.4 pixels
    # In our coordinate system, each pixel is 1/32 = 0.03125 units
    # So radius 0.2 lattice constants = 0.2 in our coordinate system
    
    # Let's try several initial designs
    candidates = []
    
    # 1. Circular rod, radius 0.2
    candidates.append([(0.0, 0.0, 0.2, 0.2, 2.0)])
    
    # 2. Circular rod, radius 0.15
    candidates.append([(0.0, 0.0, 0.15, 0.15, 2.0)])
    
    # 3. Circular rod, radius 0.25
    candidates.append([(0.0, 0.0, 0.25, 0.25, 2.0)])
    
    # 4. Square-ish rod (n=4 superellipse), half-width 0.15
    candidates.append([(0.0, 0.0, 0.15, 0.15, 4.0)])
    
    # 5. Square-ish rod (n=4 superellipse), half-width 0.2
    candidates.append([(0.0, 0.0, 0.2, 0.2, 4.0)])
    
    # 6. Diamond shape (n=1 superellipse), half-width 0.2
    candidates.append([(0.0, 0.0, 0.2, 0.2, 1.0)])
    
    # 7. Diamond shape (n=1 superellipse), half-width 0.25
    candidates.append([(0.0, 0.0, 0.25, 0.25, 1.0)])
    
    # 8. Elliptical rod, a=0.25, b=0.15
    candidates.append([(0.0, 0.0, 0.25, 0.15, 2.0)])
    
    # 9. Elliptical rod, a=0.15, b=0.25
    candidates.append([(0.0, 0.0, 0.15, 0.25, 2.0)])
    
    # 10. Larger circular rod, radius 0.3
    candidates.append([(0.0, 0.0, 0.3, 0.3, 2.0)])
    
    # 11. Smaller circular rod, radius 0.1
    candidates.append([(0.0, 0.0, 0.1, 0.1, 2.0)])
    
    # 12. Very square (n=8)
    candidates.append([(0.0, 0.0, 0.18, 0.18, 8.0)])
    
    # 13. n=6
    candidates.append([(0.0, 0.0, 0.18, 0.18, 6.0)])
    
    # 14. n=3
    candidates.append([(0.0, 0.0, 0.2, 0.2, 3.0)])
    
    # 15. n=1.5
    candidates.append([(0.0, 0.0, 0.22, 0.22, 1.5)])
    
    # 16. Cross shape: two overlapping rectangles
    # This is harder with superellipses, skip for now
    
    # Evaluate all candidates
    best_cell = None
    best_score = -float('inf')
    
    for cand in candidates:
        if time.time() - t_start > DEADLINE:
            break
        cell = make_symmetric_cell(cand)
        score = calculate_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # If we have time, do a local optimization around the best candidate
    if time.time() - t_start < DEADLINE - 30 and best_cell is not None:
        # Try perturbing the best candidate
        # Get the best candidate parameters
        # For simplicity, try a grid of radius and n values around the best
        
        # Find which candidate was best
        best_cand = None
        for cand in candidates:
            cell = make_symmetric_cell(cand)
            if np.array_equal(cell, best_cell):
                best_cand = cand
                break
        
        if best_cand is not None:
            cx, cy, a, b, n = best_cand[0]
            
            # Try nearby values
            for da in [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]:
                for dn in [-1.0, -0.5, 0.5, 1.0]:
                    if time.time() - t_start > DEADLINE - 5:
                        break
                    new_a = a + da
                    new_n = n + dn
                    if new_a > 0.05 and new_n > 0.5:
                        new_cand = [(cx, cy, new_a, b, new_n)]
                        cell = make_symmetric_cell(new_cand)
                        score = calculate_gap(cell)
                        if score > best_score:
                            best_score = score
                            best_cell = cell.copy()
    
    # Convert to nested list
    return best_cell.tolist()
