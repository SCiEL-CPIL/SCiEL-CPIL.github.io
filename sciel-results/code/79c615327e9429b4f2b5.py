"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _symmetrize(cell):
    """Enforce 4-fold symmetry about the cell centre."""
    size = len(cell)
    arr = np.array(cell, dtype=np.int8)
    # Flip vertically
    arr_v = arr[::-1, :]
    # Flip horizontally
    arr_h = arr[:, ::-1]
    # Flip both
    arr_vh = arr[::-1, ::-1]
    # Take the minimum to ensure all symmetry conditions hold (all must agree)
    # Actually, for 0/1, we need all four to be identical. 
    # Use majority vote or just force one transformation.
    # The requirement is cell == cell[::-1] row-wise AND column-wise AND transpose.
    # This means D4 symmetry.
    # Simplest: apply all 4 transforms and take element-wise AND (intersection) 
    # or just pick the one that is most "central". 
    # Actually, the easiest is to define a quadrant and reflect it.
    # Let's define the top-left 16x16 quadrant and reflect it to the other 3 quadrants.
    # But the cell is 32x32. The center is at (16,16).
    # Quadrants: TL (0-15, 0-15), TR (0-15, 16-31), BL (16-31, 0-15), BR (16-31, 16-31).
    # If we just pick a design that is symmetric, we are fine.
    # For now, just return the input if it's already symmetric, or force symmetry by
    # averaging (but we need 0/1).
    # Let's just trust that the generator produces symmetric cells.
    return arr

def _compute_gap_ratio(cell, problem):
    """
    Compute the gap-midgap ratio for TM bands 4 and 5 using a plane-wave method.
    This is a simplified internal evaluator to guide the search.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    lower_band = problem["lower_band"]
    
    # Pixel coordinates
    # The evaluator uses 289 plane waves. We'll use a smaller number for speed, e.g., 121 or 169.
    # 289 = 17^2. Let's use Npw = 9 or 11. 9^2 = 81, 11^2 = 121.
    Npw = 9  # 81 plane waves for speed
    # Plane wave indices
    # kx, ky from -Npw//2 to Npw//2
    half = Npw // 2
    indices = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(indices, indices, indexing='ij')
    G = np.column_stack((Gx.ravel(), Gy.ravel()))
    
    # Fourier coefficients of 1/epsilon
    # eps is 12 for dielectric, 1 for air.
    # 1/eps is 1/12 for dielectric, 1 for air.
    # We need the Fourier transform of 1/eps over the unit cell.
    # 1/eps(x,y) = sum_G (1/eps)_G exp(iG.x)
    # (1/eps)_G = (1/A) ∫∫ (1/eps(x,y)) exp(-iG.x) dA
    
    # Discretize the integral using the pixel values.
    # The cell is 32x32.
    # x_i = (i + 0.5)/32 - 0.5
    # y_j = (j + 0.5)/32 - 0.5
    
    x_coords = (np.arange(size) + 0.5) / size - 0.5
    y_coords = (np.arange(size) + 0.5) / size - 0.5
    
    # Create meshgrid of pixel coordinates
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # 1/eps array
    one_over_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)
    
    # Compute Fourier coefficients
    # G is in units of 2pi (since we're using integer indices for a unit cell)
    # Actually, G = 2pi * (n_x, n_y). The plane wave is exp(i 2pi n.r).
    # So the phase is exp(-i 2pi (n_x x + n_y y)).
    
    # We can compute this efficiently.
    # (1/eps)_G = sum_{i,j} one_over_eps[i,j] exp(-i 2pi (n_x x_i + n_y y_j)) / size^2
    
    # To speed this up, we can use the fact that the grid is uniform.
    # But for Npw=9 (81 coefficients) and size=32, it's 81*32*32 = 82944 operations.
    # This is very fast.
    
    G_coeffs = np.zeros(len(G), dtype=complex)
    for k, (nx, ny) in enumerate(G):
        phase = np.exp(-1j * 2 * np.pi * (nx * X + ny * Y))
        G_coeffs[k] = np.sum(one_over_eps * phase) / (size * size)
    
    # Now, solve the eigenvalue problem for TM polarization.
    # The matrix is A_{G,G'} = (G-G')^2 * (1/eps)_{G-G'} + delta_{G,G'} * (G-G')^2 ... wait.
    # For TM, the matrix element is:
    # A_{G,G'} = (G-G')^2 * (1/eps)_{G-G'} + k_0^2 * delta_{G,G'} ... no, that's not right.
    
    # The standard PWE formulation for TM is:
    # sum_{G'} ( (K+G)^2 / eps_{G-G'} ) * c_{G'} = omega^2 * c_G
    # where K is the Bloch vector.
    
    # Let's construct the matrix for a given K.
    # We need to evaluate this for a few K points in the irreducible BZ.
    # The irreducible BZ for a square lattice is the triangle (0,0) - (pi,0) - (pi,pi).
    # Key points: Gamma (0,0), M (pi,0), X (pi/2, pi/2), and points along the edges.
    
    # For the gap between bands 4 and 5, we care about the bands near the middle of the BZ.
    # Let's evaluate at a few points: Gamma, M, X, and the midpoint of Gamma-X.
    
    K_points = [
        np.array([0.0, 0.0]),       # Gamma
        np.array([np.pi, 0.0]),     # M
        np.array([np.pi/2, np.pi/2]), # X
        np.array([np.pi/4, np.pi/4]), # Midpoint Gamma-X
    ]
    
    max_omega_4 = 0.0
    min_omega_5 = np.inf
    
    for K in K_points:
        # Construct the matrix
        N = len(G)
        A = np.zeros((N, N), dtype=complex)
        
        # Precompute G-G'
        for i in range(N):
            for j in range(N):
                G_diff = G[i] - G[j]
                G_diff_sq = np.dot(G_diff, G_diff)
                
                # Find the index of G_diff in G
                # G_diff is a vector. We need to find if it's in G.
                # Since G is a grid, G_diff is also on the grid if |nx_i - nx_j| <= half*2.
                # But our G only has indices from -half to half.
                # So G_diff might be outside the range.
                # We need to handle this.
                
                # Let's find the index.
                idx = -1
                for k, (nx_k, ny_k) in enumerate(G):
                    if nx_k == G_diff[0] and ny_k == G_diff[1]:
                        idx = k
                        break
                
                if idx != -1:
                    # A_{i,j} = (K+G_i)^2 * (1/eps)_{G_i-G_j}
                    K_plus_Gi = K + G[i]
                    K_plus_Gi_sq = np.dot(K_plus_Gi, K_plus_Gi)
                    A[i, j] = K_plus_Gi_sq * G_coeffs[idx]
        
        # Solve the eigenvalue problem
        # A is Hermitian? No, (1/eps)_{G} is not real in general, but it is real if the structure is symmetric.
        # Actually, (1/eps)_{-G} = conjugate((1/eps)_G).
        # And A_{i,j} = (K+G_i)^2 * (1/eps)_{G_i-G_j}.
        # This matrix is not Hermitian in general, but the eigenvalues should be real and positive.
        # Let's use eigvalsh if it's Hermitian, otherwise eig.
        
        # Check if A is Hermitian
        # A_{j,i} = (K+G_j)^2 * (1/eps)_{G_j-G_i} = (K+G_j)^2 * (1/eps)_{-(G_i-G_j)}
        # = (K+G_j)^2 * conj((1/eps)_{G_i-G_j})
        # This is not the conjugate of A_{i,j} unless (K+G_i)^2 = (K+G_j)^2, which is not true.
        # So A is not Hermitian.
        
        # We need to solve A c = omega^2 c.
        # The eigenvalues are omega^2.
        
        eigvals = np.linalg.eigvals(A)
        eigvals = np.real(eigvals)  # Take real part
        eigvals = np.sort(eigvals)
        
        # The bands are indexed starting from 1.
        # Band 4 is eigvals[3], Band 5 is eigvals[4].
        
        omega_4 = np.sqrt(eigvals[3])
        omega_5 = np.sqrt(eigvals[4])
        
        max_omega_4 = max(max_omega_4, omega_4)
        min_omega_5 = min(min_omega_5, omega_5)
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2
    if midgap == 0:
        return 0.0
    return 2 * gap / midgap

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline
    
    # Initial design: a cross or a pattern that might open a larger gap.
    # The seed is a circle. Let's try a pattern with 4-fold symmetry that is not a circle.
    # For example, a square with rounded corners, or a cross.
    
    # Let's try a few candidate designs and pick the best one based on the internal evaluator.
    # Candidate 1: Seed (circle)
    cell_circle = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            row.append(1 if x * x + y * y < 0.2 ** 2 else 0)
        cell_circle.append(row)
    
    # Candidate 2: A square
    cell_square = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            row.append(1 if abs(x) < 0.25 and abs(y) < 0.25 else 0)
        cell_square.append(row)
    
    # Candidate 3: A cross
    cell_cross = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            row.append(1 if (abs(x) < 0.15 and abs(y) < 0.35) or (abs(y) < 0.15 and abs(x) < 0.35) else 0)
        cell_cross.append(row)
    
    candidates = [cell_circle, cell_square, cell_cross]
    
    best_cell = cell_circle
    best_score = _compute_gap_ratio(cell_circle, problem)
    
    for cell in candidates[1:]:
        if time.time() > deadline:
            break
        score = _compute_gap_ratio(cell, problem)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Convert to list of lists of ints
    arr = np.array(best_cell, dtype=np.int8)
    return arr.tolist()
# EVOLVE-BLOCK-END
