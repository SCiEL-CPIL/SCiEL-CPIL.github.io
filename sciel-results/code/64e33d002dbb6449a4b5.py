"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _pw_solver(cell, k_points, n_pw=12):
    """
    Solve the TM band structure for a given pixel cell using a plane-wave method.
    Returns the gap-midgap ratio for bands 4 and 5.
    """
    size = len(cell)
    eps = np.array([[12.0 if c == 1 else 1.0 for c in row] for row in cell], dtype=float)
    mu = 1.0 / eps
    
    # Compute Fourier coefficients of mu
    # We need mu(G) for G in the set of plane waves used.
    # The plane waves are indexed by integers (nx, ny) in [-n_pw, n_pw].
    # mu(nx, ny) = (1/size^2) * sum_{i,j} mu(i,j) * exp(-2*pi*i*(nx*i + ny*j)/size)
    # This is just the 2D FFT of mu, shifted appropriately.
    
    mu_fft = np.fft.fft2(mu)
    # FFT gives frequencies 0..size-1. We need -n_pw..n_pw.
    # Since size=32 and n_pw=12, we need indices -12..12.
    # In FFT output, index k corresponds to frequency k if k < size/2, else k-size.
    # So index 12 is freq 12, index 20 is freq -12 (since 20-32 = -12).
    
    # Let's construct the matrix of mu(G-G') for the plane wave set.
    # Plane waves: G_k = (nx, ny) where nx, ny in [-n_pw, n_pw].
    # Total number of plane waves M = (2*n_pw + 1)^2.
    
    n = 2 * n_pw + 1
    G_indices = np.arange(-n_pw, n_pw + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    M = len(G_vectors)
    
    # We need the sub-matrix of mu_fft corresponding to differences of G vectors.
    # mu(G_i - G_j)
    
    # Precompute indices for mu_fft
    # mu_fft is size x size.
    # Frequency f is at index f if f >= 0, else f + size.
    # We need to extract mu_fft[f_x, f_y] for f_x, f_y in [-2*n_pw, 2*n_pw].
    
    # Let's create a lookup table for mu_fft values for all needed differences.
    # The differences d_x = Gx_i - Gx_j range from -2*n_pw to 2*n_pw.
    
    # Efficient way:
    # Create a 2D array of differences for all pairs? No, M is (2*12+1)^2 = 25^2 = 625.
    # 625x625 matrix. That's 390k entries.
    
    # Let's precompute the indices into mu_fft for each unique difference.
    # Actually, we can just loop or vectorize the extraction.
    
    # Let's build the matrix H directly.
    # H_{ij} = mu(G_i - G_j) * (k + G_j)^2
    
    # First, get mu(G_i - G_j) for all i, j.
    # diff_x = Gx_i - Gx_j, diff_y = Gy_i - Gy_j.
    
    # To avoid MxM memory blowup if M is large, but M=625 is fine.
    
    Gx_flat = Gx.ravel()
    Gy_flat = Gy.ravel()
    
    # Vectorized difference calculation
    # This creates M x M arrays.
    # diff_x[i, j] = Gx_flat[i] - Gx_flat[j]
    
    # To save memory, we can compute row by row or use broadcasting carefully.
    # But 625x625 is small.
    
    # Let's compute the indices into mu_fft.
    # idx_x = diff_x % size
    # idx_y = diff_y % size
    
    # We can precompute a lookup for mu_fft values for all possible differences in [-2*n_pw, 2*n_pw].
    # max_diff = 2 * n_pw
    # lookup = np.zeros((2*max_diff+1, 2*max_diff+1))
    
    # Let's just compute it directly for each k-point.
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    # Precompute G_flat squared? No, (k+G)^2 depends on k.
    
    for k in k_points:
        kx, ky = k
        # k is in units of 2*pi/a? No, standard BZ is -pi..pi.
        # Let's assume k is in radians/length, and a=1.
        # G = 2*pi * (nx, ny).
        
        # (k + G_j)^2 = (kx + 2*pi*Gx_j)^2 + (ky + 2*pi*Gy_j)^2
        # Wait, the eigenvalue equation is:
        # sum_j mu(G_i - G_j) (k + G_j)^2 c_j = omega^2 c_i
        # where mu is the Fourier coefficient of 1/epsilon.
        
        # Let's verify the scaling.
        # Usually, the equation is:
        # sum_j \frac{1}{\epsilon(G_i - G_j)} (k + G_j)^2 c_j = \omega^2 c_i
        # where \epsilon(G) are Fourier coefficients of epsilon.
        # Or is it 1/epsilon?
        # For TM polarization, the equation involves 1/epsilon.
        # So mu = 1/epsilon is correct.
        
        # However, the Fourier coefficients of a periodic function f(x) are:
        # f(G) = \int f(x) e^{-i G x} dx
        # For a discrete pixel map, f(G) = (1/A) \sum_{i,j} f(i,j) e^{-i G r_{ij}}
        # This is exactly what np.fft.fft2 gives (scaled by 1/N^2 if we divide by N^2).
        
        # Let's compute the matrix A.
        # A_{ij} = mu(G_i - G_j) * (k + G_j)^2
        
        # We need mu(G_i - G_j) for all i, j.
        # Let's precompute the indices for mu_fft.
        
        # To speed up, let's precompute the difference indices once outside the k loop?
        # The differences G_i - G_j are constant.
        
        # Let's do that.
        
        # diff_x[i, j] = Gx_flat[i] - Gx_flat[j]
        # We can compute this once.
        
        # But we are inside the function. Let's assume we can cache it or compute it efficiently.
        
        # For now, compute inside.
        
        # Optimization: Precompute diff indices for the specific G set.
        
        # Let's create a 2D array of indices into mu_fft.
        # idx_x[i, j] = (Gx_flat[i] - Gx_flat[j]) % size
        # idx_y[i, j] = (Gy_flat[i] - Gy_flat[j]) % size
        
        # This is M x M.
        
        # Let's try to vectorize the mu extraction.
        
        # Create meshgrid of Gx_flat and Gy_flat for differences?
        # No, Gx_flat is 1D.
        
        # diff_x = Gx_flat[:, None] - Gx_flat[None, :]
        # diff_y = Gy_flat[:, None] - Gy_flat[None, :]
        
        # idx_x = diff_x % size
        # idx_y = diff_y % size
        
        # mu_vals = mu_fft[idx_x, idx_y] / (size * size)
        
        # This creates M x M arrays. 625 x 625 = 390k. This is fine.
        
        # Let's do it.
        
        diff_x = Gx_flat[:, None] - Gx_flat[None, :]
        diff_y = Gy_flat[:, None] - Gy_flat[None, :]
        
        idx_x = diff_x % size
        idx_y = diff_y % size
        
        # Extract mu values
        mu_vals = mu_fft[idx_x, idx_y] / (size * size)
        
        # Compute (k + G_j)^2
        # G_j = 2*pi * (Gx_j, Gy_j)
        # k + G_j = (kx + 2*pi*Gx_j, ky + 2*pi*Gy_j)
        
        kg_x = kx + 2.0 * np.pi * Gx_flat
        kg_y = ky + 2.0 * np.pi * Gy_flat
        kg_sq = kg_x**2 + kg_y**2
        
        # A = mu_vals * kg_sq[None, :]
        # Note: A_{ij} = mu(G_i - G_j) * (k + G_j)^2
        # So we multiply each column j by kg_sq[j].
        
        A = mu_vals * kg_sq[None, :]
        
        # Solve eigenvalue problem
        # We only need the smallest 5 eigenvalues.
        # np.linalg.eigvalsh computes all.
        # For M=625, this is fast enough.
        
        evals = np.linalg.eigvalsh(A)
        
        # The eigenvalues are omega^2.
        # We want bands 4 and 5 (indices 3 and 4).
        # Note: The lowest eigenvalue corresponds to the lowest band.
        
        w4 = np.sqrt(evals[3])
        w5 = np.sqrt(evals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def _generate_k_points(num_points=30):
    """Generate 30 k-points in the irreducible BZ of a square lattice."""
    points = []
    # Gamma-X: kx in [0, pi], ky = 0
    # X-M: kx = pi, ky in [0, pi]
    # M-Gamma: kx = ky, kx in [0, pi]
    
    # Let's sample 10 points on each segment.
    # Total 30 points.
    
    # Gamma (0,0) to X (pi, 0)
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
        
    # X (pi, 0) to M (pi, pi)
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
        
    # M (pi, pi) to Gamma (0, 0) along diagonal
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
        
    return np.array(points)

def _symmetrize(cell):
    """
    Enforce square symmetry: 
    cell == cell[::-1] row-wise and column-wise and cell == its transpose.
    This implies 4-fold symmetry.
    The top-left 8x8 quadrant defines the whole cell?
    No, 32x32. Half is 16.
    Wait, 32x32. Center is between 15 and 16.
    Symmetry about the cell centre.
    So (i, j) == (31-i, 31-j) (rotation 180)
    (i, j) == (i, 31-j) (reflection x)
    (i, j) == (31-i, j) (reflection y)
    (i, j) == (j, i) (transpose)
    
    This means the top-left 16x16 quadrant (indices 0..15) defines the rest?
    Let's check.
    If we define q[i][j] for i, j in 0..15.
    Then:
    cell[i][j] = q[i][j]
    cell[i][31-j] = q[i][j]
    cell[31-i][j] = q[i][j]
    cell[31-i][31-j] = q[i][j]
    
    And transpose symmetry:
    cell[i][j] == cell[j][i]
    So q[i][j] must equal q[j][i] if both i, j < 16?
    If i < 16 and j < 16, then cell[i][j] = q[i][j] and cell[j][i] = q[j][i].
    So yes, q must be symmetric.
    
    What if i < 16 and j >= 16?
    Then j' = 31-j < 16.
    cell[i][j] = cell[i][31-j'] = q[i][j'].
    cell[j][i] = cell[31-j'][i] = q[31-j'][i]? No.
    cell[j][i] where j = 31-j'.
    So cell[31-j'][i].
    Since 31-j' >= 16, we use the reflection rule.
    cell[31-j'][i] = cell[j'][i] (by vertical reflection? No, 31-j' is the row index).
    Let's use the definition:
    cell[r][c] is determined by the "canonical" position (i, j) where i, j in 0..15.
    i = min(r, 31-r)
    j = min(c, 31-c)
    
    If we define q[i][j] for i, j in 0..15, then:
    cell[r][c] = q[min(r, 31-r)][min(c, 31-c)]
    
    Does this satisfy transpose?
    cell[r][c] = q[i_r][j_c] where i_r = min(r, 31-r), j_c = min(c, 31-c).
    cell[c][r] = q[i_c][j_r] where i_c = min(c, 31-c) = j_c, j_r = min(r, 31-r) = i_r.
    So cell[c][r] = q[j_c][i_r].
    For cell[r][c] == cell[c][r], we need q[i_r][j_c] == q[j_c][i_r].
    So q must be symmetric.
    
    So the algorithm is:
    1. Take the top-left 16x16 quadrant.
    2. Enforce symmetry on this quadrant (q[i][j] = q[j][i]).
    3. Fill the 32x32 cell using the reflection rule.
    """
    size = len(cell)
    half = size // 2
    q = [row[:half] for row in cell[:half]]
    
    # Enforce symmetry on q
    for i in range(half):
        for j in range(i, half):
            val = q[i][j]
            q[i][j] = val
            q[j][i] = val
            
    new_cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            ri = i if i < half else size - 1 - i
            rj = j if j < half else size - 1 - j
            new_cell[i][j] = q[ri][rj]
            
    return new_cell

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    lower_band = problem["lower_band"]
    
    # Start with a good initial guess: 4 circles
    # Or use the seed program's single circle?
    # Inspiration 1 (4 circles) scored 5.78.
    # Let's start with 4 circles and optimize.
    
    # Initial cell: 4 circles of radius 0.15 at (+/-0.25, +/-0.25)
    cell = [[0]*size for _ in range(size)]
    r = 0.15
    centers = [
        (0.25, 0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (-0.25, -0.25)
    ]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for cx, cy in centers:
                if (x-cx)**2 + (y-cy)**2 < r*r:
                    cell[i][j] = 1
                    break
    
    cell = _symmetrize(cell)
    
    k_points = _generate_k_points(30)
    
    # Evaluate initial score
    current_score = _pw_solver(cell, k_points)
    best_cell = [row[:] for row in cell]
    best_score = current_score
    
    # Optimization loop
    # Use local search: flip pixels in the 16x16 quadrant.
    # To respect symmetry, we only flip q[i][j] and q[j][i] together if i!=j.
    # Actually, _symmetrize handles the full cell.
    # We can just flip a pixel in the full cell, then symmetrize.
    # But flipping one pixel in the full cell breaks symmetry, then symmetrize fixes it.
    # This effectively flips the orbit of that pixel.
    
    # Let's use a greedy hill-climbing with random restarts or simulated annealing?
    # Given time budget, simple hill climbing with randomization.
    
    start_time = time.time()
    deadline = start_time + 180  # Leave 20s margin
    
    # List of pixels to try flipping.
    # We can iterate through all pixels in the 16x16 quadrant.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle the order of pixels to avoid bias?
        # Or just iterate.
        
        # Let's try all pixels in the 16x16 quadrant.
        # To save time, we can evaluate only if the flip changes the cell.
        
        for i in range(16):
            for j in range(16):
                # Flip pixel (i, j) in the quadrant
                # This affects the whole orbit in the 32x32 cell.
                
                # Create a copy
                new_cell = [row[:] for row in cell]
                new_cell[i][j] = 1 - new_cell[i][j]
                
                # Symmetrize
                new_cell = _symmetrize(new_cell)
                
                # Check if it changed
                # If not, skip
                # We can check if the orbit was uniform before.
                # Or just evaluate. Evaluation is the bottleneck.
                
                # Let's check if the pixel's orbit is uniform.
                # If all pixels in the orbit are the same, flipping changes the cell.
                # If not, it might still change the cell (making it more uniform or less).
                # Actually, symmetrize takes the value from the quadrant.
                # So if we flip q[i][j], the whole orbit becomes 1-q[i][j].
                # So the cell always changes unless the orbit was already uniform and we flipped to the same value? No, we flip 0->1 or 1->0.
                # So the cell always changes.
                
                # Evaluate
                new_score = _pw_solver(new_cell, k_points)
                
                if new_score > best_score:
                    best_score = new_score
                    best_cell = new_cell
                    cell = new_cell
                    improved = True
                elif new_score > current_score:
                    # Accept if better than current, even if not best?
                    # Standard hill climbing accepts only if better than current.
                    # But we want to escape local minima.
                    # Let's use a simple strategy: accept if better than current.
                    # If we get stuck, we can add random jumps.
                    
                    # For now, let's stick to strict hill climbing.
                    # If no improvement in a full pass, we might be stuck.
                    pass
                
                # Check time
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
        
    return best_cell

# EVOLVE-BLOCK-END
