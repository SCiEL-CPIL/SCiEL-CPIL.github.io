import numpy as np
import time

# Precompute G vectors for 289 plane waves (17x17 grid, -8 to 8)
N_PW = 8
G_range = np.arange(-N_PW, N_PW + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
Gx_flat = G_vectors[:, 0]
Gy_flat = G_vectors[:, 1]
Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
idx_x = (Diff_x + 16) % 32
idx_y = (Diff_y + 16) % 32

# Precompute k-points and (k+G)^2 for each k-point
# The evaluator uses 30 wave vectors covering the irreducible BZ.
# We replicate a similar set for our local evaluation.
def generate_k_points(num_points=30):
    """Generate k-points in the irreducible BZ of a square lattice."""
    points = []
    # Gamma-X: kx in [0, pi], ky = 0
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
        
    # X-M: kx = pi, ky in [0, pi]
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
        
    # M-Gamma: kx = ky
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
        
    return np.array(points)

K_POINTS = generate_k_points()
KG_SQ = []
for k in K_POINTS:
    kx, ky = k
    # G vectors are integers, k is in radians (2*pi units).
    # (k + 2*pi*G)^2
    kg = k[None, :] + G_vectors * (2 * np.pi)
    kg_sq = np.sum(kg**2, axis=1)
    KG_SQ.append(kg_sq)
KG_SQ = np.array(KG_SQ)

# Precompute the diagonal matrices D_k = diag((k+G)^2) for each k
# We will use them to build the TM Hamiltonian: H = D_k @ F_mu
# Note: F_mu is the matrix of Fourier coefficients of mu = 1/eps.
# H_{ij} = (k+G_i)^2 * mu(G_i - G_j)

def evaluate_cell(cell, verbose=False):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    cell: list of lists of 0/1.
    1 = dielectric (eps=12), 0 = air (eps=1)
    """
    size = len(cell)
    eps_arr = np.array(cell, dtype=float)
    eps_arr[eps_arr == 1] = 12.0
    eps_arr[eps_arr == 0] = 1.0
    mu = 1.0 / eps_arr
    
    # Compute Fourier coefficients of mu
    # mu_fft = fft2(mu)
    # We need mu(G) for G in -8..8
    # Using the property that for a pixelated grid, the Fourier coefficient is
    # F[G] = (1/N^2) * sum_{i,j} mu[i,j] * exp(-i 2 pi (Gx*i/N + Gy*j/N))
    # This is exactly the DFT.
    mu_fft = np.fft.fft2(mu)
    # fft2 output is ordered 0..N-1. We need -8..8.
    # fftshift moves the zero frequency to the center.
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    # mu_diff[idx_x, idx_y] gives mu(G_i - G_j)
    mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for ki in range(len(K_POINTS)):
        # D_k is diagonal with (k+G)^2
        # H = D_k @ mu_diff
        # We can avoid explicit matrix multiplication by scaling rows
        # H[i, j] = KG_SQ[ki, i] * mu_diff[i, j]
        H = KG_SQ[ki][:, None] * mu_diff
        
        # H is Hermitian
        evals = np.linalg.eigvalsh(H)
        
        # The first few eigenvalues correspond to the lowest bands.
        # We need bands 4 and 5 (1-indexed), so indices 3 and 4 (0-indexed).
        # Ensure we have at least 5 positive eigenvalues
        if len(evals) < 5:
            continue
            
        # Filter out any negative eigenvalues due to numerical noise (should be positive)
        # omega^2 = evals
        w4 = np.sqrt(evals[3])
        w5 = np.sqrt(evals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    if min_omega_5 == float('inf'):
        return 0.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def flip_pixel(cell, i, j):
    """
    Flip pixel at i, j in the 8x8 quadrant, and update the full 32x32 cell.
    i, j in 0..7.
    The cell must be invariant under D4 symmetry.
    """
    size = 32
    val = cell[i][j]
    new_val = 1 - val
    
    # Positions in the full 32x32 cell corresponding to (i,j) in the top-left 8x8
    # and their symmetry equivalents.
    # (i, j)
    # (i, 31-j)
    # (31-i, j)
    # (31-i, 31-j)
    # And transposes:
    # (j, i)
    # (j, 31-i)
    # (31-j, i)
    # (31-j, 31-i)
    
    idxs = [
        (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
        (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
    ]
    seen = set()
    for r, c in idxs:
        if (r, c) not in seen:
            seen.add((r, c))
            cell[r][c] = new_val
            
    return cell

def quadrant_to_cell(q):
    """
    Convert an 8x8 symmetric quadrant to a 32x32 cell.
    q[i][j] defines the value for the orbit of (i,j).
    """
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            val = q[i][j]
            cell[i][j] = val
            cell[i][size-1-j] = val
            cell[size-1-i][j] = val
            cell[size-1-i][size-1-j] = val
            # Transpose symmetry is handled by the fact that we iterate i,j in 0..7
            # and the mapping covers the transpose.
            # However, if we just set these 4, do we cover the transpose?
            # If we set cell[i][j], we should also ensure cell[j][i] is consistent.
            # The mapping above sets:
            # (i,j), (i, 31-j), (31-i, j), (31-i, 31-j)
            # When we iterate to (j,i) in the loop, we will set:
            # (j,i), (j, 31-i), (31-j, i), (31-j, 31-i)
            # This covers all 8 points (or 4 if i=j, etc).
            # So this is correct.
    return cell

def cell_to_quadrant(cell):
    """
    Extract the 8x8 quadrant from a 32x32 cell.
    """
    return [row[:8] for row in cell[:8]]

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    size = 32
    candidates = []
    
    # Strategy: Dielectric background (1) with Air holes (0)
    # This is the inversion that showed high scores.
    
    # Candidate 1: Large central Air hole
    cell_hole = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.15**2:
                cell_hole[i][j] = 0
    candidates.append(cell_hole)
    
    # Candidate 2: Air hole with square shape
    cell_sq_hole = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.12 and abs(y) < 0.12:
                cell_sq_hole[i][j] = 0
    candidates.append(cell_sq_hole)
    
    # Candidate 3: Cross-shaped Air hole
    cell_cross_hole = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.10 or abs(y) < 0.10:
                cell_cross_hole[i][j] = 0
    candidates.append(cell_cross_hole)
    
    # Candidate 4: Ring of Air holes (Dielectric ring)
    cell_ring_air = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.10 < r < 0.25:
                cell_ring_air[i][j] = 0
    candidates.append(cell_ring_air)
    
    # Candidate 5: Small Air holes (checkerboard of holes)
    # This might be too dense, but let's try a sparse version
    cell_sparse_holes = [[1]*size for _ in range(size)]
    for i in range(0, size, 4):
        for j in range(0, size, 4):
            cell_sparse_holes[i][j] = 0
            cell_sparse_holes[i+1][j] = 0
            cell_sparse_holes[i][j+1] = 0
            cell_sparse_holes[i+1][j+1] = 0
    candidates.append(cell_sparse_holes)
    
    # Candidate 6: Dielectric background with small dielectric dots (Original strategy, but inverted context)
    # Actually, let's stick to Air holes in Dielectric.
    # Let's try a "Plus" shape of Air
    cell_plus_air = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.08 or abs(y) < 0.08:
                cell_plus_air[i][j] = 0
    candidates.append(cell_plus_air)
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
            
    # Local Search
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    stagnation = 0
    total_iters = 0
    
    while time.time() < deadline:
        total_iters += 1
        # Adaptive multi-flip
        if stagnation > 15 and np.random.random() < 0.5:
            # Multi-flip move
            n_flips = min(2 + stagnation // 10, 5)
            new_cell = [row[:] for row in current_cell]
            for _ in range(n_flips):
                i = np.random.randint(0, 8)
                j = np.random.randint(0, 8)
                flip_pixel(new_cell, i, j)
            new_score = evaluate_cell(new_cell)
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                best_cell = [row[:] for row in current_cell]
                best_score = current_score
                stagnation = 0
            else:
                stagnation += 1
        else:
            # Single flip
            i = np.random.randint(0, 8)
            j = np.random.randint(0, 8)
            
            new_cell = [row[:] for row in current_cell]
            flip_pixel(new_cell, i, j)
            
            new_score = evaluate_cell(new_cell)
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                best_cell = [row[:] for row in current_cell]
                best_score = current_score
                stagnation = 0
            else:
                stagnation += 1
                
    return best_cell
