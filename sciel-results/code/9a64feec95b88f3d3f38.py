import numpy as np
import time

# Precompute G vectors and difference indices
N_PW = 8
G_range = np.arange(-N_PW, N_PW + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
Gx_flat = G_vectors[:, 0]
Gy_flat = G_vectors[:, 1]

# Precompute difference indices for mu lookup
# mu_coeffs will be indexed by (diff_x + 16) % 32, (diff_y + 16) % 32
Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
idx_x = (Diff_x + 16) % 32
idx_y = (Diff_y + 16) % 32

def generate_k_points():
    """Generate 30 k-points in the irreducible BZ of a square lattice."""
    points = []
    # Gamma-X: kx in [0, pi], ky = 0
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
        
    # X-M: kx = pi, ky in [0, pi]
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
        
    # M-Gamma: kx = ky, k in [0, pi]
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
        
    return np.array(points)

K_POINTS = generate_k_points()
KX = K_POINTS[:, 0]
KY = K_POINTS[:, 1]

def evaluate_cell_fast(cell):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    cell: numpy array 32x32 of 0/1.
    """
    size = 32
    # 1 = dielectric (eps=12), 0 = air (eps=1)
    eps = np.where(cell == 1, 12.0, 1.0)
    mu = 1.0 / eps
    
    # Compute Fourier coefficients of mu
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft) / (size * size)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for kx, ky in K_POINTS:
        # Precompute (k + G)^2 for all G
        kg = np.column_stack((kx + 2 * np.pi * Gx_flat, ky + 2 * np.pi * Gy_flat))
        kg_sq = np.sum(kg**2, axis=1)
        
        # Construct mu_diff matrix
        mu_diff = mu_fft_shifted[idx_x, idx_y]
        
        # Symmetrize: S = B^{-1/2} A B^{-1/2}
        B_inv_sqrt = 1.0 / np.sqrt(kg_sq)
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
        # Solve for 5 smallest eigenvalues
        evals = np.linalg.eigvalsh(S)
        
        w4 = np.sqrt(evals[3])
        w5 = np.sqrt(evals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def flip_pixel(cell, i, j):
    """
    Flip pixel at i, j in the 8x8 quadrant, and update the full cell.
    i, j in 0..7.
    """
    size = 32
    val = cell[i][j]
    new_val = 1 - val
    
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

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    size = 32
    candidates = []
    
    # 1. Small Rod (radius 0.2)
    cell_rod = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    # 2. Medium Rod (radius 0.3)
    cell_rod2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.3**2:
                cell_rod2[i][j] = 1
    candidates.append(cell_rod2)
    
    # 3. Checkerboard
    cell_cb = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell_cb[i][j] = 1
    candidates.append(cell_cb)
    
    # 4. Inverted Checkerboard
    cell_icb = 1 - cell_cb
    candidates.append(cell_icb)
    
    # 5. Cross (bars)
    cell_cross = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5:
                cell_cross[i][j] = 1
    candidates.append(cell_cross)
    
    # 6. Ring (annulus)
    cell_ring = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_ring[i][j] = 1
    candidates.append(cell_ring)
    
    # 7. Large Cross
    cell_cross2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 4.5 or abs(j - 15.5) < 4.5:
                cell_cross2[i][j] = 1
    candidates.append(cell_cross2)
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell_fast(c)
        if s > best_score:
            best_score = s
            best_cell = c.copy()
            
    # Local Search with Simulated Annealing
    current_cell = best_cell.copy()
    current_score = best_score
    
    # Convert to list of lists for flip_pixel? 
    # flip_pixel expects list of lists. 
    # Let's make a helper to flip on numpy array directly for speed.
    
    def flip_pixel_np(cell_np, i, j):
        size = 32
        val = cell_np[i][j]
        new_val = 1 - val
        idxs = [
            (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
            (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
        ]
        seen = set()
        for r, c in idxs:
            if (r, c) not in seen:
                seen.add((r, c))
                cell_np[r][c] = new_val
        return cell_np

    T = 0.05
    T_min = 0.001
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # Pick random quadrant pixel
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)
        
        new_cell = current_cell.copy()
        flip_pixel_np(new_cell, i, j)
        
        new_score = evaluate_cell_fast(new_cell)
        
        # Simulated Annealing acceptance
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
        else:
            delta = new_score - current_score
            if T > T_min:
                if np.random.random() < np.exp(delta / T):
                    current_cell = new_cell
                    current_score = new_score
            T *= cooling_rate
            
    return [list(row) for row in best_cell]
