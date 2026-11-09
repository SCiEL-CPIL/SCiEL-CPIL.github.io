import numpy as np
import time

# Precompute G vectors
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
def generate_k_points(num_points=30):
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
        
    # M-Gamma: kx = ky
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
        
    return np.array(points)

K_POINTS = generate_k_points()
KG_SQ = []
for k in K_POINTS:
    kx, ky = k
    kg = k[None, :] + G_vectors * (2 * np.pi)
    kg_sq = np.sum(kg**2, axis=1)
    KG_SQ.append(kg_sq)
KG_SQ = np.array(KG_SQ)

# Precompute inverse sqrt of KG_SQ for each k-point
KG_INV_SQRT = [1.0 / np.sqrt(kg_sq) for kg_sq in KG_SQ]

def evaluate_cell(cell, verbose=False):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    cell: list of lists of 0/1. 1 = dielectric (eps=12), 0 = air (eps=1).
    """
    size = len(cell)
    eps = np.array([[12.0 if c == 1 else 1.0 for c in row] for row in cell], dtype=float)
    mu = 1.0 / eps
    
    # Compute Fourier coefficients of mu
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for ki in range(len(K_POINTS)):
        B_inv_sqrt = KG_INV_SQRT[ki]
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        S = (S + S.T) / 2.0
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

def make_cell_from_quadrant(q):
    """Create a full 32x32 cell from an 8x8 symmetric quadrant."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            val = q[i][j]
            cell[i][j] = val
            cell[i][size-1-j] = val
            cell[size-1-i][j] = val
            cell[size-1-i][size-1-j] = val
    return cell

def flip_pixel(cell, i, j):
    """Flip pixel at (i,j) in the 8x8 quadrant, updating full cell."""
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

def perlin_noise_2d(size, seed, freq=4.0):
    """Generate 2D value noise with smooth interpolation."""
    rng = np.random.RandomState(seed)
    grid_size = max(2, int(freq))
    coarse = rng.rand(grid_size, grid_size)
    
    x_coords = np.linspace(0, grid_size - 1, size)
    y_coords = np.linspace(0, grid_size - 1, size)
    
    x0 = np.floor(x_coords).astype(int)
    x1 = np.minimum(x0 + 1, grid_size - 1)
    y0 = np.floor(y_coords).astype(int)
    y1 = np.minimum(y0 + 1, grid_size - 1)
    
    dx = x_coords - x0
    dy = y_coords - y0
    dx = dx * dx * (3 - 2 * dx)
    dy = dy * dy * (3 - 2 * dy)
    
    field = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            v00 = coarse[y0[i], x0[j]]
            v01 = coarse[y0[i], x1[j]]
            v10 = coarse[y1[i], x0[j]]
            v11 = coarse[y1[i], x1[j]]
            v0 = v00 * (1 - dx[j]) + v01 * dx[j]
            v1 = v10 * (1 - dx[j]) + v11 * dx[j]
            field[i, j] = v0 * (1 - dy[i]) + v1 * dy[i]
    return field

def generate_noise_cell(seed, freq, threshold):
    """Generate a C4v-symmetric cell from noise field."""
    size = 32
    field = perlin_noise_2d(size, seed, freq)
    
    sym_field = np.zeros((size, size))
    sym_field += field
    sym_field += np.rot90(field, 1)
    sym_field += np.rot90(field, 2)
    sym_field += np.rot90(field, 3)
    sym_field += field[:, ::-1]
    sym_field += field[::-1, :]
    sym_field += np.rot90(field[:, ::-1], 1)
    sym_field += np.rot90(field[::-1, :], 1)
    sym_field /= 8.0
    
    cell_arr = (sym_field > threshold).astype(int)
    
    # Extract top-left 8x8 and enforce symmetry
    q = [[cell_arr[i][j] for j in range(8)] for i in range(8)]
    for i in range(8):
        for j in range(i, 8):
            val = (q[i][j] + q[j][i]) // 2
            q[i][j] = val
            q[j][i] = val
    
    return make_cell_from_quadrant(q)

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 230s kill
    
    size = 32
    candidates = []
    
    # === INVERTED: Dielectric background with air holes ===
    # These are the key candidates based on measured improvement
    
    # Full dielectric background with small air hole at center
    cell_bg_hole = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.15**2:
                cell_bg_hole[i][j] = 0
    candidates.append(cell_bg_hole)
    
    # Dielectric background with larger air hole
    cell_bg_hole2 = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_bg_hole2[i][j] = 0
    candidates.append(cell_bg_hole2)
    
    # Dielectric background with air ring
    cell_bg_air_ring = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.1 < r < 0.3:
                cell_bg_air_ring[i][j] = 0
    candidates.append(cell_bg_air_ring)
    
    # Dielectric background with air cross (holes along axes)
    cell_bg_air_cross = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.08 or abs(y) < 0.08:
                cell_bg_air_cross[i][j] = 0
    candidates.append(cell_bg_air_cross)
    
    # Dielectric background with air diamond hole
    cell_bg_air_diamond = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < 0.2:
                cell_bg_air_diamond[i][j] = 0
    candidates.append(cell_bg_air_diamond)
    
    # Dielectric background with 4 air holes (merged-hole geometry)
    cell_bg_4holes = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for dx in [0, 0.25]:
                for dy in [0, 0.25]:
                    dx2 = min(abs(x - dx), 1 - abs(x - dx))
                    dy2 = min(abs(y - dy), 1 - abs(y - dy))
                    if dx2*dx2 + dy2*dy2 < 0.08**2:
                        cell_bg_4holes[i][j] = 0
    candidates.append(cell_bg_4holes)
    
    # Dielectric background with 4 larger merged air holes
    cell_bg_4holes_large = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for dx in [0, 0.25]:
                for dy in [0, 0.25]:
                    dx2 = min(abs(x - dx), 1 - abs(x - dx))
                    dy2 = min(abs(y - dy), 1 - abs(y - dy))
                    if dx2*dx2 + dy2*dy2 < 0.12**2:
                        cell_bg_4holes_large[i][j] = 0
    candidates.append(cell_bg_4holes_large)
    
    # Dielectric background with square air hole
    cell_bg_sq_hole = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if max(abs(x), abs(y)) < 0.15:
                cell_bg_sq_hole[i][j] = 0
    candidates.append(cell_bg_sq_hole)
    
    # Dielectric background with hexagonal-like air hole
    cell_bg_hex_hole = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if max(abs(x), abs(y), abs(x+y)/1.5, abs(x-y)/1.5) < 0.18:
                cell_bg_hex_hole[i][j] = 0
    candidates.append(cell_bg_hex_hole)
    
    # Original candidates (air background with dielectric features) for comparison
    cell_rod = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    cell_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_ring[i][j] = 1
    candidates.append(cell_ring)
    
    # Noise-based candidates (both orientations)
    for seed in [42, 123, 456, 789, 101]:
        for freq in [3, 5, 7]:
            for thresh in [0.4, 0.5, 0.6]:
                cell_n = generate_noise_cell(seed, freq, thresh)
                candidates.append(cell_n)
                # Also inverted
                cell_n_inv = [[1 - cell_n[i][j] for j in range(size)] for i in range(size)]
                candidates.append(cell_n_inv)
    
    # Evaluate all candidates
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
    
    # Simulated Annealing search
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    T = 0.15
    T_min = 0.005
    cooling = 0.997
    
    while time.time() < deadline and T > T_min:
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)
        
        new_cell = [row[:] for row in current_cell]
        flip_pixel(new_cell, i, j)
        
        new_score = evaluate_cell(new_cell)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
        
        T *= cooling
        
        # Multi-flip move
        if np.random.random() < 0.15:
            n_flips = np.random.randint(2, 6)
            new_cell2 = [row[:] for row in current_cell]
            for _ in range(n_flips):
                ii = np.random.randint(0, 8)
                jj = np.random.randint(0, 8)
                flip_pixel(new_cell2, ii, jj)
            new_score2 = evaluate_cell(new_cell2)
            delta2 = new_score2 - current_score
            if delta2 > 0 or np.random.random() < np.exp(delta2 / T):
                current_cell = new_cell2
                current_score = new_score2
                if current_score > best_score:
                    best_score = current_score
                    best_cell = [row[:] for row in current_cell]
        
        T *= cooling
    
    return best_cell
