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

def generate_k_points(num_points=30):
    """Generate 30 k-points in the irreducible BZ of a square lattice."""
    points = []
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
    return np.array(points)

K_POINTS = generate_k_points()
K_POINTS_LIST = [(k[0], k[1]) for k in K_POINTS]

def evaluate_cell(cell, verbose=False):
    """Evaluate the gap-midgap ratio for a 32x32 cell."""
    size = len(cell)
    eps = np.array([[12 if c == 1 else 1 for c in row] for row in cell], dtype=float)
    mu = 1.0 / eps
    
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for kx, ky in K_POINTS_LIST:
        kg = np.array([[kx + 2*np.pi*gx, ky + 2*np.pi*gy] for gx, gy in G_vectors])
        kg_sq = np.sum(kg**2, axis=1)
        
        B_inv_sqrt = 1.0 / np.sqrt(kg_sq)
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
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
    deadline = start_time + 115
    
    size = 32
    candidates = []
    
    # Rod
    cell_rod = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    # Checkerboard
    cell_cb = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell_cb[i][j] = 1
    candidates.append(cell_cb)
    
    # Inverted Checkerboard
    cell_icb = [[1 - cell_cb[i][j] for j in range(size)] for i in range(size)]
    candidates.append(cell_icb)
    
    # Cross
    cell_cross = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5:
                cell_cross[i][j] = 1
    candidates.append(cell_cross)
    
    # Ring
    cell_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_ring[i][j] = 1
    candidates.append(cell_ring)
    
    # Cross with hole
    cell_cross_hole = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            in_cross = (abs(x) < 0.12 or abs(y) < 0.12)
            in_hole = r < 0.08
            if in_cross and not in_hole:
                cell_cross_hole[i][j] = 1
    candidates.append(cell_cross_hole)
    
    # Square ring
    cell_sq_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            max_coord = max(abs(x), abs(y))
            if 0.15 < max_coord < 0.35:
                cell_sq_ring[i][j] = 1
    candidates.append(cell_sq_ring)
    
    # Thick ring
    cell_thick_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.1 < r < 0.4:
                cell_thick_ring[i][j] = 1
    candidates.append(cell_thick_ring)
    
    # Thin ring
    cell_thin_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.2 < r < 0.3:
                cell_thin_ring[i][j] = 1
    candidates.append(cell_thin_ring)
    
    # Dielectric square with air hole
    cell_sq_hole = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.4 and abs(y) < 0.4 and not (abs(x) < 0.15 and abs(y) < 0.15):
                cell_sq_hole[i][j] = 1
    candidates.append(cell_sq_hole)
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
    
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    stagnation = 0
    temperature = 1.0
    
    while time.time() < deadline:
        if stagnation > 30:
            temperature = max(0.1, temperature * 0.95)
        
        if stagnation > 50 and np.random.random() < 0.15:
            n_flips = np.random.randint(2, 6)
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
                temperature = 1.0
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(delta / temperature):
                    current_cell = new_cell
                    current_score = new_score
                stagnation += 1
        else:
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
                temperature = 1.0
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(delta / temperature):
                    current_cell = new_cell
                    current_score = new_score
                stagnation += 1
                
    return best_cell
