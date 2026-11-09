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
    cell: list of lists of 0/1.
    """
    size = len(cell)
    # 1 = dielectric (eps=12), 0 = air (eps=1)
    # mu = 1/eps
    eps = np.array([[12 if c == 1 else 1 for c in row] for row in cell], dtype=float)
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
        
        # Compute eigenvalues using NumPy
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

def quadrant_to_cell(q):
    """
    Convert an 8x8 symmetric quadrant to a full 32x32 cell.
    The quadrant must be symmetric: q[i][j] == q[j][i].
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
    return cell

def cell_to_quadrant(cell):
    """Extract the 8x8 quadrant from a 32x32 cell."""
    return [cell[i][:8] for i in range(8)]

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

def generate_rbf_cell(center_offsets, widths, threshold=0.5):
    """
    Generate a C4v-symmetric cell from radial basis functions.
    center_offsets: list of (dx, dy) offsets from center for one quadrant
    widths: list of Gaussian widths
    threshold: threshold for binarization
    """
    size = 32
    cell = [[0]*size for _ in range(size)]
    
    # Pixel centers in normalized coordinates
    xs = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
    ys = np.array([(j + 0.5) / size - 0.5 for j in range(size)])
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    # Build the RBF field
    field = np.zeros((size, size))
    for (dx, dy), w in zip(center_offsets, widths):
        # C4v symmetry: reflect across both axes and diagonal
        # Centers in all 8 symmetric positions
        centers = [
            (dx, dy), (dx, -dy), (-dx, dy), (-dx, -dy),
            (dy, dx), (dy, -dx), (-dy, dx), (-dy, -dx)
        ]
        for cx, cy in centers:
            dist_sq = (XX - cx)**2 + (YY - cy)**2
            field += np.exp(-dist_sq / (2 * w**2))
    
    # Binarize
    for i in range(size):
        for j in range(size):
            cell[i][j] = 1 if field[i][j] > threshold else 0
    
    return cell

def generate_4circle_cell(r, offset, threshold=0.5):
    """Generate a 4-circle cluster cell."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    
    xs = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
    ys = np.array([(j + 0.5) / size - 0.5 for j in range(size)])
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    # 4 circles at (±offset, ±offset)
    for sx in [1, -1]:
        for sy in [1, -1]:
            cx = sx * offset
            cy = sy * offset
            dist_sq = (XX - cx)**2 + (YY - cy)**2
            cell_arr = np.where(dist_sq < r**2, 1, 0)
            for i in range(size):
                for j in range(size):
                    if cell_arr[i][j] == 1:
                        cell[i][j] = 1
    
    return cell

def generate_initial_candidates():
    """Generate diverse initial candidates including RBF-based designs."""
    size = 32
    candidates = []
    
    # 4-circle clusters (key finding: this topology works well)
    for r in [0.08, 0.10, 0.12, 0.14, 0.16]:
        for offset in [0.15, 0.20, 0.25, 0.30]:
            if offset + r < 0.5:  # Keep within cell
                cell = generate_4circle_cell(r, offset)
                candidates.append(cell)
    
    # RBF-based cells with various parameters
    for n_centers in [1, 2, 3]:
        for _ in range(5):
            center_offsets = []
            widths = []
            for _ in range(n_centers):
                dx = np.random.uniform(0.05, 0.35)
                dy = np.random.uniform(0.05, 0.35)
                center_offsets.append((dx, dy))
                widths.append(np.random.uniform(0.05, 0.15))
            threshold = np.random.uniform(0.3, 0.7)
            cell = generate_rbf_cell(center_offsets, widths, threshold)
            candidates.append(cell)
    
    # Single circle
    for r in [0.1, 0.15, 0.2, 0.25]:
        cell = [[0]*size for _ in range(size)]
        xs = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
        ys = np.array([(j + 0.5) / size - 0.5 for j in range(size)])
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        dist_sq = XX**2 + YY**2
        for i in range(size):
            for j in range(size):
                if dist_sq[i][j] < r**2:
                    cell[i][j] = 1
        candidates.append(cell)
    
    # Ring
    for r_inner, r_outer in [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3), (0.25, 0.35)]:
        cell = [[0]*size for _ in range(size)]
        xs = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
        ys = np.array([(j + 0.5) / size - 0.5 for j in range(size)])
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        dist_sq = XX**2 + YY**2
        for i in range(size):
            for j in range(size):
                r_sq = dist_sq[i][j]
                if r_inner**2 < r_sq < r_outer**2:
                    cell[i][j] = 1
        candidates.append(cell)
    
    # Inverted ring (air in center)
    for r_inner, r_outer in [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3)]:
        cell = [[1]*size for _ in range(size)]
        xs = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
        ys = np.array([(j + 0.5) / size - 0.5 for j in range(size)])
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        dist_sq = XX**2 + YY**2
        for i in range(size):
            for j in range(size):
                r_sq = dist_sq[i][j]
                if r_sq < r_inner**2:
                    cell[i][j] = 0
        candidates.append(cell)
    
    # Checkerboard
    cell_cb = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell_cb[i][j] = 1
    candidates.append(cell_cb)
    
    # Inverted checkerboard
    cell_icb = [[1 - cell_cb[i][j] for j in range(size)] for i in range(size)]
    candidates.append(cell_icb)
    
    # Cross
    cell_cross = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5:
                cell_cross[i][j] = 1
    candidates.append(cell_cross)
    
    # Random cells
    for _ in range(15):
        q = [[0]*8 for _ in range(8)]
        for i in range(8):
            for j in range(i, 8):
                val = 1 if np.random.random() > 0.5 else 0
                q[i][j] = val
                q[j][i] = val
        candidates.append(quadrant_to_cell(q))
    
    return candidates

def design(problem):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Generate initial candidates
    candidates = generate_initial_candidates()
    
    # Evaluate all candidates
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        if time.time() > deadline:
            break
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
    
    if best_cell is None:
        # Fallback: return a simple cell
        size = 32
        return [[0]*size for _ in range(size)]
    
    # Simulated Annealing search with adaptive moves
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    T = 0.20  # Initial temperature
    T_min = 0.001
    cooling = 0.997  # Slower cooling
    
    eval_count = 0
    max_evals = 8000  # More evaluations with faster eval
    
    no_improve_count = 0
    patience = 200  # Reset patience if no improvement
    
    while time.time() < deadline and T > T_min and eval_count < max_evals:
        # Choose move type based on phase
        progress = eval_count / max_evals
        if progress < 0.3:
            # Early: more exploration
            move_type = np.random.random()
            if move_type < 0.5:
                n_flips = 1
            elif move_type < 0.8:
                n_flips = np.random.randint(2, 5)
            else:
                n_flips = np.random.randint(3, 8)
        elif progress < 0.7:
            # Mid: balanced
            move_type = np.random.random()
            if move_type < 0.6:
                n_flips = 1
            elif move_type < 0.9:
                n_flips = np.random.randint(2, 4)
            else:
                n_flips = np.random.randint(3, 6)
        else:
            # Late: exploitation
            move_type = np.random.random()
            if move_type < 0.7:
                n_flips = 1
            elif move_type < 0.95:
                n_flips = 2
            else:
                n_flips = np.random.randint(2, 4)
        
        new_cell = [row[:] for row in current_cell]
        for _ in range(n_flips):
            ii = np.random.randint(0, 8)
            jj = np.random.randint(0, 8)
            flip_pixel(new_cell, ii, jj)
        
        new_score = evaluate_cell(new_cell)
        eval_count += 1
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Adaptive temperature: increase if stuck
        if no_improve_count > patience:
            T = max(T, 0.1)  # Reheat
            no_improve_count = 0
        
        T *= cooling
    
    return best_cell
