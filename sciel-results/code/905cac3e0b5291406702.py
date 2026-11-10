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

def generate_fourier_cell(A_diag, A_edge, A_diag2, A_edge2, threshold):
    """
    Generate a C4v-symmetric cell from a Fourier superposition.
    This is the structural idea from inspiration 1.
    """
    size = 32
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    # Center coordinates
    x_c = (X - (size - 1) / 2.0) / size
    y_c = (Y - (size - 1) / 2.0) / size
    
    # Superposition of modes
    # Mode (1,1): targets M point
    # Mode (1,0), (0,1): target X point
    # Mode (2,2): higher order diagonal
    # Mode (2,0), (0,2): higher order edge
    V = A_diag * np.cos(2 * np.pi * x_c) * np.cos(2 * np.pi * y_c)
    V += A_edge * (np.cos(2 * np.pi * x_c) + np.cos(2 * np.pi * y_c))
    V += A_diag2 * np.cos(4 * np.pi * x_c) * np.cos(4 * np.pi * y_c)
    V += A_edge2 * (np.cos(4 * np.pi * x_c) + np.cos(4 * np.pi * y_c))
    
    # Normalize
    V = V - np.mean(V)
    std = np.std(V)
    if std > 0:
        V = V / std
        
    # Threshold
    cell_np = (V > threshold).astype(int)
    
    # Enforce symmetry explicitly by averaging with symmetric counterparts
    # and re-thresholding, or just use the top-left quadrant to be safe
    # Since the function is mathematically symmetric, discretization might break it slightly.
    # Let's extract quadrant and rebuild to guarantee symmetry.
    q = [[cell_np[i][j] for j in range(8)] for i in range(8)]
    for i in range(8):
        for j in range(i, 8):
            val = (q[i][j] + q[j][i]) // 2
            q[i][j] = val
            q[j][i] = val
            
    return make_cell_from_quadrant(q)

def generate_hole_cell(radius, centers):
    """
    Generate a dielectric background with air holes.
    Structural idea from inspiration 2 (high score).
    """
    size = 32
    cell = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            is_air = 0
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < radius * radius:
                    is_air = 1
                    break
            if is_air:
                cell[i][j] = 0
    return cell

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 230s kill
    
    size = 32
    candidates = []
    
    # 1. High-performing geometric seed (Inspiration 2)
    # 4 holes at (+/-0.25, +/-0.25) with radius 0.28
    centers = [
        (0.25, 0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (-0.25, -0.25)
    ]
    # Vary radius slightly
    for r in [0.25, 0.28, 0.31]:
        candidates.append(generate_hole_cell(r, centers))
        
    # 2. Fourier synthesis seeds (Inspiration 1)
    # Sweep amplitudes and thresholds
    for A_diag in [0.5, 1.0, 1.5]:
        for A_edge in [0.0, 0.5, 1.0]:
            for A_diag2 in [0.0, 0.5]:
                for A_edge2 in [0.0, 0.5]:
                    for thresh in [0.0, 0.2, 0.4]:
                        candidates.append(generate_fourier_cell(A_diag, A_edge, A_diag2, A_edge2, thresh))
                        
    # 3. Other geometric variations
    # Single central hole
    candidates.append(generate_hole_cell(0.2, [(0,0)]))
    # Square hole
    cell_sq = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if max(abs(x), abs(y)) < 0.2:
                cell_sq[i][j] = 0
    candidates.append(cell_sq)

    # Evaluate all candidates
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
            
    # If no candidates were good, fallback to the known good one
    if best_cell is None:
        best_cell = generate_hole_cell(0.28, centers)
        best_score = evaluate_cell(best_cell)

    # Simulated Annealing search on the best candidate
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    T = 0.15
    T_min = 0.005
    cooling = 0.997
    
    # Use a random seed for reproducibility within the run
    rng = np.random.RandomState(42)
    
    while time.time() < deadline and T > T_min:
        i = rng.randint(0, 8)
        j = rng.randint(0, 8)
        
        new_cell = [row[:] for row in current_cell]
        flip_pixel(new_cell, i, j)
        
        new_score = evaluate_cell(new_cell)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
        
        T *= cooling
        
        # Multi-flip move
        if rng.random() < 0.15:
            n_flips = rng.randint(2, 6)
            new_cell2 = [row[:] for row in current_cell]
            for _ in range(n_flips):
                ii = rng.randint(0, 8)
                jj = rng.randint(0, 8)
                flip_pixel(new_cell2, ii, jj)
            new_score2 = evaluate_cell(new_cell2)
            delta2 = new_score2 - current_score
            if delta2 > 0 or rng.random() < np.exp(delta2 / T):
                current_cell = new_cell2
                current_score = new_score2
                if current_score > best_score:
                    best_score = current_score
                    best_cell = [row[:] for row in current_cell]
        
        T *= cooling
        
    return best_cell
