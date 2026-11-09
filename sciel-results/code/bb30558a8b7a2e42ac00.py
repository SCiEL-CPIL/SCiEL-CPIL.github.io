import numpy as np
import time
import random

# Precompute G vectors
N_PW = 8
G_range = np.arange(-N_PW, N_PW + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
Gx_flat = G_vectors[:, 0]
Gy_flat = G_vectors[:, 1]

# Precompute differences G_i - G_j
# We need mu(G_i - G_j)
# G_i, G_j are in [-8, 8]. Diff is in [-16, 16].
# We will map diff to index in a 33x33 array.
# idx = diff + 16
Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
# Shape: (289, 289)
idx_diff_x = Diff_x + 16
idx_diff_y = Diff_y + 16

# Precompute k-points and (k+G)^2
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
    # G vectors are in units of 2*pi? 
    # In the code, G_vectors are integers.
    # The physical G is 2*pi * G_int.
    # k is in rad/m (or similar).
    # (k + 2*pi*G_int)^2
    kg = k[None, :] + G_vectors * (2 * np.pi)
    kg_sq = np.sum(kg**2, axis=1)
    KG_SQ.append(kg_sq)
KG_SQ = np.array(KG_SQ) # Shape: (30, 289)

def evaluate_cell(cell):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    cell: list of lists of 0/1.
    """
    size = 32
    # 1 = dielectric (eps=12), 0 = air (eps=1)
    # mu = 1/eps
    eps = np.array([[12 if c == 1 else 1 for c in row] for row in cell], dtype=float)
    mu = 1.0 / eps
    
    # Compute Fourier coefficients of mu
    # c[n,m] = 1/N^2 sum mu[x,y] exp(-i 2pi (nx+my)/N)
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    # mu_fft_shifted is indexed by n, m in [-16, 16] for 32x32?
    # fftshift shifts the zero frequency to the center.
    # For N=32, indices 0..31 become -16..15.
    # We need indices -16..16.
    # The max diff is 16.
    # mu_fft_shifted[16 + dx, 16 + dy] corresponds to coeff for G_diff = (dx, dy).
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for ki in range(len(K_POINTS)):
        # Build matrix A
        # A[i, j] = mu(G_i - G_j) * (k + G_j)^2
        # mu(G_i - G_j) is mu_fft_shifted[idx_diff_x[i,j], idx_diff_y[i,j]] / (size*size)
        
        # Extract submatrix of mu coeffs
        mu_sub = mu_fft_shifted[idx_diff_x, idx_diff_y] / (size * size)
        
        # Multiply by (k+G_j)^2
        # KG_SQ[ki] is shape (289,)
        # A[i,j] = mu_sub[i,j] * KG_SQ[ki, j]
        A = mu_sub * KG_SQ[ki, None, :]
        
        # Eigenvalues
        evals = np.linalg.eigvalsh(A)
        
        # We need the 4th and 5th bands (indices 3 and 4)
        # Ensure positive eigenvalues
        evals = np.maximum(evals, 0)
        
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
    """Convert 8x8 symmetric quadrant to 32x32 cell."""
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

def generate_perlin_noise(size, seed, scale):
    """
    Generate simple value noise on an 8x8 grid.
    """
    rng = np.random.RandomState(seed)
    # Generate low resolution grid
    low_size = max(2, int(size / scale))
    low_noise = rng.rand(low_size, low_size)
    
    # Upsample using bilinear interpolation
    # Create coordinates
    x_coords = np.linspace(0, low_size - 1, size)
    y_coords = np.linspace(0, low_size - 1, size)
    
    # For simplicity, use nearest neighbor or simple linear interp
    # Since 8x8 is small, we can just do a simple loop or use scipy if available
    # But standard library only. Let's do a simple linear interpolation manually or just use random values if scale is 1.
    # Actually, for 8x8, "noise" can just be random values smoothed by a simple kernel.
    
    # Simple approach: Random 8x8, then apply a simple blur if scale > 1
    noise = rng.rand(8, 8)
    
    # If scale > 1, we want smoother noise.
    # We can generate low-res noise and interpolate.
    if scale > 1:
        low_res = rng.rand(low_size, low_size)
        # Bilinear interpolation
        noise = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                # Map 0..7 to 0..low_size-1
                y = i * (low_size - 1) / 7.0
                x = j * (low_size - 1) / 7.0
                y0 = int(y)
                x0 = int(x)
                y1 = min(y0 + 1, low_size - 1)
                x1 = min(x0 + 1, low_size - 1)
                fy = y - y0
                fx = x - x0
                noise[i, j] = (1-fy)*(1-fx)*low_res[y0, x0] + (1-fy)*fx*low_res[y0, x1] + fy*(1-fx)*low_res[y1, x0] + fy*fx*low_res[y1, x1]
    
    return noise

def generate_noise_cell(seed, freq, thresh):
    """
    Generate a cell from noise.
    """
    noise = generate_perlin_noise(8, seed, freq)
    # Threshold
    q = (noise > thresh).astype(int)
    return quadrant_to_cell(q)

def design(problem):
    start_time = time.time()
    deadline = start_time + 180  # Safety margin for 230s limit
    
    size = 32
    candidates = []
    
    # 1. Geometric Candidates
    # Rod
    cell_rod = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
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
    
    # Cross
    cell_cross = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5):
                cell_cross[i][j] = 1
    candidates.append(cell_cross)
    
    # 2. Noise Candidates
    # Generate diverse noise patterns
    for seed in range(20):
        for freq in [1, 2, 4]:
            for thresh in [0.3, 0.5, 0.7]:
                candidates.append(generate_noise_cell(seed, freq, thresh))
    
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
            
    # Simulated Annealing
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    T = 0.1
    T_min = 0.001
    cooling = 0.99
    
    # Precompute indices for 8x8 quadrant
    q_indices = [(i, j) for i in range(8) for j in range(8)]
    
    while time.time() < deadline and T > T_min:
        # Random flip
        i, j = random.choice(q_indices)
        
        new_cell = [row[:] for row in current_cell]
        flip_pixel(new_cell, i, j)
        
        new_score = evaluate_cell(new_cell)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
        
        T *= cooling
        
        # Occasional multi-flip
        if random.random() < 0.1:
            new_cell2 = [row[:] for row in current_cell]
            for _ in range(3):
                ii, jj = random.choice(q_indices)
                flip_pixel(new_cell2, ii, jj)
            new_score2 = evaluate_cell(new_cell2)
            delta2 = new_score2 - current_score
            if delta2 > 0 or random.random() < np.exp(delta2 / T):
                current_cell = new_cell2
                current_score = new_score2
                if current_score > best_score:
                    best_score = current_score
                    best_cell = [row[:] for row in current_cell]
        
        T *= cooling
        
    return best_cell
