import numpy as np
import time
import scipy.sparse.linalg as spla

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

def cell_to_matrix(cell):
    """Convert a 32x32 list of lists to a numpy array and compute mu_diff."""
    size = len(cell)
    eps = np.array([[12 if c == 1 else 1 for c in row] for row in cell], dtype=float)
    mu = 1.0 / eps
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
    return mu_diff

def evaluate_cell(cell, mu_diff=None):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    """
    if mu_diff is None:
        mu_diff = cell_to_matrix(cell)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for ki in range(len(K_POINTS)):
        kg_sq = KG_SQ[ki]
        B_inv_sqrt = 1.0 / np.sqrt(kg_sq)
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
        # Use sparse eigenvalue solver for speed, finding the 5 smallest eigenvalues
        # Since we need indices 3 and 4, we need at least 5.
        # S is symmetric.
        # Use eigsh with 'SA' (smallest algebraic) or 'SR' (smallest magnitude).
        # Eigenvalues should be positive.
        try:
            # k=5 to get 5 smallest
            evals, evecs = spla.eigsh(S, k=5, which='SA', return_eigenvectors=False, tol=1e-8)
            # eigsh returns unsorted in some cases? No, it returns sorted.
            # But let's be safe.
            evals = np.sort(evals)
            
            # Check for negative eigenvalues due to numerical error
            if evals[0] < 0:
                evals[0] = 0.0
            if evals[3] < 0:
                evals[3] = 0.0
            if evals[4] < 0:
                evals[4] = 0.0

            w4 = np.sqrt(evals[3])
            w5 = np.sqrt(evals[4])
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        except Exception:
            # Fallback to dense if sparse fails (rare)
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
    deadline = start_time + 115  # Safety margin for 120s limit
    
    size = 32
    candidates = []
    
    # 1. Rod (Circle)
    cell_rod = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.15**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    # 2. Square Rod
    cell_sq = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.1 and abs(y) < 0.1:
                cell_sq[i][j] = 1
    candidates.append(cell_sq)
    
    # 3. Cross
    cell_cross = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5):
                cell_cross[i][j] = 1
    candidates.append(cell_cross)
    
    # 4. Checkerboard
    cell_cb = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell_cb[i][j] = 1
    candidates.append(cell_cb)
    
    # 5. Inverted Checkerboard
    cell_icb = [[1 - cell_cb[i][j] for j in range(size)] for i in range(size)]
    candidates.append(cell_icb)
    
    # 6. Diagonal Bars
    cell_diag = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (abs(i - j) < 2.5 or abs(i + j - 31) < 2.5):
                cell_diag[i][j] = 1
    candidates.append(cell_diag)
    
    # 7. Hexagon-like (approximate with square)
    cell_hex = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < 0.15:
                cell_hex[i][j] = 1
    candidates.append(cell_hex)

    best_cell = None
    best_score = -float('inf')
    
    # Evaluate initial candidates
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
            
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    
    # Local Search with Multi-Flip Moves
    # We will try single flips and occasionally double flips.
    # To speed up, we can precompute the effect of a flip on mu_diff?
    # mu_diff is linear in the pixel values.
    # mu_diff_new = mu_diff_old + delta_mu_diff
    # delta_mu_diff can be precomputed for each possible flip (i,j).
    
    # There are 8x8 = 64 unique quadrant pixels.
    # Precompute delta_mu_diff for each of these 64 flips.
    # Note: Flipping (i,j) affects 8 pixels in the full 32x32 cell (or fewer if on symmetry axes).
    # The change in mu is: delta_mu(x,y) = +11/12 if 0->1, -11/12 if 1->0.
    # Wait, mu = 1/eps.
    # If 0->1: mu goes from 1 to 1/12. Delta mu = 1/12 - 1 = -11/12.
    # If 1->0: mu goes from 1/12 to 1. Delta mu = 1 - 1/12 = 11/12.
    
    # Let's precompute the delta_mu_diff matrix for flipping each quadrant pixel from 0 to 1.
    # Let base_cell be all zeros.
    # base_mu is all 1s.
    # base_mu_fft is delta at 0.
    # Actually, it's easier to just compute the FFT of the delta pattern.
    
    delta_mus = []
    for i in range(8):
        for j in range(8):
            delta_cell = [[0]*size for _ in range(size)]
            flip_pixel(delta_cell, i, j) # This sets the symmetric pixels to 1 (since 0->1)
            
            # delta_mu array
            # Where delta_cell is 1, mu changes by -11/12. Else 0.
            delta_mu = np.array([[-11.0/12.0 if c == 1 else 0.0 for c in row] for row in delta_cell])
            
            delta_mu_fft = np.fft.fft2(delta_mu)
            delta_mu_fft_shifted = np.fft.fftshift(delta_mu_fft)
            delta_mu_diff = delta_mu_fft_shifted[idx_x, idx_y] / (size * size)
            delta_mus.append(delta_mu_diff)
            
    delta_mus = np.array(delta_mus) # Shape (64, 289, 289)
    
    # Mapping from (i,j) to index in delta_mus
    # idx = i * 8 + j
    
    # Current mu_diff
    current_mu_diff = cell_to_matrix(current_cell)
    
    # Local search loop
    num_steps = 0
    while time.time() < deadline:
        # Choose a move
        # Mostly single flips, sometimes double flips
        if np.random.random() < 0.8:
            # Single flip
            i = np.random.randint(0, 8)
            j = np.random.randint(0, 8)
            idx = i * 8 + j
            
            # Determine if it's a 0->1 or 1->0 flip
            is_one = current_cell[i][j] == 1
            if is_one:
                # 1->0, so delta_mu is +11/12 where 1s are.
                # Our precomputed delta_mus[idx] is for 0->1 (which is -11/12).
                # So we need to subtract delta_mus[idx] (which is negative) -> add positive.
                # Or simply: new_mu = old_mu - delta_mus[idx]
                new_mu_diff = current_mu_diff - delta_mus[idx]
            else:
                # 0->1, so delta_mu is -11/12.
                # new_mu = old_mu + delta_mus[idx]
                new_mu_diff = current_mu_diff + delta_mus[idx]
                
            new_score = evaluate_cell(current_cell, new_mu_diff)
            
            if new_score > current_score:
                # Accept
                flip_pixel(current_cell, i, j)
                current_mu_diff = new_mu_diff
                current_score = new_score
                best_cell = [row[:] for row in current_cell]
                best_score = current_score
            else:
                # Reject, do nothing
                pass
        else:
            # Double flip
            i1 = np.random.randint(0, 8)
            j1 = np.random.randint(0, 8)
            i2 = np.random.randint(0, 8)
            j2 = np.random.randint(0, 8)
            
            if (i1, j1) == (i2, j2):
                continue
                
            idx1 = i1 * 8 + j1
            idx2 = i2 * 8 + j2
            
            is_one_1 = current_cell[i1][j1] == 1
            is_one_2 = current_cell[i2][j2] == 1
            
            new_mu_diff = current_mu_diff.copy()
            
            if is_one_1:
                new_mu_diff -= delta_mus[idx1]
            else:
                new_mu_diff += delta_mus[idx1]
                
            if is_one_2:
                new_mu_diff -= delta_mus[idx2]
            else:
                new_mu_diff += delta_mus[idx2]
                
            new_score = evaluate_cell(current_cell, new_mu_diff)
            
            if new_score > current_score:
                # Accept
                flip_pixel(current_cell, i1, j1)
                flip_pixel(current_cell, i2, j2)
                current_mu_diff = new_mu_diff
                current_score = new_score
                best_cell = [row[:] for row in current_cell]
                best_score = current_score
                
        num_steps += 1
        # Small sleep to avoid CPU spike? No, we are CPU bound.
        
    return best_cell
