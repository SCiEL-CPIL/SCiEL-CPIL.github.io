"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _apply_symmetry(cell):
    """Enforce 4-fold symmetry about center: flip rows, flip cols, transpose."""
    cell = np.array(cell, dtype=int)
    cell = np.minimum(cell, cell[::-1, :].copy())
    cell = np.minimum(cell, cell[:, ::-1].copy())
    cell = np.minimum(cell, cell.T.copy())
    return cell

def _make_four_rod_cell(size, offset, radius, shape_param=0.0):
    """Generate a 4-rod symmetric cell."""
    cell = np.zeros((size, size), dtype=int)
    centers = [(-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if shape_param < 0.5:
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radius:
                        cell[i, j] = 1
                else:
                    p = 2.0 + 2.0 * shape_param
                    val = (abs(dx)/radius)**p + (abs(dy)/radius)**p
                    if val < 1.0:
                        cell[i, j] = 1
    return _apply_symmetry(cell).tolist()

def _make_ring_cell(size, inner_r, outer_r):
    """Generate a dielectric ring cell."""
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r2 = x*x + y*y
            if inner_r*inner_r < r2 < outer_r*outer_r:
                cell[i, j] = 1
    return _apply_symmetry(cell).tolist()

def _make_cross_cell(size, arm_width, arm_length):
    """Generate a cross/plus shape cell."""
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if (abs(y) < arm_width/2 and abs(x) < arm_length/2) or \
               (abs(x) < arm_width/2 and abs(y) < arm_length/2):
                cell[i, j] = 1
    return _apply_symmetry(cell).tolist()

def _make_four_rings_cell(size, inner_r, outer_r, offset):
    """Generate four small rings at (±offset, ±offset)."""
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    r2 = dx*dx + dy*dy
                    if inner_r*inner_r < r2 < outer_r*outer_r:
                        cell[i, j] = 1
    return _apply_symmetry(cell).tolist()

def _make_staggered_cell(size, base_r, offset_r, offset_x, offset_y):
    """Generate a staggered pattern with base rod and offset rods."""
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Center rod
            if x*x + y*y < base_r*base_r:
                cell[i, j] = 1
            # Offset rods
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset_x
                    dy = y - sy * offset_y
                    if dx*dx + dy*dy < offset_r*offset_r:
                        cell[i, j] = 1
    return _apply_symmetry(cell).tolist()

def _make_checkerboard_cell(size, Px, Py, Dx, Dy, phi_x=0.0, phi_y=0.0):
    """
    Generate a cell from the product of two 1D square wave gratings.
    """
    cell = np.zeros((size, size), dtype=int)
    xs = (np.arange(size) + 0.5) / size - 0.5
    ys = (np.arange(size) + 0.5) / size - 0.5
    
    ux = (xs + 0.5 * Px + phi_x) % Px
    Sy_x = (np.abs(ux - 0.5 * Px) < 0.5 * Dx * Px).astype(int)
    
    uy = (ys + 0.5 * Py + phi_y) % Py
    Sy_y = (np.abs(uy - 0.5 * Py) < 0.5 * Dy * Py).astype(int)
    
    cell_2d = np.outer(Sy_x, Sy_y)
    
    return _apply_symmetry(cell_2d).tolist()

def _compute_eps_fft(cell, size):
    """Compute the full 2D FFT of the epsilon field."""
    eps = np.array(cell, dtype=float)
    eps = 1.0 + (12.0 - 1.0) * eps
    return np.fft.fftn(eps) / (size * size)

def _tm_eigenvalues_full(kx, ky, full_eps_fft, size, pw_idx):
    """
    Solve TM eigenvalue problem using full FFT for epsilon coefficients.
    H_{ij} = (k_i . k_j) * eps_{G_i - G_j}
    """
    n_pw = len(pw_idx)
    N = n_pw * n_pw
    
    Gx = np.repeat(pw_idx, n_pw)
    Gy = np.tile(pw_idx, n_pw)
    
    kx_g = kx + Gx
    ky_g = ky + Gy
    
    K = np.column_stack((kx_g, ky_g))
    K_dot_K = K @ K.T
    
    Gx_diff = Gx[:, None] - Gx[None, :]
    Gy_diff = Gy[:, None] - Gy[None, :]
    
    # Map differences to FFT indices
    idx_m = Gx_diff % size
    idx_n = Gy_diff % size
    
    # Extract epsilon coefficients from full FFT
    eps_coeffs = full_eps_fft[idx_m, idx_n]
    
    H = K_dot_K * eps_coeffs
    H = (H + H.conj().T) / 2.0
    
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals)

def _quick_gap_estimate(cell, size=32, n_pw=7, n_kpts=8):
    """Fast surrogate evaluation using reduced plane-wave basis."""
    if n_pw % 2 == 1:
        half = n_pw // 2
        pw_idx = np.arange(-half, half+1)
    else:
        half = n_pw // 2
        pw_idx = np.arange(-half, half)
        
    full_eps_fft = _compute_eps_fft(cell, size)
    
    k_points = [
        (0.0, 0.0), (0.5, 0.0), (0.5, 0.5),
        (0.25, 0.25), (0.3, 0.1), (0.1, 0.3),
        (0.4, 0.2), (0.2, 0.4),
    ][:n_kpts]
    
    band4_max = 0.0
    band5_min = np.inf
    
    for kx, ky in k_points:
        eigvals = _tm_eigenvalues_full(kx, ky, full_eps_fft, size, pw_idx)
        if len(eigvals) >= 5:
            band4_max = max(band4_max, eigvals[3])
            band5_min = min(band5_min, eigvals[4])
            
    if band5_min <= band4_max:
        return -1.0
    gap = band5_min - band4_max
    midgap = (band5_min + band4_max) / 2.0
    if midgap == 0:
        return 0.0
    return 2.0 * gap / midgap

def _mutate_cell(cell, n_flips):
    """Randomly flip n pixels while maintaining symmetry."""
    cell = cell.copy()
    size = cell.shape[0]
    for _ in range(n_flips):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        new_val = 1 - cell[i, j]
        positions = [
            (i, j), (j, i),
            (size-1-i, size-1-j), (size-1-j, size-1-i),
            (i, size-1-j), (size-1-j, i),
            (size-1-i, j), (j, size-1-i),
        ]
        for pi, pj in positions:
            cell[pi, pj] = new_val
    return cell

def _mutate_checkerboard(params, size):
    """Mutate checkerboard parameters slightly."""
    new_params = params.copy()
    # Small random perturbations
    new_params[0] += random.gauss(0, 0.02)  # Px
    new_params[1] += random.gauss(0, 0.02)  # Py
    new_params[2] += random.gauss(0, 0.05)  # Dx
    new_params[3] += random.gauss(0, 0.05)  # Dy
    new_params[4] += random.gauss(0, 0.1)   # phi_x
    new_params[5] += random.gauss(0, 0.1)   # phi_y
    
    # Clamp values
    new_params[0] = np.clip(new_params[0], 0.3, 0.8)
    new_params[1] = np.clip(new_params[1], 0.3, 0.8)
    new_params[2] = np.clip(new_params[2], 0.1, 0.9)
    new_params[3] = np.clip(new_params[3], 0.1, 0.9)
    
    return new_params

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the 4-5 TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0  # 120s limit, leave 10s margin
    random.seed(42)
    
    best_cell = None
    best_score = -np.inf
    
    # Phase 1: Geometric seeds
    seeds = []
    
    # Four-rod patterns
    for offset in [0.20, 0.25, 0.30]:
        for radius in [0.10, 0.14, 0.18]:
            for shape in [0.0, 0.5, 1.0]:
                seeds.append(_make_four_rod_cell(size, offset, radius, shape))
    
    # Ring patterns
    for inner_r in [0.15, 0.20, 0.25]:
        for outer_r in [0.35, 0.40, 0.45]:
            seeds.append(_make_ring_cell(size, inner_r, outer_r))
    
    # Cross patterns
    for arm_w in [0.10, 0.15, 0.20]:
        for arm_l in [0.30, 0.35, 0.40]:
            seeds.append(_make_cross_cell(size, arm_w, arm_l))
    
    # Four rings
    for inner_r in [0.05, 0.08]:
        for outer_r in [0.12, 0.15]:
            for offset in [0.20, 0.25, 0.30]:
                seeds.append(_make_four_rings_cell(size, inner_r, outer_r, offset))
    
    # Staggered patterns
    for base_r in [0.15, 0.20]:
        for offset_r in [0.08, 0.10]:
            for offset in [0.25, 0.30]:
                seeds.append(_make_staggered_cell(size, base_r, offset_r, offset, offset))
    
    # Checkerboard patterns (new family)
    for Px in [0.5, 0.6, 0.7]:
        for Py in [0.5, 0.6, 0.7]:
            for Dx in [0.4, 0.5, 0.6]:
                for Dy in [0.4, 0.5, 0.6]:
                    seeds.append(_make_checkerboard_cell(size, Px, Py, Dx, Dy))
    
    for cell in seeds:
        if time.time() > deadline - 50.0:
            break
        score = _quick_gap_estimate(cell, size)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Phase 2: Stochastic search from best seed
    if best_cell is not None:
        best_arr = np.array(best_cell)
        current_arr = best_arr.copy()
        current_score = best_score
        
        n_iter = 0
        while time.time() < deadline - 10.0:
            n_iter += 1
            n_flips = 1
            
            trial = _mutate_cell(current_arr, n_flips)
            trial_score = _quick_gap_estimate(trial.tolist(), size)
            
            if trial_score > current_score:
                current_arr = trial
                current_score = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_cell = trial.tolist()
    
    # Phase 3: Checkerboard parameter optimization
    if time.time() < deadline - 20.0:
        # Try optimizing checkerboard parameters
        best_cb_params = None
        best_cb_score = -np.inf
        
        # Initial grid search
        for Px in np.arange(0.4, 0.8, 0.1):
            for Py in np.arange(0.4, 0.8, 0.1):
                for Dx in np.arange(0.3, 0.7, 0.1):
                    for Dy in np.arange(0.3, 0.7, 0.1):
                        cell = _make_checkerboard_cell(size, Px, Py, Dx, Dy)
                        score = _quick_gap_estimate(cell, size)
                        if score > best_cb_score:
                            best_cb_score = score
                            best_cb_params = [Px, Py, Dx, Dy, 0.0, 0.0]
        
        if best_cb_params is not None and best_cb_score > best_score:
            best_score = best_cb_score
            best_cell = _make_checkerboard_cell(size, *best_cb_params)
        
        # Local refinement of checkerboard parameters
        if best_cb_params is not None:
            current_params = best_cb_params.copy()
            current_score = best_cb_score
            
            for _ in range(500):
                if time.time() > deadline - 15.0:
                    break
                new_params = _mutate_checkerboard(current_params, size)
                cell = _make_checkerboard_cell(size, *new_params)
                score = _quick_gap_estimate(cell, size)
                
                if score > current_score:
                    current_params = new_params
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_cell = cell
    
    # Phase 4: Random restarts
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        fill_prob = random.uniform(0.2, 0.5)
        rand_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if i <= j and i + j < size:
                    val = 1 if random.random() < fill_prob else 0
                    rand_cell[i, j] = val
                    rand_cell[j, i] = val
                    rand_cell[size-1-i, size-1-j] = val
                    rand_cell[size-1-j, size-1-i] = val
        
        rand_score = _quick_gap_estimate(rand_cell.tolist(), size)
        
        if rand_score > best_score:
            best_score = rand_score
            best_cell = rand_cell.tolist()
            
        current_arr = rand_cell.copy()
        current_score = rand_score
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            n_flips = 1
            trial = _mutate_cell(current_arr, n_flips)
            trial_score = _quick_gap_estimate(trial.tolist(), size)
            if trial_score > current_score:
                current_arr = trial
                current_score = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_cell = trial.tolist()
    
    if best_cell is None:
        best_cell = _make_four_rod_cell(size, 0.25, 0.12, 0.0)
    
    return best_cell
# EVOLVE-BLOCK-END
