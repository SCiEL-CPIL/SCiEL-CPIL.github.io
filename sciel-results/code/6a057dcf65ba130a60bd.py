"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _apply_symmetry(cell):
    """Enforce 4-fold symmetry about center: flip rows, flip cols, transpose."""
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

def _compute_eps_ft(cell, size, pw_idx):
    """Compute 2D Fourier coefficients of epsilon field for given plane-wave indices."""
    eps = np.array(cell, dtype=float)
    eps = 1.0 + (12.0 - 1.0) * eps
    n_pw = len(pw_idx)
    eps_ft = np.zeros((n_pw, n_pw), dtype=complex)
    coords = np.arange(size) / size
    for im, m in enumerate(pw_idx):
        for in_, n in enumerate(pw_idx):
            phase_x = np.exp(-1j * 2.0 * np.pi * m * coords)
            phase_y = np.exp(-1j * 2.0 * np.pi * n * coords)
            phase2d = phase_x[:, None] * phase_y[None, :]
            eps_ft[im, in_] = np.sum(eps * phase2d) / (size * size)
    return eps_ft

def _tm_eigenvalues(kx, ky, eps_ft, pw_idx, size):
    """Solve TM eigenvalue problem at a single k-point."""
    n_pw = len(pw_idx)
    N = n_pw * n_pw
    H = np.zeros((N, N), dtype=complex)
    pw_map = {v: i for i, v in enumerate(pw_idx)}
    for im, m in enumerate(pw_idx):
        for in_, n in enumerate(pw_idx):
            idx = im * n_pw + in_
            kx_g = kx + m
            ky_g = ky + n
            k_g_sq = kx_g**2 + ky_g**2
            for im2, m2 in enumerate(pw_idx):
                for in2, n2 in enumerate(pw_idx):
                    idx2 = im2 * n_pw + in2
                    dm = m - m2
                    dn = n - n2
                    if dm in pw_map and dn in pw_map:
                        H[idx, idx2] += k_g_sq * eps_ft[pw_map[dm], pw_map[dn]]
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals)

def _quick_gap_estimate(cell, size=32, n_pw=11, n_kpts=8):
    """Fast surrogate evaluation using reduced plane-wave basis."""
    if n_pw % 2 == 1:
        half = n_pw // 2
        pw_idx = np.arange(-half, half+1)
    else:
        half = n_pw // 2
        pw_idx = np.arange(-half, half)
    eps_ft = _compute_eps_ft(cell, size, pw_idx)
    k_points = [
        (0.0, 0.0), (0.5, 0.0), (0.5, 0.5),
        (0.25, 0.25), (0.3, 0.1), (0.1, 0.3),
        (0.4, 0.2), (0.2, 0.4),
    ][:n_kpts]
    band4_max = 0.0
    band5_min = np.inf
    for kx, ky in k_points:
        eigvals = _tm_eigenvalues(kx, ky, eps_ft, pw_idx, size)
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

def _random_symmetric_cell(size, fill_prob):
    """Generate a random symmetric binary cell."""
    # Generate upper-right quadrant with 4-fold symmetry
    # The cell has symmetry: cell[i,j] == cell[size-1-i, j] == cell[i, size-1-j] == cell[size-1-i, size-1-j]
    # Also cell[i,j] == cell[j,i] (transpose symmetry)
    # So we need: cell[i,j] == cell[j,i] == cell[size-1-i, size-1-j] == cell[size-1-j, size-1-i]
    # Unique positions: for i <= j and i + j < size (roughly the fundamental domain)
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if i <= j and i + j < size:
                val = 1 if random.random() < fill_prob else 0
                # Apply all symmetries
                cell[i, j] = val
                cell[j, i] = val
                cell[size-1-i, size-1-j] = val
                cell[size-1-j, size-1-i] = val
    return cell

def _mutate_cell(cell, n_flips):
    """Randomly flip n pixels while maintaining symmetry."""
    cell = cell.copy()
    size = cell.shape[0]
    for _ in range(n_flips):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        new_val = 1 - cell[i, j]
        # Apply all 8 symmetries (4-fold + transpose)
        positions = [
            (i, j), (j, i),
            (size-1-i, size-1-j), (size-1-j, size-1-i),
            (i, size-1-j), (size-1-j, i),
            (size-1-i, j), (j, size-1-i),
        ]
        for pi, pj in positions:
            cell[pi, pj] = new_val
    return cell

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the 4-5 TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 210.0
    random.seed(42)
    
    best_cell = None
    best_score = -np.inf
    
    # Phase 1: Geometric seeds
    seeds = []
    for offset in [0.20, 0.25, 0.30]:
        for radius in [0.10, 0.14, 0.18]:
            for shape in [0.0, 0.5, 1.0]:
                seeds.append(_make_four_rod_cell(size, offset, radius, shape))
    
    for cell in seeds:
        if time.time() > deadline - 100.0:
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
        while time.time() < deadline - 30.0 and n_iter < 5000:
            n_iter += 1
            # Annealing: start with more flips, reduce over time
            progress = min(n_iter / 3000.0, 1.0)
            n_flips = max(1, int(5 * (1 - progress) + 1))
            
            trial = _mutate_cell(current_arr, n_flips)
            trial_score = _quick_gap_estimate(trial.tolist(), size)
            
            if trial_score > current_score:
                current_arr = trial
                current_score = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_cell = trial.tolist()
            # Occasionally accept worse solutions (simulated annealing)
            elif random.random() < 0.05 * (1 - progress):
                current_arr = trial
                current_score = trial_score
    
    # Phase 3: Random restarts with stochastic search
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        # Random symmetric cell
        fill_prob = random.uniform(0.2, 0.5)
        rand_cell = _random_symmetric_cell(size, fill_prob)
        rand_score = _quick_gap_estimate(rand_cell.tolist(), size)
        
        if rand_score > best_score:
            best_score = rand_score
            best_cell = rand_cell.tolist()
        
        # Local search from this random start
        current_arr = rand_cell.copy()
        current_score = rand_score
        for _ in range(500):
            if time.time() > deadline - 15.0:
                break
            n_flips = random.randint(1, 4)
            trial = _mutate_cell(current_arr, n_flips)
            trial_score = _quick_gap_estimate(trial.tolist(), size)
            if trial_score > current_score:
                current_arr = trial
                current_score = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_cell = trial.tolist()
    
    # Phase 4: Fine-tuning on best cell
    if best_cell is not None and time.time() < deadline - 10.0:
        best_arr = np.array(best_cell)
        current_arr = best_arr.copy()
        current_score = best_score
        for _ in range(2000):
            if time.time() > deadline - 5.0:
                break
            trial = _mutate_cell(current_arr, 1)
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
