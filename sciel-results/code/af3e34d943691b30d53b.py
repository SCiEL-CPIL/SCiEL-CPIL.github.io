import time
import numpy as np
from numpy.linalg import eigh

def _compute_gap_fast(cell, n_k=8, pw_radius=10):
    N = len(cell)
    inv_eps = 1.0 / (1.0 + 11.0 * np.array(cell, dtype=float))
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    size_d = 4 * pw_radius + 1
    offset_d = 2 * pw_radius
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    # Vectorized indexing for inv_eps matrix
    dqx = Gx_diff.astype(int)
    dqy = Gy_diff.astype(int)
    inv_eps_mat = fft_inv[dqx % N, dqy % N]
    
    k_points = [
        (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.5, 0.25),
        (0.5, 0.5), (0.35, 0.35), (0.4, 0.2), (0.2, 0.1)
    ][:n_k]
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2 * (min_w5 - max_w4) / (min_w5 + max_w4)


def _compute_gap_smooth(cell, n_k=8, pw_radius=10, beta=20.0):
    """Smooth surrogate using soft-min/soft-max for the gap objective."""
    N = len(cell)
    inv_eps = 1.0 / (1.0 + 11.0 * np.array(cell, dtype=float))
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    dqx = Gx_diff.astype(int)
    dqy = Gy_diff.astype(int)
    inv_eps_mat = fft_inv[dqx % N, dqy % N]
    
    k_points = [
        (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.5, 0.25),
        (0.5, 0.5), (0.35, 0.35), (0.4, 0.2), (0.2, 0.1)
    ][:n_k]
    
    w4_vals = []
    w5_vals = []
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                w4_vals.append(eigvals[3])
                w5_vals.append(eigvals[4])
        except:
            pass
    
    if len(w4_vals) == 0:
        return -1.0
    
    w4_arr = np.array(w4_vals)
    w5_arr = np.array(w5_vals)
    
    # Soft-max of w4 (approximates max)
    m4 = np.max(w4_arr)
    soft_max_w4 = m4 + np.log(np.sum(np.exp(w4_arr - m4)))
    
    # Soft-min of w5 (approximates min)
    m5 = np.min(w5_arr)
    soft_min_w5 = m5 - np.log(np.sum(np.exp(-w5_arr + m5)))
    
    if soft_min_w5 <= 0 or soft_max_w4 <= 0:
        return -1.0
    return 2 * (soft_min_w5 - soft_max_w4) / (soft_min_w5 + soft_max_w4)


def _make_symmetric(cell):
    """Ensure cell is symmetric under transpose, row-reverse, col-reverse."""
    N = len(cell)
    c = np.array(cell, dtype=int)
    # Average over the 4 symmetry operations
    c_t = c.T
    c_r = c[::-1, :]
    c_c = c[:, ::-1]
    c_rt = c[::-1, ::-1]
    c_averaged = (c + c_t + c_r + c_c) / 4.0
    c_sym = (c_averaged > 0.5).astype(int)
    # Verify symmetry
    c_sym = np.maximum(c_sym, c_sym.T)  # ensure transpose symmetry
    c_sym = np.maximum(c_sym, c_sym[::-1, :])  # row reverse
    c_sym = np.maximum(c_sym, c_sym[:, ::-1])  # col reverse
    return c_sym.tolist()


def _gen_rod(r):
    """Circular dielectric rod of radius r (in units of cell size)."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < r*r:
                cell[i][j] = 1
    return cell


def _gen_hole(r):
    """Dielectric with circular air hole of radius r."""
    size = 32
    cell = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < r*r:
                cell[i][j] = 0
    return cell


def _gen_square(r):
    """Square dielectric inclusion of half-size r."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < r and abs(y) < r:
                cell[i][j] = 1
    return cell


def _gen_cross(arm_width, arm_length):
    """Cross/plus shape."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if (abs(x) < arm_width/2 and abs(y) < arm_length/2) or \
               (abs(y) < arm_width/2 and abs(x) < arm_length/2):
                cell[i][j] = 1
    return cell


def _gen_random_symmetric(density):
    """Random symmetric pixel pattern."""
    size = 32
    cell = np.zeros((size, size), dtype=int)
    # Fill the fundamental domain (top-left quadrant) randomly
    for i in range(size//2):
        for j in range(size//2):
            if np.random.random() < density:
                # Mirror to all 4 quadrants
                cell[i, j] = 1
                cell[size-1-i, j] = 1
                cell[i, size-1-j] = 1
                cell[size-1-i, size-1-j] = 1
    return cell.tolist()


def _gen_ring(inner_r, outer_r):
    """Dielectric ring (annulus)."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r2 = x*x + y*y
            if inner_r*inner_r < r2 < outer_r*outer_r:
                cell[i][j] = 1
    return cell


def _gen_diamond(r):
    """Diamond (rotated square) dielectric inclusion."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < r:
                cell[i][j] = 1
    return cell


def _gen_four_rods(rod_r, offset):
    """Four small rods in a square arrangement."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    centers = [(offset, offset), (offset, -offset), (-offset, offset), (-offset, -offset)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx*dx + dy*dy < rod_r*rod_r:
                    cell[i][j] = 1
                    break
    return cell


def _mutate(cell, mutation_rate=0.1):
    """Randomly flip some pixels while maintaining symmetry."""
    size = len(cell)
    c = np.array(cell, dtype=int)
    # Flip in the fundamental domain
    for i in range(size//2):
        for j in range(size//2):
            if np.random.random() < mutation_rate:
                val = 1 - c[i, j]
                c[i, j] = val
                c[size-1-i, j] = val
                c[i, size-1-j] = val
                c[size-1-i, size-1-j] = val
    return c.tolist()


def _mutate_adaptive(cell, current_score, target_improve=0.01):
    """Adaptive mutation rate based on how far from target."""
    # If score is poor, mutate more aggressively
    base_rate = 0.08
    if current_score < 0.0:
        rate = 0.2
    elif current_score < 0.1:
        rate = 0.15
    elif current_score < 0.2:
        rate = 0.1
    else:
        rate = 0.05
    return _mutate(cell, mutation_rate=rate)


def design(problem):
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin
    
    size = problem["pixels"]
    assert size == 32
    
    # Initial candidates
    candidates = []
    
    # Rod of various sizes
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append(_gen_rod(r))
    
    # Square inclusions
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(_gen_square(r))
    
    # Cross shapes
    for w in [0.05, 0.1, 0.15]:
        for l in [0.3, 0.4, 0.45]:
            candidates.append(_gen_cross(w, l))
    
    # Dielectric with hole
    for r in [0.3, 0.4, 0.5]:
        candidates.append(_gen_hole(r))
    
    # Rings
    for inner in [0.1, 0.15, 0.2]:
        for outer in [0.3, 0.35, 0.4, 0.45]:
            if outer > inner:
                candidates.append(_gen_ring(inner, outer))
    
    # Diamonds
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(_gen_diamond(r))
    
    # Four rods
    for rod_r in [0.08, 0.1, 0.12]:
        for offset in [0.15, 0.2, 0.25]:
            candidates.append(_gen_four_rods(rod_r, offset))
    
    # Random symmetric patterns
    for density in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for _ in range(3):
            candidates.append(_gen_random_symmetric(density))
    
    # Evaluate all candidates with fast evaluator
    best_cell = None
    best_score = -1.0
    
    for idx, cell in enumerate(candidates):
        if time.time() > deadline - 40:
            break
        try:
            score = _compute_gap_fast(cell, n_k=5, pw_radius=7)
            if score > best_score:
                best_score = score
                best_cell = cell
        except:
            pass
    
    # Local search using smooth surrogate for guidance, then verify with exact
    if best_cell is not None and time.time() < deadline - 50:
        current = best_cell
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 30
        
        while time.time() < deadline - 30 and no_improve_count < max_no_improve:
            # Adaptive mutation
            mutated = _mutate_adaptive(current, current_score)
            try:
                # Use smooth surrogate for quick evaluation during search
                smooth_score = _compute_gap_smooth(mutated, n_k=5, pw_radius=7, beta=15.0)
                if smooth_score > current_score - 0.02:  # Accept if close or better
                    # Verify with exact evaluator
                    exact_score = _compute_gap_fast(mutated, n_k=6, pw_radius=8)
                    if exact_score > current_score:
                        current_score = exact_score
                        current = mutated
                        best_score = max(best_score, exact_score)
                        best_cell = current
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            except:
                no_improve_count += 1
    
    # Final polish: try more random candidates if time allows
    if time.time() < deadline - 30:
        for _ in range(30):
            if time.time() > deadline - 20:
                break
            cell = _gen_random_symmetric(np.random.uniform(0.1, 0.45))
            try:
                score = _compute_gap_fast(cell, n_k=5, pw_radius=7)
                if score > best_score:
                    best_score = score
                    best_cell = cell
                    # Quick local search from this new best
                    if time.time() < deadline - 25:
                        for _inner in range(20):
                            if time.time() > deadline - 20:
                                break
                            mutated = _mutate_adaptive(cell, best_score)
                            s = _compute_gap_fast(mutated, n_k=5, pw_radius=7)
                            if s > best_score:
                                best_score = s
                                cell = mutated
                                best_cell = cell
            except:
                pass
    
    if best_cell is None:
        # Fallback: seed
        best_cell = _gen_rod(0.2)
    
    return best_cell
