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
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N]
    
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
        
        inv_eps_mat = inv_eps_full[
            Gx_diff.astype(int) + offset_d, 
            Gy_diff.astype(int) + offset_d
        ]
        
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


def _make_symmetric(cell):
    N = len(cell)
    c = np.array(cell, dtype=int)
    c_t = c.T
    c_r = c[::-1, :]
    c_c = c[:, ::-1]
    c_rt = c[::-1, ::-1]
    c_averaged = (c + c_t + c_r + c_c) / 4.0
    c_sym = (c_averaged > 0.5).astype(int)
    c_sym = np.maximum(c_sym, c_sym.T)
    c_sym = np.maximum(c_sym, c_sym[::-1, :])
    c_sym = np.maximum(c_sym, c_sym[:, ::-1])
    return c_sym.tolist()


def _gen_rod(r):
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


def _gen_ring(inner_r, outer_r):
    """Annular dielectric ring."""
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


def _gen_cross_ring(arm_w, arm_l, inner_r, outer_r):
    """Cross shape with a hole in the center."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r2 = x*x + y*y
            in_cross = (abs(x) < arm_w/2 and abs(y) < arm_l/2) or \
                       (abs(y) < arm_w/2 and abs(x) < arm_l/2)
            in_hole = r2 < inner_r*inner_r
            in_outer = r2 < outer_r*outer_r
            if in_cross and not in_hole and in_outer:
                cell[i][j] = 1
    return cell


def _gen_square_ring(half_outer, half_inner):
    """Square annular ring."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            max_xy = max(abs(x), abs(y))
            if half_inner < max_xy < half_outer:
                cell[i][j] = 1
    return cell


def _gen_four_rods(offset, radius):
    """Four dielectric rods at (±offset, ±offset)."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx*offset
                    dy = y - sy*offset
                    if dx*dx + dy*dy < radius*radius:
                        cell[i][j] = 1
    return cell


def _gen_random_symmetric(density):
    size = 32
    cell = np.zeros((size, size), dtype=int)
    for i in range(size//2):
        for j in range(size//2):
            if np.random.random() < density:
                cell[i, j] = 1
                cell[size-1-i, j] = 1
                cell[i, size-1-j] = 1
                cell[size-1-i, size-1-j] = 1
    return cell.tolist()


def _mutate(cell, mutation_rate=0.1):
    size = len(cell)
    c = np.array(cell, dtype=int)
    for i in range(size//2):
        for j in range(size//2):
            if np.random.random() < mutation_rate:
                val = 1 - c[i, j]
                c[i, j] = val
                c[size-1-i, j] = val
                c[i, size-1-j] = val
                c[size-1-i, size-1-j] = val
    return c.tolist()


def _field_driven_optimize(cell, n_k=4, pw_radius=6, iterations=30):
    """Use field intensity to guide dielectric placement."""
    size = 32
    c = np.array(cell, dtype=float)
    
    inv_eps = 1.0 / (1.0 + 11.0 * c)
    fft_inv = np.fft.fft2(inv_eps) / (size * size)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    size_d = 4 * pw_radius + 1
    offset_d = 2 * pw_radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N] if N == size else fft_inv[dqx % size, dqy % size]
    
    # Use a few k-points to compute field distributions
    k_points = [(0.5, 0.0), (0.5, 0.5), (0.25, 0.0), (0.0, 0.0)]
    
    # Accumulate field intensities for band 4 and band 5
    field4 = np.zeros((size, size))
    field5 = np.zeros((size, size))
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        inv_eps_mat = inv_eps_full[
            Gx_diff.astype(int) + offset_d, 
            Gy_diff.astype(int) + offset_d
        ]
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals, eigvecs = eigh(M)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                # Band 4 is index 3, band 5 is index 4
                vec4 = eigvecs[:, 3]
                vec5 = eigvecs[:, 4]
                
                # Compute field intensity in real space
                # The field is sum of plane waves with coefficients from eigenvector
                for i in range(size):
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        # Sum over plane waves
                        f4 = 0.0
                        f5 = 0.0
                        for idx, (Gx, Gy) in enumerate(G_list):
                            phase4 = np.exp(2j * np.pi * ((k[0]+Gx)*x + (k[1]+Gy)*y))
                            phase5 = np.exp(2j * np.pi * ((k[0]+Gx)*x + (k[1]+Gy)*y))
                            f4 += vec4[idx] * phase4
                            f5 += vec5[idx] * phase5
                        field4[i, j] += np.abs(f4)**2
                        field5[i, j] += np.abs(f5)**2
        except:
            pass
    
    # Normalize
    if field4.max() > 0:
        field4 /= field4.max()
    if field5.max() > 0:
        field5 /= field5.max()
    
    # Update dielectric: place dielectric where field4 is high and field5 is low
    # This pushes band 4 down and band 5 up
    score_map = field4 - field5
    
    # Apply update with some smoothing
    c_new = c.copy()
    for i in range(size//2):
        for j in range(size//2):
            # Average over symmetric positions
            avg_score = (score_map[i, j] + score_map[size-1-i, j] + 
                        score_map[i, size-1-j] + score_map[size-1-i, size-1-j]) / 4.0
            if avg_score > 0.1:
                c_new[i, j] = 1
                c_new[size-1-i, j] = 1
                c_new[i, size-1-j] = 1
                c_new[size-1-i, size-1-j] = 1
            elif avg_score < -0.1:
                c_new[i, j] = 0
                c_new[size-1-i, j] = 0
                c_new[i, size-1-j] = 0
                c_new[size-1-i, size-1-j] = 0
    
    return (c_new > 0.5).astype(int).tolist()


def design(problem):
    start_time = time.time()
    deadline = start_time + 200.0
    
    size = problem["pixels"]
    assert size == 32
    
    candidates = []
    
    # Rods
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append(_gen_rod(r))
    
    # Squares
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(_gen_square(r))
    
    # Crosses
    for w in [0.05, 0.1, 0.15, 0.2]:
        for l in [0.3, 0.4, 0.45, 0.5]:
            candidates.append(_gen_cross(w, l))
    
    # Holes
    for r in [0.3, 0.4, 0.5]:
        candidates.append(_gen_hole(r))
    
    # Rings
    for inner, outer in [(0.1, 0.3), (0.15, 0.35), (0.2, 0.4), (0.1, 0.25), (0.15, 0.3)]:
        candidates.append(_gen_ring(inner, outer))
    
    # Cross rings
    for w, l, inner in [(0.1, 0.4, 0.1), (0.15, 0.45, 0.15), (0.2, 0.5, 0.2), (0.1, 0.35, 0.05)]:
        candidates.append(_gen_cross_ring(w, l, inner, 0.5))
    
    # Square rings
    for ho, hi in [(0.3, 0.15), (0.35, 0.2), (0.4, 0.25), (0.3, 0.1)]:
        candidates.append(_gen_square_ring(ho, hi))
    
    # Four rods
    for offset, radius in [(0.2, 0.08), (0.25, 0.1), (0.3, 0.12), (0.2, 0.1), (0.15, 0.06)]:
        candidates.append(_gen_four_rods(offset, radius))
    
    # Random symmetric patterns
    for density in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for _ in range(3):
            candidates.append(_gen_random_symmetric(density))
    
    # Evaluate all candidates
    best_cell = None
    best_score = -1.0
    
    for idx, cell in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        try:
            score = _compute_gap_fast(cell, n_k=6, pw_radius=8)
            if score > best_score:
                best_score = score
                best_cell = cell
        except:
            pass
    
    # Field-driven optimization on best candidates
    if best_cell is not None and time.time() < deadline - 60:
        # Try field-driven optimization starting from top candidates
        top_cells = [best_cell]
        # Add a few other good candidates
        scored = []
        for cell in candidates[:20]:
            try:
                s = _compute_gap_fast(cell, n_k=4, pw_radius=6)
                scored.append((s, cell))
            except:
                pass
        scored.sort(reverse=True)
        for s, cell in scored[:5]:
            if cell != best_cell:
                top_cells.append(cell)
        
        for base_cell in top_cells:
            if time.time() > deadline - 45:
                break
            try:
                optimized = _field_driven_optimize(base_cell, n_k=4, pw_radius=6, iterations=10)
                score = _compute_gap_fast(optimized, n_k=6, pw_radius=8)
                if score > best_score:
                    best_score = score
                    best_cell = optimized
            except:
                pass
    
    # Local search around the best
    if best_cell is not None and time.time() < deadline - 60:
        current = best_cell
        current_score = best_score
        for _ in range(150):
            if time.time() > deadline - 30:
                break
            mutated = _mutate(current, mutation_rate=0.05)
            try:
                score = _compute_gap_fast(mutated, n_k=6, pw_radius=8)
                if score > current_score:
                    current_score = score
                    current = mutated
                    best_score = score
                    best_cell = current
            except:
                pass
    
    # Final: try a few more random candidates if time allows
    if time.time() < deadline - 30:
        for _ in range(30):
            if time.time() > deadline - 15:
                break
            cell = _gen_random_symmetric(np.random.uniform(0.1, 0.45))
            try:
                score = _compute_gap_fast(cell, n_k=6, pw_radius=8)
                if score > best_score:
                    best_score = score
                    best_cell = cell
            except:
                pass
    
    if best_cell is None:
        best_cell = _gen_rod(0.2)
    
    return best_cell
