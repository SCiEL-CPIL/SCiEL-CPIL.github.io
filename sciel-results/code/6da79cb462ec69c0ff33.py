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


def _compute_gap_and_fields(cell, n_k=6, pw_radius=8):
    """Compute gap and return modal field intensities for bands 4 and 5."""
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
    
    max_w4 = 0.0
    min_w5 = float('inf')
    field4_intensity = np.zeros((N, N))
    field5_intensity = np.zeros((N, N))
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals, eigvecs = eigh(M)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
                
                # Get modal coefficients for bands 4 and 5
                c4 = eigvecs[:, 3]  # 4th band (index 3)
                c5 = eigvecs[:, 4]  # 5th band (index 4)
                
                # Reconstruct field in real space via inverse FFT of modal coefficients
                # The plane-wave coefficients map to G-vectors; we need to map back to pixel grid
                # For a rough intensity estimate, use the magnitude of coefficients weighted by G
                Gx_arr = G_arr[:, 0]
                Gy_arr = G_arr[:, 1]
                
                # Create a 2D array of coefficients on a grid centered at 0
                grid_size = 2 * pw_radius + 1
                coeff_grid4 = np.zeros((grid_size, grid_size), dtype=complex)
                coeff_grid5 = np.zeros((grid_size, grid_size), dtype=complex)
                
                for idx in range(n_pw):
                    gx = int(Gx_arr[idx])
                    gy = int(Gy_arr[idx])
                    ix = gx + pw_radius
                    iy = gy + pw_radius
                    if 0 <= ix < grid_size and 0 <= iy < grid_size:
                        coeff_grid4[iy, ix] = c4[idx]
                        coeff_grid5[iy, ix] = c5[idx]
                
                # Inverse FFT to get real-space field
                field4 = np.abs(np.fft.ifft2(coeff_grid4)) ** 2
                field5 = np.abs(np.fft.ifft2(coeff_grid5)) ** 2
                
                # Resize to N x N if needed
                if field4.shape != (N, N):
                    from scipy.ndimage import zoom
                    factor = N / grid_size
                    field4 = zoom(field4, factor, order=1)
                    field5 = zoom(field5, factor, order=1)
                
                field4_intensity += field4
                field5_intensity += field5
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0, field4_intensity, field5_intensity
    gap = 2 * (min_w5 - max_w4) / (min_w5 + max_w4)
    return gap, field4_intensity, field5_intensity


def _make_symmetric(cell):
    """Ensure cell is symmetric under transpose, row-reverse, col-reverse."""
    N = len(cell)
    c = np.array(cell, dtype=float)
    # Average over the 4 symmetry operations
    c_t = c.T
    c_r = c[::-1, :]
    c_c = c[:, ::-1]
    c_averaged = (c + c_t + c_r + c_c) / 4.0
    c_sym = (c_averaged > 0.5).astype(int)
    # Verify and enforce symmetry
    c_sym = np.maximum(c_sym, c_sym.T)
    c_sym = np.maximum(c_sym, c_sym[::-1, :])
    c_sym = np.maximum(c_sym, c_sym[:, ::-1])
    return c_sym.tolist()


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


def _gen_rod(r):
    """Circular dielectric rod of radius r."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < r*r:
                cell[i][j] = 1
    return cell


def _gen_random_symmetric(density):
    """Random symmetric pixel pattern."""
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


def _topology_update(cell, field4, field5, step_size=0.1, momentum=0.5):
    """
    Topology optimization update:
    - Where band 4 intensity is high, reduce dielectric (push ω4 down)
    - Where band 5 intensity is high, increase dielectric (push ω5 up)
    This is a heuristic gradient-like update for the gap objective.
    """
    N = len(cell)
    c = np.array(cell, dtype=float)
    
    # Normalize field intensities
    f4 = field4 / (np.max(field4) + 1e-12)
    f5 = field5 / (np.max(field5) + 1e-12)
    
    # Update rule: 
    # Increase ε where band 5 is strong (to push ω5 up)
    # Decrease ε where band 4 is strong (to push ω4 down)
    # Net effect: gap = ω5 - ω4 increases
    update = step_size * (f5 - f4)
    
    # Apply update with momentum
    c_new = c + momentum * update
    
    # Threshold to binary
    c_binary = (c_new > 0.5).astype(int)
    
    # Enforce symmetry
    c_binary = np.maximum(c_binary, c_binary.T)
    c_binary = np.maximum(c_binary, c_binary[::-1, :])
    c_binary = np.maximum(c_binary, c_binary[:, ::-1])
    
    return c_binary.tolist()


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
    
    m4 = np.max(w4_arr)
    soft_max_w4 = m4 + np.log(np.sum(np.exp(w4_arr - m4)))
    
    m5 = np.min(w5_arr)
    soft_min_w5 = m5 - np.log(np.sum(np.exp(-w5_arr + m5)))
    
    if soft_min_w5 <= 0 or soft_max_w4 <= 0:
        return -1.0
    return 2 * (soft_min_w5 - soft_max_w4) / (soft_min_w5 + soft_max_w4)


def design(problem):
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    assert size == 32
    
    # Phase 1: Generate diverse initial candidates and pick the best
    candidates = []
    
    # Rings of various sizes
    for inner in [0.1, 0.15, 0.2, 0.25]:
        for outer in [0.35, 0.4, 0.45, 0.5]:
            if outer > inner + 0.1:
                candidates.append(_gen_ring(inner, outer))
    
    # Rods
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(_gen_rod(r))
    
    # Random symmetric patterns at various densities
    for density in [0.2, 0.3, 0.35, 0.4, 0.45]:
        for _ in range(2):
            candidates.append(_gen_random_symmetric(density))
    
    # Evaluate candidates with fast evaluator
    best_cell = None
    best_score = -1.0
    
    for idx, cell in enumerate(candidates):
        if time.time() > deadline - 80:
            break
        try:
            score = _compute_gap_fast(cell, n_k=5, pw_radius=7)
            if score > best_score:
                best_score = score
                best_cell = cell
        except:
            pass
    
    if best_cell is None:
        best_cell = _gen_ring(0.2, 0.45)
        best_score = _compute_gap_fast(best_cell, n_k=5, pw_radius=7)
    
    # Phase 2: Topology optimization
    # Start from the best candidate and iteratively update using field-driven rule
    current = best_cell
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 25
    step_size = 0.15
    momentum = 0.6
    
    # Track history for momentum
    prev_update = np.zeros((32, 32))
    
    while time.time() < deadline - 30 and no_improve_count < max_no_improve:
        # Compute gap and modal fields
        try:
            gap, field4, field5 = _compute_gap_and_fields(current, n_k=5, pw_radius=7)
        except:
            no_improve_count += 1
            continue
        
        if gap <= 0:
            # If bands overlap, use a more aggressive update
            step_size = max(step_size * 1.2, 0.2)
        elif gap < current_score:
            # Regress, reduce step
            step_size = max(step_size * 0.8, 0.05)
        else:
            # Progress, maintain or slightly increase step
            step_size = min(step_size * 1.05, 0.3)
        
        # Compute update
        f4 = field4 / (np.max(field4) + 1e-12)
        f5 = field5 / (np.max(field5) + 1e-12)
        
        update = step_size * (f5 - f4)
        
        # Apply momentum
        prev_update = momentum * prev_update + (1 - momentum) * update
        
        c = np.array(current, dtype=float)
        c_new = c + prev_update
        c_binary = (c_new > 0.5).astype(int)
        
        # Enforce symmetry
        c_binary = np.maximum(c_binary, c_binary.T)
        c_binary = np.maximum(c_binary, c_binary[::-1, :])
        c_binary = np.maximum(c_binary, c_binary[:, ::-1])
        
        new_cell = c_binary.tolist()
        
        # Evaluate with fast evaluator
        try:
            new_score = _compute_gap_fast(new_cell, n_k=5, pw_radius=7)
        except:
            no_improve_count += 1
            continue
        
        if new_score > current_score:
            current = new_cell
            current_score = new_score
            best_score = max(best_score, current_score)
            best_cell = current
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If no improvement, try a small random perturbation to escape local minimum
            if no_improve_count % 5 == 0:
                # Random symmetric flip
                c = np.array(current, dtype=int)
                for _ in range(3):
                    i = np.random.randint(0, 16)
                    j = np.random.randint(0, 16)
                    val = 1 - c[i, j]
                    c[i, j] = val
                    c[31-i, j] = val
                    c[i, 31-j] = val
                    c[31-i, 31-j] = val
                current = c.tolist()
    
    # Phase 3: Local search polish with random mutations
    if time.time() < deadline - 20:
        polish_no_improve = 0
        max_polish_no_improve = 15
        
        while time.time() < deadline - 10 and polish_no_improve < max_polish_no_improve:
            # Random symmetric mutation
            c = np.array(best_cell, dtype=int)
            for _ in range(2):
                i = np.random.randint(0, 16)
                j = np.random.randint(0, 16)
                if np.random.random() < 0.5:
                    val = 1 - c[i, j]
                    c[i, j] = val
                    c[31-i, j] = val
                    c[i, 31-j] = val
                    c[31-i, 31-j] = val
            
            mutated = c.tolist()
            
            try:
                score = _compute_gap_fast(mutated, n_k=6, pw_radius=8)
                if score > best_score:
                    best_score = score
                    best_cell = mutated
                    polish_no_improve = 0
                else:
                    polish_no_improve += 1
            except:
                polish_no_improve += 1
    
    # Final verification with higher accuracy
    if time.time() < deadline - 5:
        try:
            final_score = _compute_gap_fast(best_cell, n_k=8, pw_radius=10)
            if final_score > best_score:
                best_score = final_score
        except:
            pass
    
    return best_cell
