"""Design a 32x32 photonic crystal unit cell to maximize TM band 4-5 gap."""
import numpy as np
import time

def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12

    t_start = time.time()
    DEADLINE = 218.0  # safety margin before 230s kill

    def time_left():
        return DEADLINE - (time.time() - t_start)

    # --- Fast TM plane-wave solver (17x17 = 289 plane waves) ---

    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289

    # K-points in irreducible BZ (30 points)
    pi = np.pi
    k_points = []
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)

    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2

    # Precompute difference indices for Fourier coupling
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = int(Gx[i]) - int(Gx[j])
            dGy = int(Gy[i]) - int(Gy[j])
            idx_x = dGx + half
            idx_y = dGy + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
            else:
                diff_idx[i, j] = -1
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)

    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs

    def compute_band_ratio_full(coeffs):
        """Compute gap-midgap ratio using all 30 k-points and 289 PWs."""
        max_band4 = -np.inf
        min_band5 = np.inf

        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])

        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def compute_band_ratio_fast(coeffs):
        """Fast approximate evaluation using subset of k-points."""
        k_subset = [0, 4, 9, 14, 19, 24, 29]
        max_band4 = -np.inf
        min_band5 = np.inf

        for k_idx in k_subset:
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])

        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def evaluate_cell(cell):
        """Full evaluation of a cell."""
        coeffs = get_fourier_coeffs(cell)
        return compute_band_ratio_full(coeffs)

    def evaluate_cell_fast(cell):
        """Fast evaluation for search guidance."""
        coeffs = get_fourier_coeffs(cell)
        return compute_band_ratio_fast(coeffs)

    # --- Symmetry enforcement ---

    def enforce_symmetry(cell):
        """Enforce square symmetry by averaging over the group."""
        cell = np.array(cell, dtype=int)
        cell_rot = cell[::-1, ::-1].copy()
        cell_hflip = cell[:, ::-1].copy()
        cell_vflip = cell[::-1, :].copy()
        result = (cell + cell_rot + cell_hflip + cell_vflip)
        result = (result >= 2).astype(int)
        return result

    # --- Candidate generation ---

    def generate_rod(radius):
        """Single centered rod."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_four_rods(radius, offset):
        """Four rods at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        dx = x - sx * offset
                        dy = y - sy * offset
                        if dx * dx + dy * dy < radius * radius:
                            cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_ring(inner_r, outer_r):
        """Dielectric ring."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r2 = x * x + y * y
                if inner_r * inner_r < r2 < outer_r * outer_r:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_cross(arm_width, arm_length):
        """Cross/plus shape."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(y) < arm_width / 2 and abs(x) < arm_length / 2:
                    cell[i, j] = 1
                if abs(x) < arm_width / 2 and abs(y) < arm_length / 2:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_four_rings(inner_r, outer_r, offset):
        """Four small rings at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        dx = x - sx * offset
                        dy = y - sy * offset
                        r2 = dx * dx + dy * dy
                        if inner_r * inner_r < r2 < outer_r * outer_r:
                            cell[i, j] = 1
        return enforce_symmetry(cell)

    # --- Initial candidates ---
    candidates = []
    
    # Rods
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(generate_rod(r))
    
    # Four rods
    for r in [0.05, 0.08, 0.1]:
        for off in [0.1, 0.15, 0.2]:
            candidates.append(generate_four_rods(r, off))
    
    # Rings
    for inner in [0.05, 0.1]:
        for outer in [0.2, 0.25, 0.3]:
            if outer > inner:
                candidates.append(generate_ring(inner, outer))
    
    # Crosses
    for w in [0.05, 0.1, 0.15]:
        for l in [0.2, 0.3, 0.4]:
            candidates.append(generate_cross(w, l))
    
    # Four rings
    for inner in [0.02, 0.04]:
        for outer in [0.1, 0.15]:
            for off in [0.1, 0.15]:
                candidates.append(generate_four_rings(inner, outer, off))

    # Evaluate initial candidates
    best_cell = None
    best_score = -float('inf')

    for cand in candidates:
        if time_left() < 15:
            break
        score = evaluate_cell_fast(cand)
        if score > best_score:
            best_score = score
            best_cell = cand.copy()

    # Refine best candidate with full evaluation
    if best_cell is not None:
        full_score = evaluate_cell(best_cell)
        if full_score > best_score:
            best_score = full_score

    # --- Direct pixel optimization (hill climbing) ---
    if best_cell is not None and time_left() > 30:
        current_cell = best_cell.copy()
        current_score = best_score
        rng = np.random.RandomState(42)
        
        # Pixel indices
        pixel_indices = np.arange(size * size)
        
        # Hill climbing: flip random pixels
        max_iterations = int(min(time_left() / 0.5, 200))
        
        for it in range(max_iterations):
            if time_left() < 10:
                break
            
            # Flip a random pixel
            idx = rng.choice(pixel_indices)
            i, j = divmod(idx, size)
            
            # Check symmetry constraints: only flip if it's a symmetry-allowed operation
            # For D4 symmetry, flipping (i,j) requires flipping (size-1-i, j), (i, size-1-j), 
            # (size-1-i, size-1-j), (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
            
            new_cell = current_cell.copy()
            symmetry_pixels = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (j, size-1-i),
                (size-1-j, i),
                (size-1-j, size-1-i)
            ]
            
            # Only flip if all symmetry-related pixels have the same value
            current_val = current_cell[i, j]
            can_flip = True
            for si, sj in symmetry_pixels:
                if current_cell[si, sj] != current_val:
                    can_flip = False
                    break
            
            if not can_flip:
                continue
            
            # Flip all symmetry-related pixels
            for si, sj in symmetry_pixels:
                new_cell[si, sj] = 1 - current_val
            
            new_score = evaluate_cell_fast(new_cell)
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_cell = new_cell.copy()
                # Accept and continue
            else:
                # Reject, keep current
                pass

    # Final full evaluation
    if best_cell is not None:
        final_score = evaluate_cell(best_cell)
        if final_score > best_score:
            best_score = final_score

    # Fallback
    if best_cell is None:
        best_cell = generate_rod(0.2)

    return best_cell.tolist()
