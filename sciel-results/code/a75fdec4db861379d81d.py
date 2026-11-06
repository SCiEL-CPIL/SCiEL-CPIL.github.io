"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """Optimize 32x32 dielectric pattern for max TM gap between bands 4 and 5."""
    size = problem["pixels"]          # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    start_time = time.time()
    time_limit = 210.0                # safety margin before 230s kill

    # ---------------- Plane-wave setup (17x17 = 289) ----------------
    n_pw = 17
    half = n_pw // 2                  # 8
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw          # 289

    # ---------------- k-points in irreducible BZ ----------------
    pi = np.pi
    k_points = []
    for i in range(10):               # X->Gamma edge  (t*pi, 0)
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):               # X->M edge     (pi, t*pi)
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):               # M->Gamma      (pi(1-t), pi(1-t))
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)
    n_k = len(k_points)

    # (k+G)^2 per k-point
    k2_all = np.zeros((n_k, n_pw_total))
    for i in range(n_k):
        kx, ky = k_points[i]
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2

    # difference indices: H[i,j] uses coeff[Gx[i]-Gx[j], Gy[i]-Gy[j]]
    diff_idx = np.full((n_pw_total, n_pw_total), -1, dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            ix = int(dGx) + half
            iy = int(dGy) + half
            if 0 <= ix < n_pw and 0 <= iy < n_pw:
                diff_idx[i, j] = ix * n_pw + iy
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)
    valid_diff = diff_idx[valid_rows, valid_cols]

    # ---------------- Symmetry: work on 8x8 quadrant ----------------
    qsize = size // 4                 # 8

    def expand_quadrant(q):
        """Expand 8x8 top-left quadrant to 32x32 with full square symmetry."""
        q = np.array(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        full = np.vstack([top, top[::-1, :]])
        return full

    def quadrant_of(cell):
        return np.array(cell)[:qsize, :qsize]

    # ---------------- Fourier coefficients of 1/eps ----------------
    def compute_coeffs(cell):
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            coeffs[idx] = fft_2d[gx % size, gy % size] / (size * size)
        return coeffs

    def update_coeffs(coeffs, cell, i, j, new_val):
        """Incrementally update coeffs after setting cell[i,j]=new_val."""
        old_val = cell[i, j]
        if old_val == new_val:
            return coeffs
        delta_eps = (eps_high if new_val == 1 else eps_low) - \
                    (eps_high if old_val == 1 else eps_low)
        if delta_eps == 0:
            return coeffs
        inv_new = 1.0 / (eps_high if new_val == 1 else eps_low)
        inv_old = 1.0 / (eps_high if old_val == 1 else eps_low)
        d_inv = inv_new - inv_old
        # pixel (i,j) contributes d_inv * exp(-2pi i (gx*i + gy*j)/size)
        fx = np.exp(-2j * pi * i / size) ** Gx   # = exp(-2pi i Gx i / size)
        fy = np.exp(-2j * pi * j / size) ** Gy
        delta = d_inv * np.outer(fx, fy) / (size * size)
        coeffs = coeffs + delta.ravel()
        return coeffs

    # ---------------- Band ratio ----------------
    def compute_band_ratio(coeffs):
        max_band4 = -np.inf
        min_band5 = np.inf
        for k_idx in range(n_k):
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[valid_diff]
            H = (H + H.conj().T) * 0.5
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

    # ---------------- Candidate generators (all return 8x8 quadrant) ----------------
    def q_circle(radius):
        q = np.zeros((qsize, qsize), dtype=int)
        for i in range(qsize):
            for j in range(qsize):
                # map quadrant pixel to full-cell coordinate
                x = (4 * i + 0.5) / size - 0.5
                y = (4 * j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    q[i, j] = 1
        return q

    def q_square(half_size):
        q = np.zeros((qsize, qsize), dtype=int)
        h = half_size * size
        for i in range(qsize):
            for j in range(qsize):
                x = (4 * i + 0.5) / size - 0.5
                y = (4 * j + 0.5) / size - 0.5
                if abs(x) < h / size and abs(y) < h / size:
                    q[i, j] = 1
        return q

    def q_diamond(radius):
        q = np.zeros((qsize, qsize), dtype=int)
        for i in range(qsize):
            for j in range(qsize):
                x = (4 * i + 0.5) / size - 0.5
                y = (4 * j + 0.5) / size - 0.5
                if abs(x) + abs(y) < radius:
                    q[i, j] = 1
        return q

    def q_ring(inner_r, outer_r):
        q = np.zeros((qsize, qsize), dtype=int)
        for i in range(qsize):
            for j in range(qsize):
                x = (4 * i + 0.5) / size - 0.5
                y = (4 * j + 0.5) / size - 0.5
                r = np.sqrt(x * x + y * y)
                if inner_r < r < outer_r:
                    q[i, j] = 1
        return q

    def q_inverse_rod(hole_radius):
        q = np.ones((qsize, qsize), dtype=int)
        for i in range(qsize):
            for j in range(qsize):
                x = (4 * i + 0.5) / size - 0.5
                y = (4 * j + 0.5) / size - 0.5
                if x * x + y * y < hole_radius ** 2:
                    q[i, j] = 0
        return q

    def q_cross(arm_width):
        q = np.zeros((qsize, qsize), dtype=int)
        c = size // 2
        for i in range(qsize):
            for j in range(qsize):
                # full-cell row/col
                fi = 4 * i
                fj = 4 * j
                # horizontal bar: rows near center
                if abs(fi - c) < arm_width // 2 + 1:
                    q[i, j] = 1
                # vertical bar: cols near center
                if abs(fj - c) < arm_width // 2 + 1:
                    q[i, j] = 1
        return q

    def q_four_rods(offset, rod_r):
        q = np.zeros((qsize, qsize), dtype=int)
        for i in range(qsize):
            for j in range(qsize):
                x = (4 * i + 0.5) / size - 0.5
                y = (4 * j + 0.5) / size - 0.5
                for px, py in [(offset, offset), (-offset, offset),
                               (offset, -offset), (-offset, -offset)]:
                    if (x - px) ** 2 + (y - py) ** 2 < rod_r ** 2:
                        q[i, j] = 1
                        break
        return q

    # ---------------- Build candidate list ----------------
    candidates = []
    for r in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        candidates.append(q_circle(r))
    for h in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(q_square(h))
    for r in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(q_diamond(r))
    for ir, orr in [(0.1, 0.2), (0.15, 0.25), (0.1, 0.25), (0.2, 0.3)]:
        candidates.append(q_ring(ir, orr))
    for hr in [0.1, 0.15, 0.2]:
        candidates.append(q_inverse_rod(hr))
    for aw in [3, 4, 5, 6]:
        candidates.append(q_cross(aw))
    for off in [0.2, 0.25, 0.3]:
        for rr in [0.08, 0.1, 0.12]:
            candidates.append(q_four_rods(off, rr))

    # ---------------- Evaluate candidates ----------------
    best_ratio = -np.inf
    best_cell = None
    for q in candidates:
        if time.time() - start_time > time_limit * 0.25:
            break
        cell = expand_quadrant(q)
        coeffs = compute_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell

    if best_cell is None:
        best_cell = expand_quadrant(q_circle(0.25))
        best_ratio = compute_band_ratio(compute_coeffs(best_cell))

    # ---------------- Local search on quadrant ----------------
    current_cell = best_cell.copy()
    current_q = quadrant_of(current_cell)
    current_ratio = best_ratio
    current_coeffs = compute_coeffs(current_cell)

    indep_pixels = [(i, j) for i in range(qsize) for j in range(qsize)]
    no_improve = 0
    max_no_improve = 40
    threshold = 1e-8

    while time.time() - start_time < time_limit * 0.9 and no_improve < max_no_improve:
        improved = False
        np.random.shuffle(indep_pixels)
        for (i, j) in indep_pixels:
            if time.time() - start_time > time_limit * 0.9:
                break
            old_val = current_q[i, j]
            new_val = 1 - old_val
            # flip in quadrant, expand, update coeffs incrementally
            current_q[i, j] = new_val
            new_cell = expand_quadrant(current_q)
            new_coeffs = compute_coeffs(new_cell)  # full recompute for correctness
            new_ratio = compute_band_ratio(new_coeffs)
            if new_ratio > current_ratio + threshold:
                current_ratio = new_ratio
                current_cell = new_cell
                current_coeffs = new_coeffs
                improved = True
                no_improve = 0
            else:
                current_q[i, j] = old_val
                no_improve += 1
        if not improved:
            no_improve += 1

    return current_cell.tolist()
# EVOLVE-BLOCK-END
