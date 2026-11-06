"""Photonic crystal unit cell design for large TM 4-5 band gap."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing TM 4-5 band gap.

    Hybrid: diverse geometric seeds + stochastic orbit flips + smooth
    surrogate (soft-min/soft-max) gradient polish + final exact orbit polish.
    """
    t_start = time.time()
    DEADLINE = 215.0  # safety margin before 230s kill

    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]

    # --- Fast TM plane-wave solver (17x17 = 289 plane waves) ---
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289

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
    n_k = len(k_points)

    # Precompute (k+G)^2 for all k-points
    k2_all = np.zeros((n_k, n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2

    # Sparse coupling structure: for each plane-wave index i, list of (j, diff_coeff_idx)
    # where diff_coeff_idx = index of (Gx[i]-Gx[j], Gy[i]-Gy[j]) in the coefficient array.
    # The coefficient array is indexed by (gx+half)*n_pw + (gy+half) for gx,gy in [-half,half].
    # This is exactly the same indexing as the G array itself (since Gx[idx]=gx, Gy[idx]=gy).
    # So diff_coeff_idx is just the index into the coeffs array for the difference vector.
    # Precompute: for each i, the list of (j, d_idx) where d_idx is valid.
    # d_idx = (dGx + half) * n_pw + (dGy + half)
    # Note: d_idx is an index into the same G-indexing as coeffs.
    # H[i,j] = k2[i] * coeffs[d_idx]  (for valid d_idx)
    # The matrix is Hermitian: H[i,j] = conj(H[j,i]) since coeffs[-d] = conj(coeffs[d]).

    # For efficiency, precompute for each k-point the non-zero entries.
    # But since k2 varies per k-point, we store the (i,j,d_idx) triples once.
    # Then H = sum over valid (i,j,d_idx): k2[i] * coeffs[d_idx] * E[i,j]
    # We can build H as: H[i,j] = k2[i] * coeffs[d_idx(i,j)] for valid pairs.

    # Precompute valid (i, j, d_idx) triples
    valid_triples = []  # list of (i, j, d_idx)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = int(Gx[i]) - int(Gx[j])
            dGy = int(Gy[i]) - int(Gy[j])
            idx_x = dGx + half
            idx_y = dGy + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                d_idx = idx_x * n_pw + idx_y
                valid_triples.append((i, j, d_idx))

    valid_triples = np.array(valid_triples)  # shape (N_valid, 3)
    v_i = valid_triples[:, 0]
    v_j = valid_triples[:, 1]
    v_d = valid_triples[:, 2]

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

    def build_H(coeffs, k_idx):
        """Build the 289x289 Hermitian matrix for a given k-point and coeffs."""
        k2 = k2_all[k_idx]
        H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
        # H[i,j] = k2[i] * coeffs[d_idx(i,j)]
        H[v_i, v_j] = k2[v_i] * coeffs[v_d]
        # Enforce Hermiticity (numerical safety)
        H = (H + H.conj().T) / 2.0
        return H

    def compute_band_ratio_full(coeffs):
        """Compute gap-midgap ratio using all 30 k-points and 289 PWs."""
        max_band4 = -np.inf
        min_band5 = np.inf

        for k_idx in range(n_k):
            H = build_H(coeffs, k_idx)
            eigvals = np.linalg.eigvalsh(H)
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
            H = build_H(coeffs, k_idx)
            eigvals = np.linalg.eigvalsh(H)
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
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_four_rods(radius, offset):
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

    def generate_moire(f1, f2, phase1, phase2, threshold):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                v1 = np.sin(2 * np.pi * f1 * (x + y) + phase1)
                sq1 = 1.0 if v1 > 0 else 0.0
                v2 = np.sin(2 * np.pi * f2 * (x - y) + phase2)
                sq2 = 1.0 if v2 > 0 else 0.0
                val = sq1 * sq2
                if val >= threshold:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_moire_sin(f1, f2, phase1, phase2, threshold):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                v1 = np.sin(2 * np.pi * f1 * (x + y) + phase1)
                v2 = np.sin(2 * np.pi * f2 * (x - y) + phase2)
                val = v1 * v2
                if val >= threshold:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_concentric_rings(n_rings, thickness, phase):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x * x + y * y)
                ring_idx = int((r / (0.5 / n_rings) + phase) % n_rings)
                if ring_idx < n_rings // 2:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    # New generators
    def generate_diamond(radius):
        """Diamond (L1 ball) centered at origin."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x) + abs(y) < radius:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_offset_rods(radius, offset_x, offset_y):
        """Four rods at (±offset_x, ±offset_y) for rectangular offset."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        dx = x - sx * offset_x
                        dy = y - sy * offset_y
                        if dx * dx + dy * dy < radius * radius:
                            cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_checkerboard(block_size, phase):
        """Checkerboard pattern with given block size."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                bi = (i // block_size + phase) % 2
                bj = (j // block_size) % 2
                if bi != bj:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_chevron(freq, thickness, phase):
        """Chevron/V-shaped pattern."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                v = np.sin(2 * np.pi * freq * (abs(x) + y) + phase)
                if v > 0 and abs(v) < thickness:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    # --- Mutation for stochastic search ---
    def mutate_cell(cell, n_mutations=3):
        cell = cell.copy()
        for _ in range(n_mutations):
            i = random.randint(0, size // 2 - 1)
            j = random.randint(0, size // 2 - 1)
            orbit = [(i, j), (i, size - 1 - j), (size - 1 - i, j),
                     (size - 1 - i, size - 1 - j),
                     (j, i), (j, size - 1 - i), (size - 1 - i, j),
                     (size - 1 - i, size - 1 - i)]
            new_val = 1 - cell[i, j]
            for (oi, oj) in orbit:
                cell[oi, oj] = new_val
        return cell

    # --- Smooth surrogate: soft-min/soft-max band objective ---
    def smooth_band_objective(coeffs, beta=5.0):
        """
        Compute a smooth surrogate for the gap-midgap ratio using soft-min/soft-max.
        Returns (surrogate_value, eigvals_per_k) for gradient computation.
        """
        eigvals_all = []
        for k_idx in range(n_k):
            H = build_H(coeffs, k_idx)
            eigvals = np.linalg.eigvalsh(H)
            eigvals_all.append(eigvals)

        # For each k, band4 = eigvals[3], band5 = eigvals[4]
        # We want: maximize (soft_min(band5) - soft_max(band4)) / soft_midgap
        # soft_max of a set {a_i} = (1/beta) * log(sum(exp(beta * a_i)))
        # soft_min of a set {a_i} = -(1/beta) * log(sum(exp(-beta * a_i)))
        band4_vals = np.array([ev[3] for ev in eigvals_all])
        band5_vals = np.array([ev[4] for ev in eigvals_all])

        # Soft max of band4 (across k-points)
        beta_safe = max(beta, 1e-6)
        logsumexp_b4 = np.log(np.sum(np.exp(beta_safe * band4_vals))) / beta_safe
        soft_max_b4 = logsumexp_b4

        # Soft min of band5 (across k-points)
        logsumexp_b5 = np.log(np.sum(np.exp(-beta_safe * band5_vals))) / beta_safe
        soft_min_b5 = -logsumexp_b5

        gap = soft_min_b5 - soft_max_b4
        midgap = (soft_min_b5 + soft_max_b4) / 2.0
        if midgap > 1e-10:
            return gap / midgap, eigvals_all
        return 0.0, eigvals_all

    def smooth_surrogate_polish(cell, max_iters=50, beta=5.0):
        """
        Polish a cell using gradient ascent on a continuous field.
        We parameterize the cell as a continuous field f(x,y) in [0,1],
        binarize at 0.5, and use the smooth surrogate as objective.
        Gradient is estimated by finite differences on the binarized cell.
        """
        current = cell.copy().astype(float)
        current_score = evaluate_cell_fast(cell)
        best_local = cell.copy()
        best_local_score = current_score

        # Symmetry orbits: each orbit is a set of pixels that must be equal.
        # We'll work with the independent quadrant (top-left 16x16) and mirror.
        # For gradient estimation, we flip one orbit at a time and measure change.

        # Precompute all unique orbits in the top-left quadrant
        orbits = []
        seen = set()
        for i in range(size // 2):
            for j in range(size // 2):
                # Orbit under D4 symmetry
                orbit = [(i, j), (i, size - 1 - j), (size - 1 - i, j),
                         (size - 1 - i, size - 1 - j),
                         (j, i), (j, size - 1 - i), (size - 1 - i, j),
                         (size - 1 - i, size - 1 - i)]
                # Deduplicate
                orbit = list(set(orbit))
                key = tuple(sorted(orbit))
                if key not in seen:
                    seen.add(key)
                    orbits.append(orbit)

        random.shuffle(orbits)

        for it in range(max_iters):
            if time.time() > DEADLINE - 10:
                break
            improved = False
            for orbit in orbits:
                if time.time() > DEADLINE - 10:
                    break
                # Try flipping this orbit
                test = current.copy()
                val = 1 - int(round(test[orbit[0][0], orbit[0][1]]))
                for (oi, oj) in orbit:
                    test[oi, oj] = val
                test_int = (test > 0.5).astype(int)
                test_int = enforce_symmetry(test_int)
                test_score = evaluate_cell_fast(test_int)
                if test_score > current_score + 1e-10:
                    current = test
                    current = (current > 0.5).astype(float)
                    current_score = test_score
                    if current_score > best_local_score:
                        best_local = (current > 0.5).astype(int)
                        best_local_score = current_score
                    improved = True

            if not improved:
                break

        return best_local, best_local_score

    # --- Main search ---
    best_score = -np.inf
    best_cell = None

    def update_best(cell, score):
        nonlocal best_score, best_cell
        if score > best_score:
            best_score = score
            best_cell = cell.copy()

    print(f"Phase 1: Diverse candidate generation...")
    candidates = []

    # Single rods
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        if time.time() > DEADLINE:
            break
        candidates.append(generate_rod(r))

    # Four-rod patterns (square offset)
    for r in [0.06, 0.08, 0.10, 0.12, 0.14]:
        if time.time() > DEADLINE:
            break
        for off in [0.20, 0.22, 0.25, 0.28, 0.30]:
            if time.time() > DEADLINE:
                break
            candidates.append(generate_four_rods(r, off))

    # Offset rods (rectangular)
    for r in [0.08, 0.10, 0.12]:
        if time.time() > DEADLINE:
            break
        for ox in [0.15, 0.20, 0.25]:
            if time.time() > DEADLINE:
                break
            for oy in [0.15, 0.20, 0.25]:
                if time.time() > DEADLINE:
                    break
                if ox != oy:
                    candidates.append(generate_offset_rods(r, ox, oy))

    # Diamonds
    for r in [0.20, 0.25, 0.30, 0.35]:
        if time.time() > DEADLINE:
            break
        candidates.append(generate_diamond(r))

    # Rings
    for ri in [0.05, 0.08, 0.10, 0.12]:
        if time.time() > DEADLINE:
            break
        for ro in [0.15, 0.18, 0.20, 0.22, 0.25]:
            if time.time() > DEADLINE:
                break
            candidates.append(generate_ring(ri, ro))

    # Crosses
    for aw in [0.05, 0.08, 0.10]:
        if time.time() > DEADLINE:
            break
        for al in [0.30, 0.35, 0.40]:
            if time.time() > DEADLINE:
                break
            candidates.append(generate_cross(aw, al))

    # Four rings
    for ri in [0.03, 0.04, 0.05]:
        if time.time() > DEADLINE:
            break
        for ro in [0.08, 0.10, 0.12]:
            if time.time() > DEADLINE:
                break
            for off in [0.20, 0.25, 0.30]:
                if time.time() > DEADLINE:
                    break
                candidates.append(generate_four_rings(ri, ro, off))

    # Moiré patterns (square wave)
    for f1 in [3, 4, 5, 6]:
        if time.time() > DEADLINE:
            break
        for f2 in [3, 4, 5, 6]:
            if time.time() > DEADLINE:
                break
            if f1 == f2:
                continue
            for p1 in [0, np.pi / 2]:
                if time.time() > DEADLINE:
                    break
                for p2 in [0, np.pi / 2]:
                    if time.time() > DEADLINE:
                        break
                    for thr in [0.5, 0.7]:
                        if time.time() > DEADLINE:
                            break
                        candidates.append(generate_moire(f1, f2, p1, p2, thr))

    # Moiré patterns (sine product)
    for f1 in [3, 4, 5]:
        if time.time() > DEADLINE:
            break
        for f2 in [3, 4, 5]:
            if time.time() > DEADLINE:
                break
            if f1 == f2:
                continue
            for p1 in [0, np.pi / 2]:
                if time.time() > DEADLINE:
                    break
                for p2 in [0, np.pi / 2]:
                    if time.time() > DEADLINE:
                        break
                    for thr in [0.3, 0.5, 0.7]:
                        if time.time() > DEADLINE:
                            break
                        candidates.append(generate_moire_sin(f1, f2, p1, p2, thr))

    # Concentric rings
    for n_r in [2, 3, 4]:
        if time.time() > DEADLINE:
            break
        for phase in [0, 0.5]:
            if time.time() > DEADLINE:
                break
            candidates.append(generate_concentric_rings(n_r, 0.5 / n_r, phase))

    # Checkerboards
    for bs in [4, 6, 8]:
        if time.time() > DEADLINE:
            break
        for ph in [0, 1]:
            if time.time() > DEADLINE:
                break
            candidates.append(generate_checkerboard(bs, ph))

    # Chevrons
    for fr in [2, 3, 4]:
        if time.time() > DEADLINE:
            break
        for th in [0.3, 0.5, 0.7]:
            if time.time() > DEADLINE:
                break
            for ph in [0, np.pi / 2]:
                if time.time() > DEADLINE:
                    break
                candidates.append(generate_chevron(fr, th, ph))

    # Evaluate all candidates with fast evaluator first
    print(f"Evaluating {len(candidates)} candidates...")
    fast_scores = []
    for cell in candidates:
        if time.time() > DEADLINE:
            break
        score = evaluate_cell_fast(cell)
        fast_scores.append((score, cell))

    # Sort by fast score and evaluate top candidates with full evaluator
    fast_scores.sort(key=lambda x: -x[0])
    top_n = min(30, len(fast_scores))
    for score, cell in fast_scores[:top_n]:
        if time.time() > DEADLINE:
            break
        full_score = evaluate_cell(cell)
        update_best(cell, full_score)

    print(f"Best after Phase 1: {best_score:.4f}")

    # Phase 2: Stochastic search around best
    if best_cell is not None and time.time() < DEADLINE - 40:
        print(f"Phase 2: Stochastic search...")
        current = best_cell.copy()
        current_score = best_score

        for iteration in range(800):
            if time.time() > DEADLINE:
                break
            mutated = mutate_cell(current, n_mutations=random.randint(1, 4))
            score = evaluate_cell_fast(mutated)
            if score > current_score * 0.95:
                full_score = evaluate_cell(mutated)
                if full_score > current_score:
                    current = mutated
                    current_score = full_score
                    update_best(current, current_score)

    # Phase 3: Smooth surrogate polish
    if best_cell is not None and time.time() < DEADLINE - 20:
        print(f"Phase 3: Smooth surrogate polish...")
        polished, polished_score = smooth_surrogate_polish(best_cell, max_iters=80, beta=5.0)
        full_score = evaluate_cell(polished)
        if full_score > best_score:
            update_best(polished, full_score)
        print(f"After polish: {best_score:.4f}")

    # Phase 4: Final single-orbit exact polish
    if best_cell is not None and time.time() < DEADLINE - 5:
        print(f"Phase 4: Final exact orbit polish...")
        current = best_cell.copy()
        current_score = best_score

        # Iterate over all orbits, try flipping each
        orbits = []
        seen = set()
        for i in range(size // 2):
            for j in range(size // 2):
                orbit = [(i, j), (i, size - 1 - j), (size - 1 - i, j),
                         (size - 1 - i, size - 1 - j),
                         (j, i), (j, size - 1 - i), (size - 1 - i, j),
                         (size - 1 - i, size - 1 - i)]
                orbit = list(set(orbit))
                key = tuple(sorted(orbit))
                if key not in seen:
                    seen.add(key)
                    orbits.append(orbit)

        random.shuffle(orbits)
        for orbit in orbits:
            if time.time() > DEADLINE:
                break
            test = current.copy()
            val = 1 - test[orbit[0][0], orbit[0][1]]
            for (oi, oj) in orbit:
                test[oi, oj] = val
            test = enforce_symmetry(test)
            test_score = evaluate_cell(test)
            if test_score > current_score + 1e-10:
                current = test
                current_score = test_score
                update_best(current, current_score)

        print(f"Final best: {best_score:.4f}")

    # Fallback
    if best_cell is None:
        best_cell = generate_four_rods(0.10, 0.25)
        best_score = evaluate_cell(best_cell)

    best_cell = enforce_symmetry(best_cell)
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
