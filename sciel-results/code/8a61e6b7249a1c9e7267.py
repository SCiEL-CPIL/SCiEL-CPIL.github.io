"""Combined program: 4-hole dielectric design family + symmetric-quadrant local search."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    size = problem["pixels"]
    eps_high = float(problem.get("eps_high", 12.0))
    eps_low = float(problem.get("eps_low", 1.0))
    deadline = time.time() + 210.0  # safety margin before 230s wall-clock limit

    # ---------- Precompute plane-wave and k-point data ----------
    N = 17
    half = N // 2  # 8
    G_list = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            if gx * gx + gy * gy <= (half + 0.5) ** 2:  # |G| <= 8.5
                G_list.append((gx, gy))
    G_arr = np.array(G_list, dtype=float)
    n_pw = len(G_arr)

    # 30 k-points in irreducible BZ (in units of 2*pi/a)
    k_points = []
    for i in range(11):
        k_points.append((i * 0.5 / 10.0, 0.0))
    for i in range(1, 11):
        k_points.append((0.5, i * 0.5 / 10.0))
    for i in range(1, 11):
        t = i / 10.0
        k_points.append((0.5 * (1 - t), 0.5 * (1 - t)))
    unique_k = []
    for k in k_points:
        if not any(abs(k[0] - uk[0]) < 1e-9 and abs(k[1] - uk[1]) < 1e-9 for uk in unique_k):
            unique_k.append(k)
    k_points = np.array(unique_k[:30], dtype=float)

    # Precompute (k+G)^2 for each k-point (in units of (2*pi)^2)
    KG2 = []
    for kx, ky in k_points:
        kgx = kx + G_arr[:, 0]
        kgy = ky + G_arr[:, 1]
        KG2.append(kgx * kgx + kgy * kgy)
    KG2 = np.array(KG2)  # shape (30, n_pw)

    # ---------- Evaluator ----------
    def score_quad(q):
        """q is a flat 64-element int array (8x8 symmetric quadrant). Returns gap-midgap ratio."""
        # Build full 32x32 cell from symmetric quadrant
        cell = np.zeros((size, size), dtype=np.int8)
        q2d = q.reshape(8, 8)
        cell[:8, :8] = q2d
        cell[:8, 24:32] = q2d[:, ::-1]
        cell[24:32, :8] = q2d[:, ::-1]
        cell[24:32, 24:32] = q2d

        # Build 1/eps field
        inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)

        # FFT to get Fourier coefficients of 1/eps
        fft = np.fft.fft2(inv_eps)
        c = fft / (size * size)

        # Build the 17x17 coefficient matrix C(G_i - G_j)
        gx_diff = (G_arr[:, 0][None, :] - G_arr[:, 0][:, None]).astype(int) % size
        gy_diff = (G_arr[:, 1][None, :] - G_arr[:, 1][:, None]).astype(int) % size
        C_mat = c[gy_diff, gx_diff]

        max_w4 = 0.0
        min_w5 = float('inf')

        for ki in range(len(k_points)):
            kg2 = KG2[ki]
            # M_{ij} = C(G_i - G_j) * (k + G_j)^2
            M = C_mat * kg2[None, :]
            eigvals = np.linalg.eigvals(M)
            eigvals = np.sort(np.real(eigvals))
            if len(eigvals) >= 5:
                w4 = math.sqrt(max(0.0, eigvals[3]))
                w5 = math.sqrt(max(0.0, eigvals[4]))
                if w4 > max_w4:
                    max_w4 = w4
                if w5 < min_w5:
                    min_w5 = w5

        if min_w5 <= 0 or max_w4 <= 0:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)

    # ---------- Symmetrize a quadrant (enforce transpose symmetry within 8x8) ----------
    def symmetrize_quad(q):
        q = np.array(q, dtype=np.int8).reshape(8, 8)
        for i in range(8):
            for j in range(i + 1, 8):
                val = 1 if (q[i, j] + q[j, i]) >= 1 else 0
                q[i, j] = val
                q[j, i] = val
        return q.ravel()

    # ---------- Seed generators ----------
    def make_holes(radius, center=0.25, shape='circle'):
        q = np.ones((8, 8), dtype=np.int8)
        for i in range(8):
            for j in range(8):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                dx = x - center
                dy = y - center
                if shape == 'circle':
                    if dx * dx + dy * dy < radius * radius:
                        q[i, j] = 0
                elif shape == 'square':
                    if abs(dx) < radius and abs(dy) < radius:
                        q[i, j] = 0
        return q.ravel()

    def make_cross(width=2):
        q = np.ones((8, 8), dtype=np.int8)
        for i in range(8):
            for j in range(8):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x) < width / size or abs(y) < width / size:
                    q[i, j] = 0
        return q.ravel()

    def make_random():
        q = np.random.randint(0, 2, size=64).astype(np.int8)
        return symmetrize_quad(q)

    # ---------- Local search ----------
    rng = np.random.default_rng(12345)

    seeds = [
        make_holes(0.28, 0.25, 'circle'),
        make_holes(0.25, 0.25, 'circle'),
        make_holes(0.30, 0.25, 'circle'),
        make_holes(0.22, 0.25, 'circle'),
        make_holes(0.28, 0.25, 'square'),
        make_holes(0.25, 0.25, 'square'),
        make_cross(2),
        make_cross(3),
        make_random(),
        make_random(),
    ]
    seeds = [symmetrize_quad(s) for s in seeds]

    best_score = -1.0
    best_quad = seeds[0].copy()

    for s in seeds:
        sc = score_quad(s)
        if sc > best_score:
            best_score = sc
            best_quad = s.copy()

    # Precompute unique bit moves (upper triangle of 8x8)
    bit_moves = [i * 8 + j for i in range(8) for j in range(i, 8)]

    def flip_bit(q, idx):
        q = q.copy()
        i, j = divmod(idx, 8)
        q[idx] = 1 - q[idx]
        if i != j:
            q[j * 8 + i] = 1 - q[j * 8 + i]
        return q

    def hill_climb(start_quad, max_time):
        q = start_quad.copy()
        sc = score_quad(q)
        best_local = q.copy()
        best_local_sc = sc
        t_end = time.time() + max_time
        no_improve = 0

        while time.time() < t_end:
            improved = False
            for idx in bit_moves:
                if time.time() > t_end:
                    break
                q_new = flip_bit(q, idx)
                sc_new = score_quad(q_new)
                if sc_new > sc + 1e-12:
                    q = q_new
                    sc = sc_new
                    improved = True
                    if sc > best_local_sc:
                        best_local_sc = sc
                        best_local = q.copy()
                    no_improve = 0
            if not improved:
                n_flips = rng.integers(2, 5)
                chosen = rng.choice(len(bit_moves), size=n_flips, replace=False)
                q_new = q.copy()
                for idx in chosen:
                    q_new = flip_bit(q_new, idx)
                sc_new = score_quad(q_new)
                if sc_new > sc:
                    q = q_new
                    sc = sc_new
                    if sc > best_local_sc:
                        best_local_sc = sc
                        best_local = q.copy()
                    no_improve = 0
                else:
                    if rng.random() < 0.05:
                        q = q_new
                        sc = sc_new
                no_improve += 1
                if no_improve > 50:
                    break

        return best_local, best_local_sc

    top_seeds = sorted(seeds, key=lambda s: score_quad(s), reverse=True)[:5]

    for seed in top_seeds:
        if time.time() > deadline - 20.0:
            break
        time_per_seed = min(15.0, deadline - time.time() - 20.0)
        if time_per_seed < 2.0:
            break
        local_q, local_sc = hill_climb(seed, time_per_seed)
        if local_sc > best_score:
            best_score = local_sc
            best_quad = local_q.copy()

    if time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 10.0
        final_q, final_sc = hill_climb(best_quad, remaining)
        if final_sc > best_score:
            best_score = final_sc
            best_quad = final_q.copy()

    # ---------- Convert quadrant to full 32x32 cell ----------
    q2d = best_quad.reshape(8, 8)
    cell = np.zeros((size, size), dtype=np.int8)
    cell[:8, :8] = q2d
    cell[:8, 24:32] = q2d[:, ::-1]
    cell[24:32, :8] = q2d[:, ::-1]
    cell[24:32, 24:32] = q2d

    return [[int(cell[i][j]) for j in range(size)] for i in range(size)]
# EVOLVE-BLOCK-END
