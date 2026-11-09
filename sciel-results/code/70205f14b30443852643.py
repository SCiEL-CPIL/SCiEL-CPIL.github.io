"""Photonic crystal design: multi-start local search over ring/cross seeds with
fast FFT-based Fourier coefficients and a dense 30-k-point band sweep."""
import numpy as np
import time

def design(problem):
    start_time = time.time()
    size = problem["pixels"]          # 32
    eps_low = problem["eps_low"]      # 1
    eps_high = problem["eps_high"]    # 12

    N_pw = 8                          # (2*8+1)^2 = 289 plane waves
    G_range = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel())).astype(int)  # (289,2)
    num_pw = len(G_vectors)

    # Lookup: (gx,gy) -> index in G_vectors
    G_lookup = {}
    for idx in range(num_pw):
        G_lookup[(int(G_vectors[idx, 0]), int(G_vectors[idx, 1]))] = idx

    # ---- Symmetry orbits under the square group about cell centre ----
    def get_orbits(sz):
        visited = set()
        orbits = []
        for i in range(sz):
            for j in range(sz):
                if (i, j) not in visited:
                    orbit = {
                        (i, j),
                        (sz - 1 - i, sz - 1 - j),
                        (i, sz - 1 - j),
                        (sz - 1 - i, j),
                        (j, i),
                        (sz - 1 - j, sz - 1 - i),
                    }
                    visited.update(orbit)
                    orbits.append(list(orbit))
        return orbits

    orbits = get_orbits(size)
    num_orbits = len(orbits)

    def orbit_to_cell(orbit_vals):
        cell = np.zeros((size, size), dtype=np.uint8)
        for o in range(num_orbits):
            v = orbit_vals[o]
            for (i, j) in orbits[o]:
                cell[i, j] = v
        return cell

    # ---- Fast Fourier coefficients of 1/eps via 2D FFT ----
    def compute_fourier_eps_inv(cell):
        inv_eps = np.where(cell == 0, 1.0, 1.0 / eps_high)
        F = np.fft.fft2(inv_eps) / (size * size)
        # F[m,n] corresponds to frequency (m, n) with wrap: if m > size//2 -> m - size
        coeffs = np.zeros(num_pw, dtype=complex)
        for idx in range(num_pw):
            gx = int(G_vectors[idx, 0])
            gy = int(G_vectors[idx, 1])
            mx = gx % size
            my = gy % size
            coeffs[idx] = F[mx, my]
        return coeffs

    # ---- Dense k-points on the irreducible BZ wedge 0 <= ky <= kx <= 0.5 ----
    # 30 points: 10 along Gamma-X (ky=0), 10 along X-M (kx=0.5), 10 along Gamma-M (kx=ky)
    kx_vals = np.linspace(0.0, 0.5, 10)
    k_points = []
    for kx in kx_vals:
        k_points.append((kx, 0.0))          # Gamma-X
    for ky in kx_vals:
        k_points.append((0.5, ky))          # X-M
    for k in kx_vals:
        k_points.append((k, k))             # Gamma-M
    # Deduplicate (Gamma, X, M appear twice)
    k_points = list(set(k_points))
    # Ensure exactly 30 unique points; if fewer, add interior points
    if len(k_points) < 30:
        extra = [(0.25, 0.1), (0.35, 0.15), (0.45, 0.2), (0.3, 0.25),
                 (0.15, 0.05), (0.4, 0.3), (0.5, 0.1), (0.2, 0.15)]
        for p in extra:
            if len(k_points) >= 30:
                break
            if p not in k_points:
                k_points.append(p)
    k_points = k_points[:30]
    k_arr = np.array(k_points)  # (30, 2)

    # Precompute for each k-point: kG2 vector
    # kG2[k_idx, pw_idx] = (kx + Gx)^2 + (ky + Gy)^2
    kG2_all = np.zeros((len(k_points), num_pw))
    for ki in range(len(k_points)):
        kp = G_vectors + k_arr[ki]
        kG2_all[ki] = np.sum(kp ** 2, axis=1)

    def compute_bands_all(inv_eps_G):
        """Compute omega^2 for bands 4 and 5 at all k-points.
        Returns (max_w4_sq, min_w5_sq)."""
        # Build the 289x289 coefficient matrix C[a,b] = inv_eps_G[G_a - G_b]
        # Vectorized: for each a, the row is inv_eps_G at (G_a - G_b) for all b
        # Precompute diff matrix
        # diff[a,b] = G_vectors[a] - G_vectors[b]  (289,289,2)
        # But that's 289*289*2 = 166k entries, fine.
        diffs = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (289,289,2)
        # Map each diff to index in G_lookup
        # Vectorized lookup
        diff_flat = diffs.reshape(-1, 2)
        # Build lookup array: for gx in -8..8, gy in -8..8 -> index or -1
        # Create a 2D array indexed by (gx+8, gy+8)
        lookup_2d = np.full((2 * N_pw + 1, 2 * N_pw + 1), -1, dtype=int)
        for idx in range(num_pw):
            lookup_2d[int(G_vectors[idx, 0]) + N_pw, int(G_vectors[idx, 1]) + N_pw] = idx
        diff_idx = lookup_2d[diff_flat[:, 0] + N_pw, diff_flat[:, 1] + N_pw].reshape(num_pw, num_pw)
        # C[a,b] = inv_eps_G[diff_idx[a,b]] if diff_idx >= 0 else 0
        C = np.where(diff_idx >= 0, inv_eps_G[diff_idx], 0.0 + 0.0j)

        max_w4_sq = 0.0
        min_w5_sq = float('inf')
        for ki in range(len(k_points)):
            kG2 = kG2_all[ki]  # (289,)
            H = C * kG2[np.newaxis, :]  # H[a,b] = C[a,b] * kG2[b]
            # H should be Hermitian; ensure
            H = 0.5 * (H + H.conj().T)
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.real(eigvals)
            # Bands are 1-indexed in task; 0-indexed: band 4 = index 3, band 5 = index 4
            w4_sq = eigvals[3]
            w5_sq = eigvals[4]
            if w4_sq > max_w4_sq:
                max_w4_sq = w4_sq
            if w5_sq < min_w5_sq:
                min_w5_sq = w5_sq
        return max_w4_sq, min_w5_sq

    def evaluate_cell(orbit_vals):
        cell = orbit_to_cell(orbit_vals)
        inv_eps_G = compute_fourier_eps_inv(cell)
        max_w4_sq, min_w5_sq = compute_bands_all(inv_eps_G)
        if max_w4_sq <= 0 or min_w5_sq <= 0:
            return -1.0
        max_w4 = np.sqrt(max_w4_sq)
        min_w5 = np.sqrt(min_w5_sq)
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / midgap

    # ---- Seed generators ----
    def make_ring(inner_frac, outer_frac):
        vals = np.zeros(num_orbits, dtype=int)
        for o in range(num_orbits):
            i, j = orbits[o][0]
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x * x + y * y)
            if inner_frac <= r <= outer_frac:
                vals[o] = 1
        return vals

    def make_cross(w_frac):
        vals = np.zeros(num_orbits, dtype=int)
        for o in range(num_orbits):
            i, j = orbits[o][0]
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < w_frac or abs(y) < w_frac:
                vals[o] = 1
        return vals

    def make_ring_cross(inner_frac, outer_frac, w_frac):
        vals = np.zeros(num_orbits, dtype=int)
        for o in range(num_orbits):
            i, j = orbits[o][0]
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x * x + y * y)
            if inner_frac <= r <= outer_frac:
                vals[o] = 1
            if abs(x) < w_frac or abs(y) < w_frac:
                vals[o] = 1
        return vals

    def make_random():
        return np.random.randint(0, 2, num_orbits)

    # ---- Multi-start local search with simulated annealing ----
    np.random.seed(12345)

    seeds = [
        make_ring(0.15, 0.35),
        make_ring(0.2, 0.4),
        make_ring(0.1, 0.3),
        make_ring(0.25, 0.45),
        make_cross(0.15),
        make_cross(0.2),
        make_ring_cross(0.15, 0.35, 0.12),
        make_ring_cross(0.2, 0.4, 0.15),
        make_random(),
        make_random(),
    ]

    best_score = -float('inf')
    best_vals = None

    # Evaluate all seeds
    for seed in seeds:
        s = evaluate_cell(seed)
        if s > best_score:
            best_score = s
            best_vals = seed.copy()

    # Local search from best seed
    deadline = start_time + 210.0  # 20s safety margin before 230s kill
    current_vals = best_vals.copy()
    current_score = best_score
    T = 0.5  # initial temperature
    T_min = 0.01
    cooling = 0.995

    while time.time() < deadline:
        # Multi-flip move
        n_flips = np.random.randint(1, 6)
        indices = np.random.choice(num_orbits, n_flips, replace=False)
        new_vals = current_vals.copy()
        new_vals[indices] = 1 - new_vals[indices]
        new_score = evaluate_cell(new_vals)

        # Simulated annealing acceptance
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-9)):
            current_vals = new_vals
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_vals = current_vals.copy()
        T = max(T * cooling, T_min)

        # Occasional restart from a new seed
        if np.random.random() < 0.02:
            new_seed = np.random.choice([
                make_ring(np.random.uniform(0.1, 0.3), np.random.uniform(0.25, 0.5)),
                make_cross(np.random.uniform(0.1, 0.25)),
                make_ring_cross(np.random.uniform(0.1, 0.3), np.random.uniform(0.25, 0.5),
                                np.random.uniform(0.08, 0.2)),
                make_random(),
            ])
            s = evaluate_cell(new_seed)
            if s > current_score:
                current_vals = new_seed.copy()
                current_score = s
                if s > best_score:
                    best_score = s
                    best_vals = current_vals.copy()

    best_cell = orbit_to_cell(best_vals)
    return best_cell.tolist()
