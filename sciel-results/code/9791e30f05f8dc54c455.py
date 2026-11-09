import numpy as np
import time
import math

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    start_time = time.time()
    deadline = start_time + 195.0

    # --- Symmetry enforcement ---
    def enforce_symmetry(c):
        """Enforce C4v symmetry by majority vote over orbits."""
        sym = np.zeros_like(c, dtype=int)
        for i in range(size):
            for j in range(size):
                orbit = set()
                orbit.add((i, j))
                orbit.add((size - 1 - i, j))
                orbit.add((i, size - 1 - j))
                orbit.add((size - 1 - i, size - 1 - j))
                orbit.add((j, i))
                orbit.add((size - 1 - j, j))
                orbit.add((j, size - 1 - i))
                orbit.add((size - 1 - j, size - 1 - i))
                s = sum(c[pi, pj] for pi, pj in orbit)
                n = len(orbit)
                sym[i, j] = 1 if s > n / 2.0 else 0
        return sym

    def get_orbit(i, j):
        """Return unique orbit of (i,j) under C4v."""
        pts = {
            (i, j),
            (size - 1 - i, j),
            (i, size - 1 - j),
            (size - 1 - i, size - 1 - j),
            (j, i),
            (size - 1 - j, j),
            (j, size - 1 - i),
            (size - 1 - j, size - 1 - i),
        }
        return list(pts)

    # --- Fast plane-wave evaluator ---
    # Use a larger set of plane waves for accuracy but still fast
    pw_half = 5  # 11x11 = 121 plane waves
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1)
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    Gx_arr = np.array([g[0] for g in G_vectors], dtype=np.float64)
    Gy_arr = np.array([g[1] for g in G_vectors], dtype=np.float64)

    # Precompute G differences mod size for FFT lookup
    G_diff_h = np.zeros((N_pw, N_pw), dtype=int)
    G_diff_k = np.zeros((N_pw, N_pw), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diff_h[i, j] = (G_vectors[i][0] - G_vectors[j][0]) % size
            G_diff_k[i, j] = (G_vectors[i][1] - G_vectors[j][1]) % size

    # K-points in irreducible BZ
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # Gamma to M: (0, t)
    for t in np.linspace(0.05, 0.5, 10):
        k_points.append((0.0, t))
    # M to X: (t, 0.5-t) for t in [0.5, 1.0]... actually M=(0,0.5), X=(0.5,0.5)
    # Path M->X: (t, 0.5) for t in [0, 0.5]
    for t in np.linspace(0.05, 0.45, 10):
        k_points.append((t, 0.5))
    # Gamma to X: (t, t)
    for t in np.linspace(0.05, 0.45, 10):
        k_points.append((t, t))
    # X point
    k_points.append((0.5, 0.5))
    # Additional interior points
    k_points.append((0.25, 0.25))
    k_points.append((0.1, 0.3))
    k_points.append((0.3, 0.1))
    k_points.append((0.15, 0.4))
    k_points.append((0.4, 0.15))

    def evaluate_gap(cell_grid):
        """Compute gap-midgap ratio for TM bands 4 and 5."""
        inv_eps = np.where(cell_grid == 1, 1.0 / eps_high, 1.0 / eps_low)
        fft_1eps = np.fft.fft2(inv_eps)

        max_omega_4 = 0.0
        min_omega_5 = float('inf')

        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky

            # H_{ij} = (K_i · K_j) * eps_{G_i - G_j}
            # For TM: H_{ij} = sum over G of eps_G * (k+G_i)·(k+G_j) delta...
            # Actually for TM polarization in 2D:
            # (k+G_i)·(k+G_j) * (1/eps)_{G_i - G_j}
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            coeffs = fft_1eps[G_diff_h, G_diff_k] / (size * size)
            H = dots * coeffs
            H = (H + H.T) / 2.0  # symmetrize
            H = H.real

            # Only need eigenvalues 3 and 4 (0-indexed)
            # Use partial eigenvalue computation if possible
            try:
                eigvals = np.linalg.eigvalsh(H)
            except np.linalg.LinAlgError:
                continue

            if len(eigvals) >= 5:
                w4 = np.sqrt(max(0.0, eigvals[3]))
                w5 = np.sqrt(max(0.0, eigvals[4]))
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5

        if min_omega_5 == float('inf') or max_omega_4 <= 0:
            return 0.0

        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap <= 0:
            return 0.0
        return 2.0 * gap / midgap

    # --- Design representation: C4v-symmetric harmonic potential ---
    # The cell is defined as cell[i,j] = 1 if V(i,j) > 0
    # V is a C4v-symmetric function built from low-order basis functions.
    # Basis functions that are C4v-symmetric (invariant under all 8 group ops):
    # We work in coordinates centered at cell center: u = i - (size-1)/2, v = j - (size-1)/2
    # C4v-symmetric monomials: even powers, and symmetric combinations.
    # Use radial polynomials: r^2, r^4, r^6, r^8 and angular terms cos(4*theta), cos(8*theta)

    # Precompute coordinates
    coords = np.arange(size, dtype=np.float64)
    center = (size - 1) / 2.0
    I, J = np.meshgrid(coords, coords, indexing='ij')  # I = column, J = row
    u = I - center
    v = J - center
    r2 = u**2 + v**2
    theta = np.arctan2(v, u)

    # C4v-symmetric basis functions
    # These are all invariant under the 8-element C4v group
    basis = np.stack([
        np.ones((size, size)),                          # 0: constant
        r2,                                              # 1: r^2
        r2**2,                                           # 2: r^4
        r2**3,                                           # 3: r^6
        r2**4,                                           # 4: r^8
        np.cos(4 * theta) * r2,                          # 5: r^2 cos(4θ)
        np.cos(4 * theta) * r2**2,                       # 6: r^4 cos(4θ)
        np.cos(8 * theta) * r2,                          # 7: r^2 cos(8θ)
        np.cos(8 * theta) * r2**2,                       # 8: r^4 cos(8θ)
        np.cos(4 * theta) * r2**3,                       # 9: r^6 cos(4θ)
        np.cos(8 * theta) * r2**3,                       # 10: r^6 cos(8θ)
        np.sin(4 * theta) * np.cos(4 * theta) * r2,      # 11: r^2 sin(4θ)cos(4θ) = r^2 sin(8θ)/2
        r2**2 * np.cos(4*theta)**2 - r2**2 * np.sin(4*theta)**2,  # 12: r^4 cos(8θ) (redundant-ish)
    ], axis=0)  # shape: (n_basis, size, size)

    n_basis = basis.shape[0]
    # Flatten for fast evaluation
    basis_flat = basis.reshape(n_basis, -1)  # (n_basis, size*size)

    def params_to_cell(params):
        """Convert parameter vector to binary cell."""
        V = basis_flat @ params  # (size*size,)
        V = V.reshape(size, size)
        cell = (V > 0).astype(int)
        # Enforce exact symmetry to guard against floating point issues
        cell = enforce_symmetry(cell)
        return cell

    def score_params(params):
        cell = params_to_cell(params)
        return evaluate_gap(cell)

    # --- Optimization: Coarse-to-fine grid search + local refinement ---
    # Phase 1: Coarse grid search over a subset of parameters
    # Use the first 6 basis functions for the coarse search
    n_coarse = 6
    search_ranges = []
    for b in range(n_coarse):
        if b == 0:
            search_ranges.append((-2.0, 2.0))
        elif b == 1:
            search_ranges.append((-1.0, 1.0))
        elif b == 2:
            search_ranges.append((-1.0, 1.0))
        elif b == 3:
            search_ranges.append((-0.5, 0.5))
        elif b == 4:
            search_ranges.append((-0.3, 0.3))
        else:
            search_ranges.append((-1.0, 1.0))

    # Coarse grid: 3-4 points per dimension for n_coarse=6
    n_grid = 3
    grid_axes = []
    for b in range(n_coarse):
        lo, hi = search_ranges[b]
        grid_axes.append(np.linspace(lo, hi, n_grid))

    # Do a random sampling approach instead of full grid (6 dims x 3^6 = 729, manageable)
    # But let's use random sampling with more points
    n_random = 200
    best_score = -float('inf')
    best_params = np.zeros(n_basis)
    best_cell = np.zeros((size, size), dtype=int)

    # Initialize with a known good starting point
    init_params = np.zeros(n_basis)
    init_params[0] = 0.0
    init_params[1] = -0.3  # negative r^2 means center is air (dielectric at edges)
    init_params[2] = 0.2
    init_params[5] = 0.3   # add 4-fold angular structure
    best_params = init_params.copy()
    best_cell = params_to_cell(best_params)
    best_score = score_params(best_params)

    # Random sampling
    for _ in range(n_random):
        if time.time() > deadline - 30:
            break
        params = np.random.uniform(
            [r[0] for r in search_ranges] + [0.0] * (n_basis - n_coarse),
            [r[1] for r in search_ranges] + [0.5] * (n_basis - n_coarse)
        )
        sc = score_params(params)
        if sc > best_score:
            best_score = sc
            best_params = params.copy()
            best_cell = params_to_cell(params)

    # Phase 2: Local hill climbing / simulated annealing in parameter space
    current_params = best_params.copy()
    current_score = best_score
    temperature = 0.1
    cooling = 0.998
    min_temp = 0.005

    iteration = 0
    max_iters = 50000

    while time.time() < deadline - 10 and iteration < max_iters:
        iteration += 1

        # Perturb parameters
        step = 0.05
        delta_p = np.random.normal(0, step, n_basis)
        # Scale steps for higher-order terms
        for b in range(n_basis):
            if b >= 4:
                delta_p[b] *= 0.5

        new_params = current_params + delta_p
        new_score = score_params(new_params)

        delta = new_score - current_score
        if delta > 0 or np.random.random() < math.exp(delta / max(temperature, 1e-8)):
            current_params = new_params
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_params = new_params.copy()
                best_cell = params_to_cell(new_params)

        temperature *= cooling
        if temperature < min_temp:
            temperature = min_temp

    # Phase 3: Fine-tune with pixel flips (local search in pixel space)
    current_cell = best_cell.copy()
    current_score = best_score

    pixel_temp = 0.15
    pixel_cooling = 0.999
    pixel_min_temp = 0.02

    # Precompute all unique orbits for efficiency
    all_orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            orbit = get_orbit(i, j)
            key = tuple(sorted(orbit))
            if key not in seen:
                seen.add(key)
                all_orbits.append(orbit)

    np.random.seed(42)
    orbit_idx = 0
    for _ in range(20000):
        if time.time() > deadline - 5:
            break

        # Cycle through orbits in a shuffled order
        if orbit_idx >= len(all_orbits):
            np.random.shuffle(all_orbits)
            orbit_idx = 0
        orbit = all_orbits[orbit_idx]
        orbit_idx += 1

        new_cell = current_cell.copy()
        for pi, pj in orbit:
            new_cell[pi, pj] = 1 - new_cell[pi, pj]

        new_score = evaluate_gap(new_cell)
        delta = new_score - current_score
        if delta > 0 or np.random.random() < math.exp(delta / max(pixel_temp, 1e-8)):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = new_cell.copy()
                best_params = current_cell  # mark as pixel-optimized

        pixel_temp *= pixel_cooling
        if pixel_temp < pixel_min_temp:
            pixel_temp = pixel_min_temp

    # Final symmetry enforcement
    final_cell = enforce_symmetry(best_cell)
    return final_cell.tolist()
