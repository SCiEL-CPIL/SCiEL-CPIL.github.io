"""Refined Chebyshev-parameterized TM band-gap optimizer with binary flip refinement."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Chebyshev-parameterized continuous field constrained to C4v symmetry.
    2. Fast proxy evaluator (7x7 PW basis, 10 k-points).
    3. Hill-climbing with adaptive step size and restarts.
    4. Binary flip refinement (local pixel search) as a polish pass.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    eps_mid = 0.5 * (eps_low + eps_high)
    eps_amp = 0.5 * (eps_high - eps_low)

    start_time = time.time()
    deadline = start_time + 195.0

    # --- 1. Symmetric parameterization via even-order Chebyshev ---
    cheb_orders = [0, 2, 4, 6, 8]
    param_indices = []
    for m in cheb_orders:
        for n in cheb_orders:
            if m <= n:
                param_indices.append((m, n))
    num_params = len(param_indices)

    x = np.array([(i + 0.5) / size for i in range(size)])
    y = np.array([(j + 0.5) / size for j in range(size)])
    X, Y = np.meshgrid(x, y, indexing='ij')
    U = 2.0 * X - 1.0
    V = 2.0 * Y - 1.0

    # Precompute Chebyshev polynomials T_k for k up to max order
    max_order = max(cheb_orders)
    T_cache = {}
    def cheb_k(t, k):
        if k in T_cache:
            return T_cache[k]
        if k == 0:
            r = np.ones_like(t)
        elif k == 1:
            r = t.copy()
        else:
            T0 = np.ones_like(t)
            T1 = t.copy()
            for _ in range(2, k + 1):
                T2 = 2.0 * t * T1 - T0
                T0, T1 = T1, T2
            r = T1
        T_cache[k] = r
        return r

    # Basis functions for each parameter index
    basis = np.zeros((num_params, size, size))
    for idx, (m, n) in enumerate(param_indices):
        Tm_U = cheb_k(U, m)
        Tn_V = cheb_k(V, n)
        if m == n:
            basis[idx] = Tm_U * Tn_V
        else:
            Tn_U = cheb_k(U, n)
            Tm_V = cheb_k(V, m)
            basis[idx] = Tm_U * Tn_V + Tn_U * Tm_V

    def params_to_field(params):
        # field = sum params[i] * basis[i]
        return np.tensordot(params, basis, axes=(0, 0))

    def field_to_eps(field):
        return eps_mid + eps_amp * np.tanh(field)

    # --- 2. Fast proxy evaluator ---
    # 7x7 plane waves: G vectors (h, k) for h, k in [-3, 3]
    G_list = [(h, k) for h in range(-3, 4) for k in range(-3, 4)]
    Npw = len(G_list)  # 49

    # Precompute G arrays
    Gh = np.array([g[0] for g in G_list], dtype=np.float64)
    Gk = np.array([g[1] for g in G_list], dtype=np.float64)

    # Precompute dot products G_i . G_j
    # dot[i, j] = Gh[i]*Gh[j] + Gk[i]*Gk[j]
    dotGG = np.outer(Gh, Gh) + np.outer(Gk, Gk)

    # Precompute difference indices: for each (i, j), dh = Gh[i]-Gh[j], dk = Gk[i]-Gk[j]
    # We need to look up FFT coefficient at (dh, dk)
    # Build a lookup: diff_idx[i, j] = index into a flat array of needed (dh, dk) pairs
    # Actually, we'll compute the needed coefficients on the fly using the FFT.

    # k-points in irreducible BZ (fractional coords, 0 <= kx <= ky, kx+ky <= 0.5 for square lattice?
    # Actually the square BZ: [-0.5, 0.5] x [-0.5, 0.5]. Irreducible triangle: kx >= 0, ky >= kx, kx+ky <= 0.5
    # Key points: Gamma (0,0), M (0.5,0) which is (0, 0.5) by symmetry, X (0.5, 0.5)
    # Let's use points along high-symmetry lines:
    # Gamma-M: (0, t) for t in [0, 0.5]
    # M-X: (t, 0.5-t) for t in [0, 0.5]  -- wait, M is (0.5, 0) or (0, 0.5). X is (0.5, 0.5).
    # Let's use: kx = 0 to 0.5, ky = 0.5 to 0? No.
    # Standard path: Gamma(0,0) -> X(0.5,0) -> M(0.5,0.5) -> Gamma(0,0)
    # But irreducible zone is triangle with vertices (0,0), (0.5,0), (0.5,0.5).
    # Let's sample:
    k_points = []
    # Along kx=0: (0, ky) for ky in [0, 0.5]
    for t in np.linspace(0.0, 0.5, 4):
        k_points.append((0.0, t))
    # Along ky=kx: (t, t) for t in [0, 0.5]
    for t in np.linspace(0.0, 0.5, 4):
        if t > 0:  # skip Gamma duplicate
            k_points.append((t, t))
    # Along ky=0.5: (kx, 0.5) for kx in [0, 0.5]
    for t in np.linspace(0.0, 0.5, 4):
        if t > 0:  # skip M duplicate
            k_points.append((t, 0.5))
    # Remove duplicates
    k_points = list(set(k_points))
    Nk = len(k_points)

    def evaluate_gap_fast(eps_grid):
        """
        Fast evaluation using 7x7 PW basis and a small set of k-points.
        Returns gap-midgap ratio between bands 4 and 5.
        """
        inv_eps = 1.0 / eps_grid
        fft_inv = np.fft.fft2(inv_eps) / (size * size)

        max_w4 = 0.0
        min_w5 = float('inf')

        for kx, ky in k_points:
            # Build H matrix: H[i, j] = (G_i + k) . (G_j + k) * c_{G_i - G_j}
            # (G_i + k) . (G_j + k) = G_i.G_j + G_i.k + G_j.k + k.k
            # = dotGG[i,j] + Gh[i]*kx + Gk[i]*ky + Gh[j]*kx + Gk[j]*ky + kx^2 + ky^2
            kk = kx * kx + ky * ky
            Gi_k = Gh * kx + Gk * ky  # G_i . k
            dot_full = dotGG + np.outer(Gi_k, np.ones(Npw)) + np.ones((Npw, 1)) @ Gi_k + kk
            # Wait: (G_i + k).(G_j + k) = G_i.G_j + G_i.k + G_j.k + k.k
            # = dotGG[i,j] + Gi_k[i] + Gi_k[j] + kk
            dot_full = dotGG + Gi_k[:, None] + Gi_k[None, :] + kk

            # c_{G_i - G_j}: look up fft_inv at index (dh % size, dk % size)
            dh = Gh[:, None] - Gh[None, :]  # (Npw, Npw)
            dk = Gk[:, None] - Gk[None, :]
            idx_h = dh % size
            idx_k = dk % size
            c = fft_inv[idx_h, idx_k]

            H = dot_full * c
            # H should be real symmetric for real symmetric eps
            H = 0.5 * (H + H.T)  # enforce symmetry
            eigvals = np.linalg.eigvalsh(H)

            if len(eigvals) >= 5:
                w4 = np.sqrt(max(0.0, eigvals[3]))
                w5 = np.sqrt(max(0.0, eigvals[4]))
                if w4 > max_w4:
                    max_w4 = w4
                if w5 < min_w5:
                    min_w5 = w5

        if min_w5 == float('inf'):
            return 0.0
        gap = min_w5 - max_w4
        midgap = 0.5 * (min_w5 + max_w4)
        if midgap <= 0:
            return 0.0
        return 2.0 * gap / midgap

    # --- 3. Hill-climbing optimization ---
    def enforce_symmetry_binary(cell):
        """Enforce C4v symmetry by averaging the 8 group-equivalent pixels."""
        sym = np.zeros_like(cell, dtype=np.float64)
        for i in range(size):
            for j in range(size):
                pts = [
                    (i, j), (size - 1 - i, j), (i, size - 1 - j), (size - 1 - i, size - 1 - j),
                    (j, i), (size - 1 - j, j), (j, size - 1 - i), (size - 1 - j, size - 1 - i)
                ]
                # Deduplicate
                unique_pts = list(set(pts))
                val = np.mean([cell[pi, pj] for pi, pj in unique_pts])
                sym[i, j] = val
        return (sym >= 0.5).astype(int)

    # Initial params: try a central dielectric rod
    params = np.zeros(num_params)
    # T0(0)=1, T2(0)=-1, T4(0)=1, T6(0)=-1, T8(0)=1
    # At center: field = A00*1 + A02*(-1) + ... 
    # We want high field at center => positive A00, negative A02, positive A04, etc.
    params[0] = 2.0  # A00
    params[1] = -1.0  # A02 (m=0,n=2)
    params[2] = 0.5   # A04
    params[3] = -0.3  # A06
    params[4] = 0.2   # A08
    params[5] = -0.5  # A22

    best_params = params.copy()
    best_score = evaluate_gap_fast(field_to_eps(params_to_field(params)))
    step_size = 0.15
    min_step = 0.005
    max_step = 1.5
    iter_count = 0
    max_iters = 8000

    while time.time() < deadline - 30.0 and iter_count < max_iters:
        iter_count += 1
        # Pick random parameter
        idx = np.random.randint(num_params)
        # Gaussian perturbation
        delta = np.random.normal(0, step_size)
        new_params = params.copy()
        new_params[idx] += delta

        new_field = params_to_field(new_params)
        new_eps = field_to_eps(new_field)
        new_score = evaluate_gap_fast(new_eps)

        if new_score > best_score:
            params = new_params
            best_score = new_score
            best_params = params.copy()
            step_size = min(max_step, step_size * 1.03)
        else:
            step_size = max(min_step, step_size * 0.92)

        # Periodic restart
        if iter_count % 400 == 0:
            perturb = np.random.normal(0, 0.4, num_params)
            trial_params = best_params + perturb
            trial_score = evaluate_gap_fast(field_to_eps(params_to_field(trial_params)))
            if trial_score > best_score:
                params = trial_params
                best_score = trial_score
                best_params = params.copy()
                step_size = 0.3

    # --- 4. Binary flip polish ---
    # Convert to binary cell
    final_field = params_to_field(best_params)
    final_eps = field_to_eps(final_field)
    threshold = 0.5 * (eps_low + eps_high)
    cell = (final_eps > threshold).astype(int)
    cell = enforce_symmetry_binary(cell)

    # Local binary flip search
    # For each pixel (in symmetric orbit), try flipping and see if score improves
    # Use the fast evaluator on the binary cell (eps is either 1 or 12)
    # Build orbit representatives
    orbit_reps = []
    visited = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            # Generate orbit
            orbit = set()
            orbit.add((i, j))
            orbit.add((size - 1 - i, j))
            orbit.add((i, size - 1 - j))
            orbit.add((size - 1 - i, size - 1 - j))
            orbit.add((j, i))
            orbit.add((size - 1 - j, j))
            orbit.add((j, size - 1 - i))
            orbit.add((size - 1 - j, size - 1 - i))
            orbit = list(orbit)
            for pt in orbit:
                visited.add(pt)
            orbit_reps.append((orbit, cell[i, j]))

    # Evaluate current binary cell
    def cell_to_eps(c):
        e = np.ones((size, size), dtype=np.float64)
        e[c == 1] = eps_high
        return e

    def eval_binary(cell_int):
        eps_g = cell_to_eps(cell_int)
        return evaluate_gap_fast(eps_g)

    current_score = eval_binary(cell)
    flip_improved = True
    max_flip_rounds = 5
    for rnd in range(max_flip_rounds):
        if time.time() > deadline - 5.0:
            break
        flip_improved = False
        # Randomize order for diversity
        indices = list(range(len(orbit_reps)))
        np.random.shuffle(indices)
        for oi in indices:
            if time.time() > deadline - 5.0:
                break
            orbit, cur_val = orbit_reps[oi]
            new_val = 1 - cur_val
            # Apply flip
            for (pi, pj) in orbit:
                cell[pi, pj] = new_val
            new_score = eval_binary(cell)
            if new_score > current_score + 1e-6:
                current_score = new_score
                flip_improved = True
                orbit_reps[oi] = (orbit, new_val)
            else:
                # Revert
                for (pi, pj) in orbit:
                    cell[pi, pj] = cur_val
        if not flip_improved:
            break

    # Final symmetry enforcement (should already be symmetric)
    cell = enforce_symmetry_binary(cell)

    return cell.tolist()

# EVOLVE-BLOCK-END
