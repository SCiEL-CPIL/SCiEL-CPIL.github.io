"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _enforce_symmetry(cell, size=32):
    """Enforce full square symmetry (D4) about cell centre."""
    c = np.array(cell, dtype=np.float64)
    c = (c + c[::-1].copy() + c[:, ::-1].copy() + c[::-1].copy()[:, ::-1].copy()
         + c.T + c[::-1].copy().T + c[:, ::-1].copy().T + c[::-1].copy()[:, ::-1].copy().T) / 8.0
    return (c > 0.5).astype(np.float64)

def _make_cell_from_params(params, size=32):
    """Build a symmetric 32x32 cell from a list of (cx, cy, r) circles in the UR quadrant.
    Circles are mirrored to all 4 quadrants via |x|, |y| folding."""
    cell = np.zeros((size, size), dtype=np.float64)
    coords = (np.arange(size) + 0.5) / size - 0.5
    x_abs = np.abs(coords)
    X, Y = np.meshgrid(x_abs, x_abs, indexing='ij')
    for (cx, cy, r) in params:
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        cell += (dist2 < r ** 2).astype(np.float64)
    cell = (cell > 0).astype(np.float64)
    return _enforce_symmetry(cell, size)

def _compute_fourier(inv_eps, size, indices):
    """Compute Fourier coefficients of inv_eps for the given G indices."""
    x_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, x_coords, indexing='ij')
    coeffs = {}
    for m in indices:
        for n in indices:
            Gx = 2 * np.pi * m
            Gy = 2 * np.pi * n
            phase = np.exp(-1j * (Gx * X + Gy * Y))
            coeffs[(m, n)] = np.sum(inv_eps * phase) / (size * size)
    return coeffs

def _pw_solver(cell, n_pw=13, k_points=None):
    """Fast TM plane-wave solver. Returns (max_omega4, min_omega5)."""
    size = cell.shape[0]
    eps_low = 1.0
    eps_high = 12.0
    inv_eps = np.where(cell > 0.5, 1.0 / eps_high, 1.0 / eps_low)

    n = n_pw
    indices = list(range(-n // 2, n // 2 + 1))
    fourier_coeffs = _compute_fourier(inv_eps, size, indices)

    if k_points is None:
        k_points = []
        # Gamma (0,0) to X (pi/2, pi/2)
        for t in np.linspace(0, 1, 6):
            k_points.append((t * np.pi / 2, t * np.pi / 2))
        # X to M (pi, 0)
        for t in np.linspace(0, 1, 6):
            k_points.append((np.pi / 2 + t * np.pi / 2, np.pi / 2 - t * np.pi / 2))
        # M to Gamma
        for t in np.linspace(0, 1, 6):
            k_points.append((np.pi - t * np.pi, 0.0))
        # Some interior points of the irreducible wedge
        k_points.append((0.3 * np.pi, 0.5 * np.pi))
        k_points.append((0.5 * np.pi, 0.7 * np.pi))
        k_points.append((0.2 * np.pi, 0.3 * np.pi))

    G_list = [(m, n) for m in indices for n in indices]
    n_pw_total = len(G_list)

    max_omega4 = 0.0
    min_omega5 = np.inf

    for kx, ky in k_points:
        A = np.zeros((n_pw_total, n_pw_total), dtype=np.float64)
        B = np.zeros((n_pw_total, n_pw_total), dtype=np.complex128)
        for i, (mi, ni) in enumerate(G_list):
            kx_G = kx - 2 * np.pi * mi
            ky_G = ky - 2 * np.pi * ni
            A[i, i] = kx_G ** 2 + ky_G ** 2
        for i, (mi, ni) in enumerate(G_list):
            for j, (mj, nj) in enumerate(G_list):
                key = (mi - mj, ni - nj)
                if key in fourier_coeffs:
                    B[i, j] = fourier_coeffs[key]
        try:
            from scipy.linalg import eig
            vals, _ = eig(A, B)
            vals_real = np.real(vals)
            vals_pos = vals_real[vals_real > 1e-10]
            vals_pos.sort()
            if len(vals_pos) >= 5:
                omega4 = np.sqrt(vals_pos[3])
                omega5 = np.sqrt(vals_pos[4])
                if omega4 > max_omega4:
                    max_omega4 = omega4
                if omega5 < min_omega5:
                    min_omega5 = omega5
        except Exception:
            continue

    return max_omega4, min_omega5

def _score_cell(cell):
    """Compute the gap-midgap ratio between TM bands 4 and 5."""
    try:
        max_omega4, min_omega5 = _pw_solver(cell, n_pw=13)
        if min_omega5 > max_omega4 > 0:
            gap = min_omega5 - max_omega4
            midgap = (min_omega5 + max_omega4) / 2.0
            if midgap > 0:
                return gap / midgap
        return -1.0
    except Exception:
        return -1.0

def _generate_initial_candidates(size=32):
    """Generate a diverse set of initial candidate parameter sets.
    Structural family: clusters of rods (1-4 circles) in the UR quadrant,
    which under D4 symmetry produce multi-rod unit cells (cross, ring, cluster)."""
    candidates = []
    # Single central rod (baseline)
    candidates.append([(0.0, 0.0, 0.15)])
    candidates.append([(0.0, 0.0, 0.20)])
    candidates.append([(0.0, 0.0, 0.25)])
    # Rod off-center (creates a 4-rod cluster under symmetry)
    for cx in [0.1, 0.15, 0.2, 0.25]:
        for cy in [0.0, 0.05, 0.1, 0.15]:
            for r in [0.08, 0.10, 0.12, 0.15]:
                candidates.append([(cx, cy, r)])
    # Two circles in UR quadrant (creates 8+ rods under symmetry)
    for c1 in [(0.05, 0.05, 0.08), (0.1, 0.05, 0.08), (0.15, 0.1, 0.1), (0.2, 0.1, 0.12)]:
        for c2 in [(0.25, 0.25, 0.1), (0.3, 0.15, 0.12), (0.15, 0.25, 0.1), (0.3, 0.3, 0.15)]:
            candidates.append([c1, c2])
    # Cross-like: rod at center + rod at edge
    for r_center in [0.1, 0.15]:
        for r_edge in [0.08, 0.10, 0.12]:
            candidates.append([(0.0, 0.0, r_center), (0.35, 0.0, r_edge)])
    return candidates

def design(problem):
    """Search for a symmetric 32x32 unit cell maximizing the TM band-4/5 gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill

    # Start with the baseline rod
    best_cell = _make_cell_from_params([(0.0, 0.0, 0.20)], size)
    best_score = _score_cell(best_cell)

    # Evaluate initial candidates
    candidates = _generate_initial_candidates(size)
    for params in candidates:
        if time.time() > deadline:
            break
        try:
            cell = _make_cell_from_params(params, size)
            score = _score_cell(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
        except Exception:
            continue

    # Local refinement: perturb the best candidate's parameters
    # Find which params produced best_cell by re-searching (or track it)
    # Simpler: do a coordinate-descent style refinement on a parametric form
    # We'll refine by trying small perturbations of radii and positions.
    # Reconstruct a parametric form: use the best cell, try to fit circles.
    # Instead, do a direct pixel-level local search on the best cell.

    # Local pixel search: flip small clusters of pixels to improve the gap
    def _local_search(cell, budget_seconds):
        cell = np.array(cell, dtype=np.float64)
        best_local_score = _score_cell(cell)
        local_deadline = time.time() + budget_seconds
        size = cell.shape[0]
        # Try flipping small patches
        rng = np.random.RandomState(42)
        for _ in range(200):
            if time.time() > local_deadline:
                break
            # Pick a random location and flip a small disk
            cx = rng.randint(0, size)
            cy = rng.randint(0, size)
            r = rng.randint(1, 4)
            trial = cell.copy()
            # Flip a disk around (cx, cy)
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if di * di + dj * dj <= r * r:
                        ii = (cx + di) % size
                        jj = (cy + dj) % size
                        trial[ii, jj] = 1.0 - trial[ii, jj]
            trial = _enforce_symmetry(trial, size)
            score = _score_cell(trial)
            if score > best_local_score:
                best_local_score = score
                cell = trial
        return cell, best_local_score

    # Allocate remaining time to local search
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    if remaining > 10:
        budget = min(remaining - 5, 150)
        best_cell, local_score = _local_search(best_cell, budget)
        if local_score > best_score:
            best_score = local_score

    # Convert to nested list
    result = [[int(v) for v in row] for row in best_cell]
    return result
# EVOLVE-BLOCK-END
