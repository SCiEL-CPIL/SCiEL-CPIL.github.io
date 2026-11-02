"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _elapsed():
    if _GLOBAL_START is None:
        return 0.0
    return time.time() - _GLOBAL_START

def _remaining_time():
    if _GLOBAL_DEADLINE is None:
        return 145.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps

def _validate(xyz, tol=1e-7):
    """Check validity of a configuration."""
    n = xyz.shape[0]
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0.0, xyz[i, 2])
        xyz[i, 0] = np.clip(xyz[i, 0], r, 1.0 - r) if r < 0.5 else 0.5
        xyz[i, 1] = np.clip(xyz[i, 1], r, 1.0 - r) if r < 0.5 else 0.5
        xyz[i, 2] = r
    return xyz

def _objective(params, penalty_weight, smooth_eps=1e-4):
    """Full pairwise objective with smooth penalties."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    penalty = 0.0
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
    penalty += np.sum(_softplus(uy, smooth_eps) ** 2)
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

    return -np.sum(r) + penalty_weight * penalty

def _get_delaunay_edges(xyz, tol=1e-3):
    """
    Get contact edges via Delaunay triangulation of circle centers,
    filtered by proximity to actual contact (r_i + r_j).
    """
    points = xyz[:, :2]
    r = xyz[:, 2]
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    # Check if this pair is near contact
                    dx = points[a, 0] - points[b, 0]
                    dy = points[a, 1] - points[b, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < r[a] + r[b] + tol:
                        edges.add((a, b))
        if len(edges) == 0:
            # Fallback: all Delaunay edges
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
        return list(edges)
    except Exception:
        return []

def _objective_delaunay(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """
    Objective using Delaunay contact edges for speed.
    Falls back to full pairwise if contact set is empty or too large.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    penalty = 0.0
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
    penalty += np.sum(_softplus(uy, smooth_eps) ** 2)
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

    use_full = False
    if contact_edges is None or len(contact_edges) == 0 or len(contact_edges) > 200:
        use_full = True

    if not use_full:
        idx_i = np.array([e[0] for e in contact_edges])
        idx_j = np.array([e[1] for e in contact_edges])
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)
    else:
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

    return -np.sum(r) + penalty_weight * penalty

def _compute_energy(xyz, penalty_weight=100.0):
    """Hard penalty energy for PT-MC acceptance."""
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lx = np.maximum(0, r - x)
    ux = np.maximum(0, x + r - 1)
    ly = np.maximum(0, r - y)
    uy = np.maximum(0, y + r - 1)
    bound_pen = np.sum(lx ** 2 + ux ** 2 + ly ** 2 + uy ** 2)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
    overlap_pen = np.sum(overlap ** 2)
    neg_r_pen = np.sum(np.maximum(0, -r) ** 2)
    penalty = bound_pen + overlap_pen + neg_r_pen
    return -np.sum(r) + penalty_weight * penalty

def _pt_mc_search(initial_states, temps, max_steps, rng, penalty_weight=100.0):
    """
    Parallel tempering Monte Carlo search.
    Multiple replicas at different 'temperatures' (step sizes) explore
    different regions of configuration space.
    """
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    energies = [_compute_energy(s, penalty_weight) for s in states]
    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()

    idx_i, idx_j = np.triu_indices(N, k=1)

    for step in range(max_steps):
        if _remaining_time() < 3.0:
            break

        for rep in range(n_replicas):
            temp = temps[rep]
            # Generate a perturbation
            noise = rng.normal(0, temp, (N, 3))
            # Scale radius perturbations down
            noise[:, 2] *= 0.3
            new_state = states[rep] + noise
            new_state = _fix_boundaries(new_state)

            new_energy = _compute_energy(new_state, penalty_weight)
            delta_e = new_energy - energies[rep]

            # Metropolis acceptance
            if delta_e < 0 or rng.random() < np.exp(-delta_e / (0.01 + temp)):
                states[rep] = new_state
                energies[rep] = new_energy

                if new_energy < best_energy:
                    best_energy = new_energy
                    best_state = new_state.copy()

        # Swap adjacent replicas with Metropolis criterion
        for rep in range(n_replicas - 1):
            if rep % 2 == 0:  # Swap only even-indexed pairs to decorrelate
                e1, e2 = energies[rep], energies[rep + 1]
                t1, t2 = temps[rep], temps[rep + 1]
                # Swap probability: exp(-(e1/t2 - e2/t1))
                swap_arg = -(e1 / max(t2, 1e-6) - e2 / max(t1, 1e-6))
                if swap_arg > 0 or rng.random() < np.exp(min(swap_arg, 0)):
                    states[rep], states[rep + 1] = states[rep + 1].copy(), states[rep].copy()
                    energies[rep], energies[rep + 1] = energies[rep + 1], energies[rep]

    return best_state

def _initial_hex():
    """Hexagonal packing initial config."""
    positions = []
    r = 0.09
    cols, rows = 5, 6
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])

def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5 / side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])

def _initial_ring(seed=0):
    """Ring-shaped initial config."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.08]]
    for ring, n_circles, radius in [(1, 6, 0.10), (2, 10, 0.08), (3, 9, 0.07)]:
        for k in range(n_circles):
            angle = 2 * np.pi * k / n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist * np.cos(angle)
            y = 0.5 + dist * np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.06])
    return np.array(positions[:N])

def _initial_corner_cluster():
    """Larger circles in corners and edges."""
    positions = []
    positions.append([0.11, 0.11, 0.13])
    positions.append([0.89, 0.11, 0.13])
    positions.append([0.11, 0.89, 0.13])
    positions.append([0.89, 0.89, 0.13])
    positions.append([0.5, 0.09, 0.11])
    positions.append([0.5, 0.91, 0.11])
    positions.append([0.09, 0.5, 0.11])
    positions.append([0.91, 0.5, 0.11])
    positions.append([0.5, 0.5, 0.13])
    rng = np.random.default_rng(123)
    while len(positions) < N:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        r = rng.uniform(0.05, 0.09)
        positions.append([x, y, r])
    return np.array(positions[:N])

def _optimize_single(xyz0):
    """
    Multi-phase optimization from a single initial config:
    1. Low penalty L-BFGS-B (layout expansion)
    2. Medium penalty L-BFGS-B
    3. High penalty L-BFGS-B
    4. Very high penalty L-BFGS-B
    5. Ultra-high penalty with Delaunay contact graph
    6. PT-MC perturbation + re-optimization
    """
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    # Phase 1: Low penalty
    remaining = _remaining_time()
    if remaining > 8:
        penalty = 5.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 2: Medium penalty
    remaining = _remaining_time()
    if remaining > 6:
        penalty = 50.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 3: High penalty
    remaining = _remaining_time()
    if remaining > 4:
        penalty = 500.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective(params, penalty, 1e-5):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 4: Very high penalty with Delaunay contact graph
    remaining = _remaining_time()
    if remaining > 2:
        penalty = 5000.0
        xyz_cur = params.reshape(N, 3)
        edges = _get_delaunay_edges(xyz_cur)
        try:
            res = minimize(_objective_delaunay, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6, edges),
                          options={'maxiter': 250, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective_delaunay(params, penalty, 1e-6, edges):
                # Verify with full objective
                new_params = res.x
                if _objective(new_params, penalty, 1e-6) < _objective(params, penalty, 1e-6):
                    params = new_params
                    best_params = params.copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 5: PT-MC perturbation and re-optimize
    remaining = _remaining_time()
    if remaining > 5:
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        xyz_cur = params.reshape(N, 3)
        # Run PT-MC with a few replicas
        temps = [0.005, 0.01, 0.02, 0.04]
        n_steps = min(200, int(remaining * 30))
        pt_result = _pt_mc_search([xyz_cur], temps, n_steps, rng, penalty_weight=200.0)

        if _validate(pt_result):
            pt_score = np.sum(pt_result[:, 2])
            if pt_score > best_score:
                # Re-optimize from PT-MC result
                pt_params = pt_result.flatten().copy()
                try:
                    res = minimize(_objective, pt_params, method='L-BFGS-B',
                                  bounds=bounds, args=(5000.0, 1e-6),
                                  options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-13})
                    if res.fun < _objective(pt_params, 5000.0, 1e-6):
                        new_params = res.x
                        new_score = np.sum(new_params[2::3])
                        if new_score > best_score:
                            best_params = new_params.copy()
                            best_score = new_score
                except Exception:
                    pass
                # Also accept the PT-MC result directly if valid
                if pt_score > best_score:
                    best_params = pt_result.flatten().copy()
                    best_score = pt_score

    return best_params.reshape(N, 3), best_score

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0

    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])

    # Diverse initial configurations
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_corner_weighted(2),
        _initial_ring(0),
        _initial_ring(1),
        _initial_corner_cluster()
    ]

    # Optimize from each initialization
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 12.0:
            break
        xyz_opt, score = _optimize_single(xyz0)

        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(50):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.98
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix

        if not _validate(xyz_opt):
            continue

        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt

    # Local refinement from the best solution
    remaining = _remaining_time()
    if remaining > 12.0:
        rng = np.random.default_rng(42)
        for scale in [0.003, 0.008, 0.015, 0.025]:
            if remaining < 4.0:
                break
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)

            xyz_opt, score = _optimize_single(perturbed)

            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(50):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.98
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix

            if not _validate(xyz_opt):
                continue

            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
            remaining = _remaining_time()

    # Final high-precision polish with Delaunay contact graph
    remaining = _remaining_time()
    if remaining > 3.0:
        params = best_xyz.flatten().copy()
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))

        for penalty, eps, maxiter in [(50000, 1e-7, 150), (100000, 1e-8, 100), (200000, 1e-9, 80)]:
            remaining = _remaining_time()
            if remaining < 1.5:
                break
            xyz_cur = params.reshape(N, 3)
            edges = _get_delaunay_edges(xyz_cur)
            try:
                res = minimize(_objective_delaunay, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, eps, edges),
                              options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < _objective_delaunay(params, penalty, eps, edges):
                    new_params = res.x
                    if _objective(new_params, penalty, eps) < _objective(params, penalty, eps):
                        params = new_params
                        candidate = params.reshape(N, 3)
                        if _validate(candidate):
                            new_score = np.sum(candidate[:, 2])
                            if new_score > best_score:
                                best_score = new_score
                                best_xyz = candidate
            except Exception:
                pass

    # Final validation
    if not _validate(best_xyz):
        # Try to repair
        for _ in range(100):
            best_xyz[:, 2] *= 0.99
            best_xyz = _fix_boundaries(best_xyz)
            if _validate(best_xyz):
                break
        if not _validate(best_xyz):
            best_xyz = _initial_grid()

    return best_xyz

# EVOLVE-BLOCK-END
