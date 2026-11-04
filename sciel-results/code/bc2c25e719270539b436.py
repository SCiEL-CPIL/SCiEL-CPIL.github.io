"""
Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3 to maximize minimum radius.
Combines a violation-energy landscape with iterative radius-increase, multi-start,
FCC lattice initialization, and contact-graph refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distance
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    # Pairwise distance via cKDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair_dist = np.min(dists[:, 1])
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _energy_and_grad(x, target_r):
    """
    Energy for a fixed target radius: sum of squared violations of
    pairwise separation (2*r) and wall distance (r).
    Returns (energy, gradient_flat).
    """
    c = x.reshape(N, 3)

    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)

    active = dists < 2.0 * target_r
    active[np.diag_indices_from(active)] = False

    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    energy_pair = 0.5 * np.sum(violations ** 2)

    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]

    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)

    wall_dists = HALF - np.abs(c)
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    energy_wall = np.sum(wall_violations ** 2)

    signs = np.sign(c)
    grad_wall = 2.0 * wall_violations * signs

    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()

    return total_energy, total_grad

def _local_optimize_violation(init_pts, target_r, t_start, deadline, max_iter=2000):
    """
    L-BFGS-B minimization of the violation energy for a fixed target radius.
    Uses the global deadline to bound runtime.
    """
    def objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)

    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts

def _generate_fcc():
    """
    Generate FCC lattice points in the cube.
    FCC has points at corners and face centers of each cubic cell.
    We use a 5x5x5 grid of cubic cells, but only keep points inside [-0.5, 0.5].
    Actually, let's use a simpler approach: generate FCC points in a larger region
    and select the first N that fit inside the cube.
    """
    points = []
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # in units of the cubic lattice constant a
    # We want the total span to be about 1.0, so a ~ 1/5 = 0.2 for 5 cells
    a = 0.2
    basis = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]])
    
    # Generate points in a 6x6x6 grid to ensure we have enough
    for ix in range(6):
        for iy in range(6):
            for iz in range(6):
                for b in basis:
                    pt = a * np.array([ix, iy, iz]) + a * b
                    # Center the whole thing so it's roughly in [-0.5, 0.5]
                    # The span is 6*a = 1.2, so center at 0.6*a = 0.12
                    # Actually, let's just generate and filter
                    points.append(pt)
    
    points = np.array(points)
    # Center the cloud
    points -= np.mean(points, axis=0)
    # Scale to fit in [-0.5, 0.5] roughly
    # The max extent is now about 0.6
    points *= 0.5 / 0.6
    
    # Filter points inside [-0.5, 0.5]
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    # If we have more than N, randomly select N
    if len(points) > N:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), N, replace=False)
        points = points[idx]
    
    # If we have fewer, pad with random points
    if len(points) < N:
        rng = np.random.default_rng(43)
        extra = rng.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
        
    return points

def _generate_grid():
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _contact_graph_refine(centers, radius, deadline):
    """
    Identify near-contact pairs and refine only the involved spheres
    to maximize the minimum distance among those pairs (and wall).
    """
    if time.time() > deadline - 3.0:
        return centers, _compute_radius(centers)

    centers = centers.copy()
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.1 * radius)

    if len(pairs) == 0:
        return centers, _compute_radius(centers)

    involved = set()
    for i, j in pairs:
        involved.add(i)
        involved.add(j)
    involved = sorted(involved)

    if len(involved) < 2:
        return centers, _compute_radius(centers)

    idx_array = np.array(involved)

    def sub_objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(len(involved), 3)
        min_d = np.inf
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        # We want to maximize min_d and wall, so minimize negative
        return -(min_d / 2.0 + np.min(wall))

    def sub_grad(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(len(involved), 3)

        min_d = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
                min_pair = (i, j)

        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        min_wall_idx = np.argmin(wall)

        grad_full = np.zeros((N, 3))

        if min_pair is not None:
            i, j = min_pair
            d = c[i] - c[j]
            dist = min_d
            if dist > 1e-12:
                # Gradient of -min_d/2 w.r.t c_i is -0.5 * (c_i - c_j)/dist
                # But we are minimizing -(min_d/2 + wall), so we want to maximize min_d
                # d(-min_d/2)/dc_i = -0.5 * (c_i - c_j)/dist
                grad_full[i] += -0.5 * d / dist
                grad_full[j] += -0.5 * (-d) / dist

        i_wall = idx_array[min_wall_idx]
        abs_c = np.abs(c[i_wall])
        k_max = np.argmax(abs_c)
        # Gradient of -wall = - (HALF - |c_k|) = |c_k| - HALF
        # d|c_k|/dc_k = sign(c_k)
        grad_full[i_wall, k_max] += np.sign(c[i_wall, k_max])

        return grad_full[idx_array].flatten()

    x0 = centers[idx_array].flatten()
    bounds_sub = [(-HALF + 1e-9, HALF - 1e-9)] * (len(involved) * 3)

    try:
        result = minimize(
            sub_objective,
            x0,
            jac=sub_grad,
            method='L-BFGS-B',
            bounds=bounds_sub,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_sub = result.x.reshape(len(involved), 3)
        centers[idx_array] = new_sub
    except TimeoutError:
        pass
    except Exception:
        pass

    centers = np.clip(centers, -HALF, HALF)
    return centers, _compute_radius(centers)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    best_centers = None
    best_radius = 0.0

    rng = np.random.default_rng(42)

    # Phase 1: Multi-start with violation-energy optimization
    initializations = []

    # FCC lattice
    fcc_centers = _generate_fcc()
    initializations.append(fcc_centers)

    # Grid
    grid_centers = _generate_grid()
    initializations.append(grid_centers)

    # Perturbed FCC
    for i in range(5):
        if time.time() > deadline - 50.0:
            break
        base = _generate_fcc()
        noise_scale = 0.03 + 0.015 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)

    # Perturbed Grid
    for i in range(3):
        if time.time() > deadline - 40.0:
            break
        base = _generate_grid()
        noise_scale = 0.03 + 0.015 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)

    # Random
    for i in range(3):
        if time.time() > deadline - 30.0:
            break
        random_c = rng.uniform(-0.45, 0.45, size=(N, 3))
        initializations.append(random_c)

    # Target radii to try
    # Known best is ~0.11138
    target_radii = [0.105, 0.108, 0.110, 0.111, 0.112, 0.113]

    for init in initializations:
        if time.time() > deadline - 30.0:
            break

        # Try each target radius from this initialization
        for target_r in target_radii:
            if time.time() > deadline - 25.0:
                break

            # Add small noise to break symmetry
            noise = rng.normal(0, 0.003, size=(N, 3))
            init_pts = np.clip(init + noise, -0.49, 0.49)

            opt_pts = _local_optimize_violation(init_pts, target_r, start_time, deadline)

            if np.all(np.isfinite(opt_pts)):
                r_opt = _compute_radius(opt_pts)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = opt_pts.copy()

    # Phase 2: Refinement using contact graph
    if best_centers is not None and time.time() < deadline - 10.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            best_centers, best_radius = _contact_graph_refine(best_centers, best_radius, deadline)

    # Final validation
    if best_centers is None:
        best_centers = _generate_grid()
    
    best_centers = np.clip(best_centers, -HALF + 1e-9, HALF - 1e-9)
    
    return best_centers
