"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _validate(xyz, tol=1e-7):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -tol):
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


def _fix_boundaries(circles):
    """Project circles inside unit square to handle numerical drift."""
    circles = circles.copy()
    for i in range(N):
        r = max(0, circles[i, 2])
        if r < 0.5:
            circles[i, 0] = np.clip(circles[i, 0], r, 1 - r)
            circles[i, 1] = np.clip(circles[i, 1], r, 1 - r)
        else:
            circles[i, 0] = 0.5
            circles[i, 1] = 0.5
        circles[i, 2] = r
    return circles


def _get_contact_edges(circles, threshold_factor=1.3):
    """
    Get contact edges from Delaunay triangulation of centers.
    Only include edges where circles are close to touching.
    """
    centers = circles[:, :2]
    radii = circles[:, 2]
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k + 1) % 3]
                if a == b:
                    continue
                i, j = (a, b) if a < b else (b, a)
                dist = np.linalg.norm(centers[a] - centers[b])
                min_dist = radii[i] + radii[j]
                if dist < threshold_factor * min_dist:
                    edges.add((i, j))
        return list(edges)
    except Exception:
        return None


def _physics_relax(xyz, time_limit=5.0, grow_rate=0.0008, dt=0.004):
    """
    Force-directed physics relaxation that grows radii over time.
    This is the primary optimization engine - it naturally handles
    circle packing constraints through repulsion forces and continuous
    radius growth to push toward the boundary of feasibility.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    # Track best valid configuration
    best_score = np.sum(r)
    best_xyz = xyz.copy()
    
    # Velocity tracking for momentum
    vx = np.zeros(N)
    vy = np.zeros(N)
    velocity_damp = 0.92
    
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # Grow radii slowly to push packing tighter
        r *= (1.0 + grow_rate)
        
        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) + 1e-10
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Strong repulsion proportional to overlap
                    force = (min_dist - dist) * 80.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces - push circles away from walls
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 150.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 150.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 150.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 150.0
        
        # Update velocities with damping
        vx = vx * velocity_damp + dt * fx
        vy = vy * velocity_damp + dt * fy
        
        # Update positions
        x += vx
        y += vy
        
        # Clip to valid positions
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
            # Zero velocity if we hit a wall
            if x[i] == r[i] or x[i] == 1 - r[i]:
                vx[i] = 0
            if y[i] == r[i] or y[i] == 1 - r[i]:
                vy[i] = 0
        
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
        
        # Check validity and track best
        if _validate(xyz):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_xyz = xyz.copy()
    
    return best_xyz


def _objective_contact(params, penalty_weight, contact_edges):
    """
    Objective using only contact edges for efficiency.
    Minimize -sum(r) + penalty_weight * penalty
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    penalty = 0.0

    # Boundary violations (smooth)
    penalty += np.sum(np.maximum(0, r - x) ** 2)
    penalty += np.sum(np.maximum(0, x + r - 1.0) ** 2)
    penalty += np.sum(np.maximum(0, r - y) ** 2)
    penalty += np.sum(np.maximum(0, y + r - 1.0) ** 2)
    penalty += np.sum(np.maximum(0, -r) ** 2)

    # Overlap penalties on contact edges only
    for (i, j) in contact_edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(max(dist_sq, 1e-12))
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            penalty += overlap ** 2

    return -np.sum(r) + penalty_weight * penalty


def _objective_full(params, penalty_weight):
    """Full pairwise objective for final polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    penalty = 0.0

    # Boundary penalties
    penalty += np.sum(np.maximum(0, r - x) ** 2)
    penalty += np.sum(np.maximum(0, x + r - 1.0) ** 2)
    penalty += np.sum(np.maximum(0, r - y) ** 2)
    penalty += np.sum(np.maximum(0, y + r - 1.0) ** 2)
    penalty += np.sum(np.maximum(0, -r) ** 2)

    # Full pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2

    return -np.sum(r) + penalty_weight * penalty


def _polish(xyz, time_budget=8.0):
    """
    Gradient-based polish using contact graph constraints.
    Multiple passes with increasing penalty weights.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))

    # Phase 1: Use contact edges for efficiency
    circles = params.reshape(N, 3)
    contact_edges = _get_contact_edges(circles, threshold_factor=1.5)
    if contact_edges is None or len(contact_edges) == 0:
        contact_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]

    for penalty_weight in [50.0, 200.0, 1000.0]:
        if time.time() - start_time > time_budget - 2.0:
            break
        
        try:
            res = minimize(
                _objective_contact,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                args=(penalty_weight, contact_edges),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            params = res.x
            current_score = np.sum(params[2::3])
            
            circles = params.reshape(N, 3)
            if _validate(circles):
                if current_score > best_score:
                    best_score = current_score
                    best_params = params.copy()
        except Exception:
            pass

    # Phase 2: Full pairwise polish for final precision
    if time.time() - start_time < time_budget - 1.0:
        for penalty_weight in [5000.0, 20000.0]:
            if time.time() - start_time > time_budget:
                break
            try:
                res = minimize(
                    _objective_full,
                    best_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    args=(penalty_weight,),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
                )
                params = res.x
                circles = params.reshape(N, 3)
                if _validate(circles):
                    current_score = np.sum(params[2::3])
                    if current_score > best_score:
                        best_score = current_score
                        best_params = params.copy()
            except Exception:
                pass

    best_circles = best_params.reshape(N, 3)
    if not _validate(best_circles):
        best_circles = _fix_boundaries(best_circles)
        if not _validate(best_circles):
            for _ in range(30):
                best_circles[:, 2] *= 0.97
                if _validate(best_circles):
                    break
    
    return best_circles


def _initial_hex():
    """Create a hexagonal initial configuration."""
    positions = []
    r = 0.12
    dy = r * np.sqrt(3)
    dx = 2 * r
    y = r
    row = 0
    while y < 1 - r and len(positions) < N:
        x_offset = 0 if row % 2 == 0 else r
        x = r + x_offset
        while x < 1 - r and len(positions) < N:
            positions.append([x, y, r])
            x += dx
        y += dy
        row += 1
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.8])
    return np.array(positions[:N])


def _initial_grid():
    """Create a grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            cells.append([x, y, r])
            if len(cells) >= N:
                break
        if len(cells) >= N:
            break
    while len(cells) < N:
        cells.append([0.5, 0.5, 0.08])
    return np.array(cells[:N])


def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88],
               [0.5, 0.12], [0.5, 0.88], [0.12, 0.5], [0.88, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.11])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        positions.append([x, y, rng.uniform(0.09, 0.12)])
    return np.array(positions[:N])


def _initial_mixed(seed=0):
    """Mix of structured and random positions."""
    rng = np.random.default_rng(seed)
    positions = []
    # 4 corners
    for cx, cy in [[0.13, 0.13], [0.87, 0.13], [0.13, 0.87], [0.87, 0.87]]:
        positions.append([cx, cy, 0.12])
    # 4 edge midpoints
    for cx, cy in [[0.5, 0.12], [0.5, 0.88], [0.12, 0.5], [0.88, 0.5]]:
        positions.append([cx, cy, 0.11])
    # Center
    positions.append([0.5, 0.5, 0.13])
    # Fill rest randomly
    for _ in range(N - len(positions)):
        x = rng.uniform(0.2, 0.8)
        y = rng.uniform(0.2, 0.8)
        positions.append([x, y, rng.uniform(0.08, 0.11)])
    return np.array(positions[:N])


def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global _start_time
    _start_time = time.time()
    deadline = _TIME_LIMIT

    best_xyz = None
    best_score = -np.inf

    # Diverse initializations
    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(42),
        _initial_corner_weighted(123),
        _initial_corner_weighted(789),
        _initial_mixed(55),
        _initial_mixed(99),
    ]

    # Time budget allocation:
    # - Physics relaxation: ~15s per init (up to 6 inits = 90s)
    # - Polish: ~8s per top result
    # - Total target: ~140s

    for i, xyz0 in enumerate(inits):
        if time.time() > deadline - 15.0:
            break

        # Physics relaxation with time budget
        phys_time = min(12.0, deadline - time.time() - 10.0)
        if phys_time < 3.0:
            break

        xyz_phys = _physics_relax(xyz0, time_limit=phys_time)

        # Polish the physics result
        polish_time = min(6.0, deadline - time.time() - 5.0)
        if polish_time < 1.0:
            # Just use physics result if no time for polish
            if _validate(xyz_phys):
                score = np.sum(xyz_phys[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz_phys
            continue

        xyz_polished = _polish(xyz_phys, time_budget=polish_time)

        # Evaluate both physics and polished results
        for candidate in [xyz_phys, xyz_polished]:
            if _validate(candidate):
                score = np.sum(candidate[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = candidate
            else:
                fixed = _fix_boundaries(candidate)
                if _validate(fixed):
                    score = np.sum(fixed[:, 2])
                    if score > best_score:
                        best_score = score
                        best_xyz = fixed

    # Final: if we have time, do one more polish on the best result
    if best_xyz is not None and time.time() < deadline - 3.0:
        final_polish_time = min(5.0, deadline - time.time() - 2.0)
        if final_polish_time > 1.0:
            xyz_final = _polish(best_xyz, time_budget=final_polish_time)
            if _validate(xyz_final):
                score = np.sum(xyz_final[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz_final

    # Fallback if no valid solution found
    if best_xyz is None:
        best_xyz = _initial_hex()
        # Ensure it's valid
        if not _validate(best_xyz):
            best_xyz = _fix_boundaries(best_xyz)
            if not _validate(best_xyz):
                best_xyz[:, 2] *= 0.8
                if not _validate(best_xyz):
                    best_xyz[:, 2] *= 0.7

    return best_xyz

# EVOLVE-BLOCK-END
