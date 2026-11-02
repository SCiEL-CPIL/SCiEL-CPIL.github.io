import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-8
DEADLINE = 148.0  # safety margin before 150s kill


def _enforce_validity(params):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps by shrinking the smaller circle
    for _ in range(200):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle by ov (or split if equal)
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clamp to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


def _get_contact_graph(p, tol=1e-4):
    """
    Get the contact graph from a valid configuration.
    Returns list of (i, j) pairs that are in contact (distance <= r_i + r_j + tol)
    and list of (i, wall) where wall is 0=left, 1=right, 2=bottom, 3=top.
    """
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    contacts = []
    wall_contacts = []
    
    # Wall contacts
    for i in range(N):
        if abs(x[i] - r[i]) < tol:
            wall_contacts.append((i, 0))  # left
        if abs(1.0 - x[i] - r[i]) < tol:
            wall_contacts.append((i, 1))  # right
        if abs(y[i] - r[i]) < tol:
            wall_contacts.append((i, 2))  # bottom
        if abs(1.0 - y[i] - r[i]) < tol:
            wall_contacts.append((i, 3))  # top
    
    # Pairwise contacts
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    for k in range(len(iu[0])):
        i = iu[0][k]
        j = iu[1][k]
        if dist[k] <= r[i] + r[j] + tol:
            contacts.append((i, j))
    
    return contacts, wall_contacts


def _delaunay_polish(p, time_limit=10.0, seed=42):
    """
    Delaunay-based polishing:
    1. Identify contact graph from Delaunay triangulation.
    2. Solve for maximum sum of radii given fixed contact graph.
    3. Iterate.
    """
    best_p = p.copy()
    best_score = _score_valid(best_p)
    rng = np.random.RandomState(seed)
    start = time.time()
    
    for iteration in range(50):
        if time.time() > start + time_limit:
            break
        
        # Get contact graph
        contacts, wall_contacts = _get_contact_graph(best_p, tol=1e-3)
        
        if len(contacts) == 0:
            # No contacts, just try to grow radii
            break
        
        # Build constraints for SLSQP
        # Objective: maximize sum(r) => minimize -sum(r)
        def obj(params):
            return -np.sum(params[2::3])
            
        def obj_grad(params):
            g = np.zeros_like(params)
            g[2::3] = -1.0
            return g
            
        cons = []
        
        # Wall constraints for circles that are touching walls
        for i, wall in wall_contacts:
            if wall == 0:  # left: x_i - r_i = 0
                def c(params, i=i):
                    return params[3*i] - params[3*i+2]
                def j(params, i=i):
                    gj = np.zeros_like(params)
                    gj[3*i] = 1.0
                    gj[3*i+2] = -1.0
                    return gj
                cons.append({'type': 'eq', 'fun': c, 'jac': j})
            elif wall == 1:  # right: 1 - x_i - r_i = 0
                def c(params, i=i):
                    return 1.0 - params[3*i] - params[3*i+2]
                def j(params, i=i):
                    gj = np.zeros_like(params)
                    gj[3*i] = -1.0
                    gj[3*i+2] = -1.0
                    return gj
                cons.append({'type': 'eq', 'fun': c, 'jac': j})
            elif wall == 2:  # bottom: y_i - r_i = 0
                def c(params, i=i):
                    return params[3*i+1] - params[3*i+2]
                def j(params, i=i):
                    gj = np.zeros_like(params)
                    gj[3*i+1] = 1.0
                    gj[3*i+2] = -1.0
                    return gj
                cons.append({'type': 'eq', 'fun': c, 'jac': j})
            elif wall == 3:  # top: 1 - y_i - r_i = 0
                def c(params, i=i):
                    return 1.0 - params[3*i+1] - params[3*i+2]
                def j(params, i=i):
                    gj = np.zeros_like(params)
                    gj[3*i+1] = -1.0
                    gj[3*i+2] = -1.0
                    return gj
                cons.append({'type': 'eq', 'fun': c, 'jac': j})
        
        # Pairwise contact constraints
        for i, j in contacts:
            def c(params, i=i, j=j):
                dx = params[3*i] - params[3*j]
                dy = params[3*i+1] - params[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - params[3*i+2] - params[3*j+2]
            def j(params, i=i, j=j):
                gj = np.zeros_like(params)
                dx = params[3*i] - params[3*j]
                dy = params[3*i+1] - params[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dist = 1e-10
                gj[3*i] = dx / dist
                gj[3*j] = -dx / dist
                gj[3*i+1] = dy / dist
                gj[3*j+1] = -dy / dist
                gj[3*i+2] = -1.0
                gj[3*j+2] = -1.0
                return gj
            cons.append({'type': 'eq', 'fun': c, 'jac': j})
        
        # Also add inequality constraints to prevent overlap for non-contact pairs
        # This is important to maintain validity
        x = best_p[:, 0]
        y = best_p[:, 1]
        r = best_p[:, 2]
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx * dx + dy * dy)
        
        for k in range(len(iu[0])):
            i = iu[0][k]
            j = iu[1][k]
            # Skip if already in contact
            if (i, j) in contacts or (j, i) in contacts:
                continue
            # Add inequality constraint: dist >= r_i + r_j
            def c(params, i=i, j=j):
                dx = params[3*i] - params[3*j]
                dy = params[3*i+1] - params[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - params[3*i+2] - params[3*j+2]
            def j(params, i=i, j=j):
                gj = np.zeros_like(params)
                dx = params[3*i] - params[3*j]
                dy = params[3*i+1] - params[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dist = 1e-10
                gj[3*i] = dx / dist
                gj[3*j] = -dx / dist
                gj[3*i+1] = dy / dist
                gj[3*j+1] = -dy / dist
                gj[3*i+2] = -1.0
                gj[3*j+2] = -1.0
                return gj
            cons.append({'type': 'ineq', 'fun': c, 'jac': j})
        
        # Wall inequality constraints for all circles
        for i in range(N):
            # x_i - r_i >= 0
            def c(params, i=i):
                return params[3*i] - params[3*i+2]
            def j(params, i=i):
                gj = np.zeros_like(params)
                gj[3*i] = 1.0
                gj[3*i+2] = -1.0
                return gj
            cons.append({'type': 'ineq', 'fun': c, 'jac': j})
            
            # 1 - x_i - r_i >= 0
            def c(params, i=i):
                return 1.0 - params[3*i] - params[3*i+2]
            def j(params, i=i):
                gj = np.zeros_like(params)
                gj[3*i] = -1.0
                gj[3*i+2] = -1.0
                return gj
            cons.append({'type': 'ineq', 'fun': c, 'jac': j})
            
            # y_i - r_i >= 0
            def c(params, i=i):
                return params[3*i+1] - params[3*i+2]
            def j(params, i=i):
                gj = np.zeros_like(params)
                gj[3*i+1] = 1.0
                gj[3*i+2] = -1.0
                return gj
            cons.append({'type': 'ineq', 'fun': c, 'jac': j})
            
            # 1 - y_i - r_i >= 0
            def c(params, i=i):
                return 1.0 - params[3*i+1] - params[3*i+2]
            def j(params, i=i):
                gj = np.zeros_like(params)
                gj[3*i+1] = -1.0
                gj[3*i+2] = -1.0
                return gj
            cons.append({'type': 'ineq', 'fun': c, 'jac': j})
        
        # Bounds
        lo = np.zeros(3 * N)
        hi = np.zeros(3 * N)
        for i in range(N):
            lo[3*i] = 0.0
            hi[3*i] = 1.0
            lo[3*i+1] = 0.0
            hi[3*i+1] = 1.0
            lo[3*i+2] = 1e-6
            hi[3*i+2] = 0.5
        bounds = list(zip(lo, hi))
        
        # Solve
        try:
            res = minimize(
                obj,
                best_p.flatten(),
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.success:
                p_new = res.x.reshape(N, 3)
                p_new = _enforce_validity(p_new)
                s_new = _score_valid(p_new)
                
                if s_new > best_score + 1e-8:
                    best_score = s_new
                    best_p = p_new
                else:
                    # Stagnation: break out of the loop
                    break
                    
        except Exception:
            break
            
    return best_p


def _force_optimize(p, time_limit=10.0, seed=42):
    """
    Force-based optimization:
    - Apply repulsive forces between overlapping circles.
    - Apply attractive forces to grow radii.
    - Use velocity Verlet integration with damping.
    """
    best_p = p.copy()
    best_score = _score_valid(best_p)
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # Initialize velocities
    v = np.zeros((N, 3))
    
    # Parameters
    dt = 0.001
    damping = 0.9
    repulse_strength = 50.0
    grow_strength = 1.0
    max_force = 10.0
    
    for step in range(100000):
        if time.time() > start + time_limit:
            break
        
        x = best_p[:, 0]
        y = best_p[:, 1]
        r = best_p[:, 2]
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Repulsive forces between overlapping circles
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-8)
        
        overlap = (r[iu[0]] + r[iu[1]]) - dist
        active = overlap > 0
        
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            f = repulse_strength * overlap[k]
            fx[i] += f * dx[k] / dist_safe[k]
            fy[i] += f * dy[k] / dist_safe[k]
            fx[j] -= f * dx[k] / dist_safe[k]
            fy[j] -= f * dy[k] / dist_safe[k]
            fr[i] += f
            fr[j] += f
        
        # Wall repulsion
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += repulse_strength * (r[i] - x[i])
            if 1.0 - x[i] < r[i]:
                fx[i] -= repulse_strength * (r[i] - (1.0 - x[i]))
            if y[i] < r[i]:
                fy[i] += repulse_strength * (r[i] - y[i])
            if 1.0 - y[i] < r[i]:
                fy[i] -= repulse_strength * (r[i] - (1.0 - y[i]))
        
        # Growth force: try to grow all radii
        fr += grow_strength
        
        # Clamp forces
        fx = np.clip(fx, -max_force, max_force)
        fy = np.clip(fy, -max_force, max_force)
        fr = np.clip(fr, -max_force, max_force)
        
        # Update velocities
        v[:, 0] = (v[:, 0] + fx * dt) * damping
        v[:, 1] = (v[:, 1] + fy * dt) * damping
        v[:, 2] = (v[:, 2] + fr * dt) * damping
        
        # Update positions
        best_p[:, 0] += v[:, 0] * dt
        best_p[:, 1] += v[:, 1] * dt
        best_p[:, 2] += v[:, 2] * dt
        
        # Enforce validity
        best_p = _enforce_validity(best_p)
        
        # Check for improvement
        s_new = _score_valid(best_p)
        if s_new > best_score + 1e-8:
            best_score = s_new
            
    return best_p


def _random_initial(seed):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    r = rng.uniform(0.01, 0.05, size=N)
    p = np.column_stack([x, y, r])
    return _enforce_validity(p)


def _grid_initial(seed):
    """Grid initialization."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    cells = cells[:N]
    arr = np.array(cells, dtype=float)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.shape)
    return _enforce_validity(arr)


def _hex_initial(seed):
    """Hexagonal initialization."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(7):
        for j in range(7):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            points.append((x, y))
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    points += 0.5
    
    dists = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    
    p = points[indices]
    radii = np.full(N, 0.05)
    
    params = np.column_stack([p[:, 0], p[:, 1], radii])
    params += rng.uniform(-1e-4, 1e-4, size=params.shape)
    return _enforce_validity(params)


def _voronoi_initial(seed):
    """Voronoi-based initialization."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1-p[0], p[1], 1-p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
    
    params = np.column_stack([points[:, 0], points[:, 1], radii])
    params += rng.uniform(-1e-5, 1e-5, size=params.shape)
    return _enforce_validity(params)


def _anneal(p, time_limit=15.0, seed=42):
    """
    Simulated annealing to escape local minima.
    """
    best_p = p.copy()
    best_score = _score_valid(best_p)
    rng = np.random.RandomState(seed)
    start = time.time()
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    
    current_p = p.copy()
    current_score = _score_valid(current_p)
    
    for step in range(100000):
        if time.time() > start + time_limit:
            break
        if T < T_min:
            break
        
        # Perturb
        p_new = current_p.copy()
        p_new[:, :2] += rng.normal(0, 0.02, size=(N, 2))
        p_new[:, 2] += rng.normal(0, 0.01, size=N)
        p_new = _enforce_validity(p_new)
        
        new_score = _score_valid(p_new)
        delta = new_score - current_score
        
        # Accept with probability exp(-delta/T) if delta < 0
        if delta > 0 or rng.random() < np.exp(-delta / T):
            current_p = p_new
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_p = p_new.copy()
        
        T *= cooling_rate
        
    return best_p


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_params = None
    best_score = -np.inf

    # Multi-start with different initializations
    initializers = [
        _voronoi_initial,
        _hex_initial,
        _grid_initial,
        _random_initial,
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        for init_fn in initializers:
            if time.time() > deadline - 10:
                break
            
            x0 = init_fn(seed)
            s0 = _score_valid(x0)
            
            if s0 > best_score:
                best_score = s0
                best_params = x0.copy()
            
            # Force optimization
            time_per_method = 3.0
            if time.time() + time_per_method > deadline - 5:
                continue
            
            try:
                p_force = _force_optimize(x0, time_limit=time_per_method, seed=seed)
                s_force = _score_valid(p_force)
                if s_force > best_score:
                    best_score = s_force
                    best_params = p_force.copy()
            except Exception:
                pass
            
            # Delaunay polish
            if time.time() + time_per_method > deadline - 5:
                continue
            
            try:
                p_delaunay = _delaunay_polish(p_force if 'p_force' in dir() else x0, time_limit=time_per_method, seed=seed)
                s_delaunay = _score_valid(p_delaunay)
                if s_delaunay > best_score:
                    best_score = s_delaunay
                    best_params = p_delaunay.copy()
            except Exception:
                pass
            
            # Annealing
            if time.time() + time_per_method > deadline - 5:
                continue
            
            try:
                p_anneal = _anneal(best_params, time_limit=time_per_method, seed=seed)
                s_anneal = _score_valid(p_anneal)
                if s_anneal > best_score:
                    best_score = s_anneal
                    best_params = p_anneal.copy()
            except Exception:
                pass
    
    # If we didn't find any valid solution, use a fallback
    if best_params is None:
        best_params = _grid_initial(42)

    # Final refinement: Delaunay polish with more time
    remaining_time = deadline - time.time() - 2.0
    if remaining_time > 5.0:
        try:
            p_final = _delaunay_polish(best_params, time_limit=remaining_time, seed=42)
            s_final = _score_valid(p_final)
            if s_final > best_score:
                best_score = s_final
                best_params = p_final.copy()
        except Exception:
            pass

    # Return in the required format: (x, y, radius)
    result = best_params.copy()
    return result
