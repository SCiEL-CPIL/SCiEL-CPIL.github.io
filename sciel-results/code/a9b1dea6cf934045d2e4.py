import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize

N = 50

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii.
    Returns: hull, centers (array), radii (array), worst_idx
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-15):
        return None, None, None, None
    points = points / norms
    
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return None, None, None, None
    
    n_facets = len(hull.simplices)
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        ab = np.dot(a, b)
        ac = np.dot(a, c)
        bc = np.dot(b, c)
        
        Gram = np.array([
            [1.0, ab, ac],
            [ab, 1.0, bc],
            [ac, bc, 1.0]
        ])
        
        try:
            m = np.linalg.solve(Gram, np.ones(3))
            c_dir = m[0]*a + m[1]*b + m[2]*c
            norm_c_dir = np.linalg.norm(c_dir)
            if norm_c_dir < 1e-10:
                continue
            centers[i] = c_dir / norm_c_dir
            radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1.0, 1.0))
        except np.linalg.LinAlgError:
            continue
    
    if n_facets > 0 and np.any(radii > 0):
        worst_idx = np.argmax(radii)
    else:
        worst_idx = None
    
    return hull, centers, radii, worst_idx

def _compute_covering_radius(points):
    """
    Compute the actual covering radius (max facet circumradius).
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi
    return radii[worst_idx]

def _soft_max_radius(points, beta=50.0):
    """
    Compute soft-max of facet radii for smooth optimization.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, None
    max_r = np.max(radii)
    exp_terms = np.exp(beta * (radii - max_r))
    log_sum = np.log(np.sum(exp_terms))
    soft_radius = max_r + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    return soft_radius, weights

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max radius and its gradient w.r.t. points.
    Uses a more numerically stable approach.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    # Use log-sum-exp trick for numerical stability
    shifted = beta * (radii - max_r)
    max_shifted = np.max(shifted)
    exp_terms = np.exp(shifted - max_shifted)
    log_sum = max_shifted + np.log(np.sum(exp_terms))
    soft_radius = max_r + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    
    grads = np.zeros_like(points)
    n_facets = len(hull.simplices)
    
    for i in range(n_facets):
        if radii[i] < 1e-12 or weights[i] < 1e-15:
            continue
        w = weights[i]
        C = centers[i]
        r = radii[i]
        sin_r = np.sin(r)
        if sin_r < 1e-12:
            continue
        
        grad_contrib = -C / sin_r
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads

def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _lbfgs_stage(points, time_budget, beta_schedule=None):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    Adaptive beta scheduling: start with lower beta for global structure,
    increase for local refinement.
    """
    start_time = time.time()
    
    if beta_schedule is None:
        beta_schedule = [30.0, 50.0, 100.0]
    
    current_pts = points.copy()
    
    for beta in beta_schedule:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
        
        def objective_and_grad(x, b=beta):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _project_to_sphere(x.reshape(N, 3))
            soft_r, grads = _soft_max_radius_and_grad(pts, beta=b)
            grad_flat = grads.reshape(-1)
            return soft_r, grad_flat
        
        x0 = current_pts.reshape(-1)
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.success or result.fun < _compute_covering_radius(current_pts):
                current_pts = _project_to_sphere(result.x.reshape(N, 3))
        except Exception:
            pass
    
    return current_pts

def _local_polish(points, time_budget, beta=100.0):
    """
    Final local polish using projected gradient descent on tangent space.
    Directly minimizes the soft-max covering radius.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.02
    min_step = 1e-10
    decay = 0.9
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _soft_max_radius_and_grad(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-13:
            best_radius = true_radius
            best_points = current_points.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-14:
            break
        
        direction = grads / grad_norm
        new_points = current_points + step_size * direction
        new_points = _project_to_sphere(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.1)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _grid_slqp_polish(points, time_budget):
    """
    Grid-based SLSQP polish: use the worst facet center as a reference,
    then optimize with SLSQP using equality constraints (unit sphere).
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    
    # Multiple SLSQP passes with increasing beta
    for beta in [50.0, 100.0, 200.0]:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 5.0:
            break
        
        def objective(x, b=beta):
            if time.time() - start_time > time_budget:
                return np.pi
            pts = _project_to_sphere(x.reshape(N, 3))
            soft_r, _ = _soft_max_radius(pts, beta=b)
            return soft_r
        
        def objective_grad(x, b=beta):
            if time.time() - start_time > time_budget:
                return np.pi, np.zeros_like(x)
            pts = _project_to_sphere(x.reshape(N, 3))
            soft_r, grads = _soft_max_radius_and_grad(pts, beta=b)
            return soft_r, grads.reshape(-1)
        
        # Unit sphere constraints: ||p_i||^2 = 1 for each i
        def make_constraint(i):
            def constraint(x):
                p = x[i*3:(i+1)*3]
                return np.dot(p, p) - 1.0
            def constraint_jac(x):
                jac = np.zeros(3 * N)
                p = x[i*3:(i+1)*3]
                jac[i*3:(i+1)*3] = 2.0 * p
                return jac
            return constraint, constraint_jac
        
        constraints = []
        for i in range(N):
            c, cj = make_constraint(i)
            constraints.append({'type': 'eq', 'fun': c, 'jac': cj})
        
        x0 = current_points.reshape(-1)
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                jac=objective_grad,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            if result.success or result.fun < best_radius:
                new_pts = _project_to_sphere(result.x.reshape(N, 3))
                new_rad = _compute_covering_radius(new_pts)
                if new_rad < best_radius:
                    best_radius = new_rad
                    best_points = new_pts.copy()
                    current_points = new_pts
        except Exception:
            pass
    
    return best_points, best_radius

def _repulsive_polish(points, time_budget):
    """
    Repulsive force based polish for fine-tuning.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.01
    min_step = 1e-10
    decay = 0.9
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        hull, centers, radii, worst_idx = _compute_facet_data(current_points)
        if hull is None:
            break
        
        true_radius = radii[worst_idx] if worst_idx is not None else np.pi
        if true_radius < best_radius - 1e-13:
            best_radius = true_radius
            best_points = current_points.copy()
        
        n = len(current_points)
        forces = np.zeros_like(current_points)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i+1, n):
                d_vec = current_points[i] - current_points[j]
                d = np.linalg.norm(d_vec)
                if d < 0.01:
                    d = 0.01
                d_hat = d_vec / d
                radial_i = np.dot(d_hat, current_points[i]) * current_points[i]
                tangential_i = d_hat - radial_i
                mag = 1.0 / (d ** 2)
                forces[i] += mag * tangential_i
                forces[j] -= mag * tangential_i
        
        # Pressure from worst facets
        if worst_idx is not None and len(centers) > 0:
            sorted_idx = np.argsort(radii)[::-1]
            n_worst = min(5, len(sorted_idx))
            
            for idx in sorted_idx[:n_worst]:
                if radii[idx] < 1e-10:
                    continue
                C = centers[idx]
                r = radii[idx]
                sin_r = np.sin(r)
                if sin_r < 1e-10:
                    continue
                
                weight = (radii[idx] / np.max(radii)) ** 2
                
                for i in range(n):
                    dot_ic = np.dot(current_points[i], C)
                    ang_dist = np.arccos(np.clip(dot_ic, -1.0, 1.0))
                    if ang_dist > r + 0.15:
                        continue
                    
                    radial_C = dot_ic * current_points[i]
                    tangential_C = C - radial_C
                    tangential_C_norm = np.linalg.norm(tangential_C)
                    if tangential_C_norm < 1e-10:
                        continue
                    tangential_C = tangential_C / tangential_C_norm
                    
                    strength = weight * max(0, 1 - ang_dist / (r + 0.15))
                    forces[i] += strength * tangential_C
        
        force_norm = np.linalg.norm(forces)
        if force_norm < 1e-14:
            break
        
        direction = forces / force_norm
        new_points = current_points + step_size * direction
        new_points = _project_to_sphere(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.05)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin from 150s limit

    best_points = None
    best_radius = np.inf
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Fibonacci sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts with different seeds
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777, 2024, 31415]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
    
    # 3. Icosahedron-based
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    rng_ico = np.random.default_rng(12345)
    extra = rng_ico.normal(size=(50 - len(ico_verts), 3))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    ico_seed = np.vstack([ico_verts, extra])
    seeds.append(ico_seed)
    
    n_seeds = len(seeds)
    
    # Stage 1: L-BFGS optimization for each seed with adaptive beta
    base_time_per_seed = 7.0
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 15:
            break
        
        if i < n_seeds - 1:
            time_budget = min(base_time_per_seed, remaining / (n_seeds - i))
        else:
            time_budget = min(12.0, remaining)
        
        if time_budget < 3.0:
            break
        
        # Apply L-BFGS stage with adaptive beta
        pts = _lbfgs_stage(seed_pts, time_budget, beta_schedule=[30.0, 50.0, 100.0])
        
        # Quick local polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            pts, rad = _local_polish(pts, min(4.0, remaining - 5), beta=50.0)
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Stage 2: Final polish on best candidate
    if best_points is not None:
        # Grid SLSQP polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 25:
            budget = min(30.0, remaining - 12)
            if budget > 8:
                pts, rad = _grid_slqp_polish(best_points, budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Repulsive polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 18:
            budget = min(20.0, remaining - 10)
            if budget > 5:
                pts, rad = _repulsive_polish(best_points, budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Fine local polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 12:
            budget = min(15.0, remaining - 8)
            if budget > 3:
                pts, rad = _local_polish(best_points, budget, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 6:
            budget = min(8.0, remaining - 4)
            if budget > 2:
                pts, rad = _local_polish(best_points, budget, beta=500.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
