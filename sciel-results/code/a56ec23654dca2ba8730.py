import numpy as np
import scipy.spatial
import time

N = 50

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii.
    Returns: hull, centers (array), radii (array), worst_idx
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
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

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max covering radius and gradient w.r.t. points on the sphere.
    The gradient is computed via the analytic derivative of the circumradius
    with respect to each vertex, weighted by the soft-max weights.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)

    n_facets = len(hull.simplices)
    if n_facets == 0:
        return np.pi, np.zeros_like(points)

    max_r = np.max(radii)
    # Avoid overflow
    exp_terms = np.exp(beta * (radii - max_r))
    sum_exp = np.sum(exp_terms)
    if sum_exp < 1e-300:
        weights = np.ones(n_facets) / n_facets
    else:
        weights = exp_terms / sum_exp

    soft_radius = max_r + np.log(sum_exp) / beta

    grads = np.zeros_like(points)

    for i in range(n_facets):
        if radii[i] < 1e-10 or weights[i] < 1e-14:
            continue

        w = weights[i]
        C = centers[i]
        r = radii[i]
        simplex = hull.simplices[i]

        # Gradient of circumradius w.r.t. vertex p_k:
        # dR/dp_k = -C / sin(R) projected to tangent space
        # Actually, the circumcenter C is the point equidistant from all 3 vertices.
        # R = arccos(p_k . C)
        # dR/dp_k = -C / sqrt(1 - (p_k.C)^2) = -C / sin(R)
        # But we need the tangent component: grad_t = grad - (grad.p_k)p_k
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue

        grad_vec = -C / sin_r

        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            # Project gradient to tangent space
            proj = np.dot(grad_vec, p) * p
            tangent_grad = grad_vec - proj
            grads[p_idx] += w * tangent_grad

    return soft_radius, grads

def _repulsive_force(points, k_rep=5.0, d_min=0.05):
    """
    Compute repulsive force on each point from all other points.
    """
    n = len(points)
    forces = np.zeros_like(points)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = np.arccos(np.clip(np.dot(points[i], points[j]), -1.0, 1.0))
            d_eff = max(d, d_min)
            proj = np.dot(points[i], points[j]) * points[j]
            tangent = points[i] - proj
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-10:
                tangent = tangent / norm_t
            else:
                continue
            force_mag = k_rep / (d_eff * d_eff)
            forces[i] += force_mag * tangent

    return forces

def _local_search_softmax(points, time_budget, beta=50.0):
    """
    Local search using soft-max radius gradient descent.
    """
    current_points = points.copy()
    start_time = time.time()

    current_radius, _ = _soft_max_radius_and_grad(current_points, beta)
    best_radius = current_radius
    best_points = current_points.copy()

    step_size = 0.05
    min_step = 1e-7
    decay = 0.9
    increase = 1.15

    while time.time() - start_time < time_budget:
        soft_r, grads = _soft_max_radius_and_grad(current_points, beta)

        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-12:
            break

        # Normalize gradient to unit length
        grads = grads / grad_norm

        new_points = current_points + step_size * grads
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)

        # Evaluate hard radius for stability
        new_hull, new_centers, new_radii, new_worst = _compute_facet_data(new_points)
        new_radius = np.max(new_radii) if new_worst is not None else np.pi

        if new_radius < current_radius:
            current_points = new_points
            current_radius = new_radius
            step_size = min(step_size * increase, 0.2)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            step_size *= decay
            if step_size < min_step:
                break

    return best_points, best_radius

def _local_search_repulsive(points, time_budget, k_rep=5.0, k_press=1.0, top_k=5):
    """
    Local search using combined repulsive/pressure forces.
    """
    current_points = points.copy()
    start_time = time.time()

    hull, centers, radii, worst_idx = _compute_facet_data(current_points)
    current_radius = np.max(radii) if worst_idx is not None else np.pi
    best_radius = current_radius
    best_points = current_points.copy()

    step_size = 0.05
    min_step = 1e-6
    decay = 0.9
    increase = 1.1

    while time.time() - start_time < time_budget:
        # Compute repulsive force
        f_rep = _repulsive_force(current_points, k_rep=k_rep)

        # Compute pressure towards worst facet centers
        f_press = np.zeros_like(current_points)
        if hull is not None and worst_idx is not None:
            sorted_indices = np.argsort(radii)[::-1]
            count = 0
            max_r = np.max(radii)
            for idx in sorted_indices:
                if count >= top_k:
                    break
                if radii[idx] < 1e-6:
                    continue
                simplex = hull.simplices[idx]
                C = centers[idx]
                weight = radii[idx] / max_r if max_r > 0 else 0

                for k in range(3):
                    p_idx = simplex[k]
                    p = current_points[p_idx]
                    proj = np.dot(p, C) * C
                    tangent = C - proj
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > 1e-10:
                        tangent = tangent / norm_t
                    else:
                        continue
                    f_press[p_idx] += k_press * weight * tangent
                count += 1

        forces = f_rep + f_press
        force_norm = np.linalg.norm(forces)
        if force_norm < 1e-10:
            break
        forces = forces / force_norm

        new_points = current_points + step_size * forces
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)

        new_hull, new_centers, new_radii, new_worst = _compute_facet_data(new_points)
        new_radius = np.max(new_radii) if new_worst is not None else np.pi

        if new_radius < current_radius:
            current_points = new_points
            current_radius = new_radius
            step_size = min(step_size * increase, 0.2)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            step_size *= decay
            if step_size < min_step:
                break

    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_points = None
    best_radius = np.inf

    seeds = []

    # 1. Fibonacci Sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)

    # 2. Random starts with various seeds
    for seed_val in [42, 1, 2, 3, 7, 13, 100, 200]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)

    n_seeds = len(seeds)

    # Phase 1: Soft-max gradient local search on all seeds
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 10:
            break

        if i < n_seeds - 1:
            time_budget = min(10.0, remaining / (n_seeds - i))
        else:
            time_budget = min(25.0, remaining)

        if time_budget < 2.0:
            break

        pts, rad = _local_search_softmax(seed_pts, time_budget=time_budget, beta=50.0)
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()

    # Phase 2: Repulsive force polish on the best result
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            budget = min(30.0, remaining - 2)
            if budget > 2:
                pts, rad = _local_search_repulsive(best_points, time_budget=budget, k_rep=5.0, k_press=1.0, top_k=5)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()

    # Phase 3: Final soft-max polish with different beta
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            budget = min(25.0, remaining - 2)
            if budget > 2:
                pts, rad = _local_search_softmax(best_points, time_budget=budget, beta=100.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()

    # Phase 4: Final repulsive polish with different parameters
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            budget = min(15.0, remaining - 2)
            if budget > 2:
                pts, rad = _local_search_repulsive(best_points, time_budget=budget, k_rep=10.0, k_press=0.5, top_k=10)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()

    if best_points is None:
        best_points = seeds[0]

    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    best_points = best_points / norms

    return best_points
