"""
Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle.
Strategy:
1. Sequential Convex Programming (SCP): iteratively solve LP to maximize t subject to
   linearized area constraints for the active set of smallest triangles.
2. Genetic algorithm (GA) for diverse initializations.
3. Final polishing with L-BFGS-B and Nelder-Mead on the raw min-area objective.
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.optimize import NonlinearConstraint

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
# Number of triples is C(11, 3) = 165
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]
M = len(_IDX_TRIPLES)  # 165


def _min_triangle_area_vectorized(points):
    """
    Compute the smallest area among all triangles formed by triples of points, vectorized.
    points: (N, 2) array
    Returns: float min area
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # Cross product magnitude for 2D vectors
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
        
    return np.min(areas)

def _all_triangle_areas(points):
    """
    Compute all triangle areas, vectorized.
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    Point = lambda_1 * V1 + lambda_2 * V2 (since V0 is origin)
    """
    s = params[0::2]
    t = params[1::2]
    
    # Clip to avoid numerical issues at boundaries
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    
    return np.column_stack([x, y])


def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Inverse barycentric transformation
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params


def _build_lp_matrix(active_triples):
    """
    Build the LP constraint matrix for linearized area constraints.
    For each active triple (i,j,k), we linearize |cross| >= 2*t.
    
    The cross product for points p_i, p_j, p_k is:
    cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    
    We need to handle the absolute value. At the current solution, we know the sign
    of the cross product. We linearize around the current point.
    
    For a given sign sigma (+1 or -1), the constraint is:
    sigma * cross >= 2*t
    
    This is linear in the parameters (s_i, t_i) since each coordinate is linear in s,t.
    
    Returns: A matrix (m, 2N+1) and b vector (m,) such that A @ [params; t] >= b
    Actually, linprog uses A_ub @ x <= b_ub, so we negate:
    -A @ [params; t] <= -b
    """
    # For each active triple, we need to know the sign of the cross product
    # at the current solution. We'll pass this in.
    pass  # This function will be replaced with a more complete version

def _solve_lp(params, active_triples, deadline):
    """
    Solve the LP: maximize t subject to linearized area constraints for active triples.
    
    Variables: x = [s_1, t_1, s_2, t_2, ..., s_N, t_N, t_area]
    Total: 2*N + 1 variables
    
    Constraints: For each active triple m with sign sigma_m:
    sigma_m * cross_m(params) >= 2 * t_area
    
    The cross product is bilinear in the point coordinates, which are linear in (s,t).
    So cross = sum of terms like s_i * t_j * (coefficients).
    
    Wait - cross is NOT linear in (s,t) because it's a product of two linear terms.
    (x_j - x_i)(y_k - y_i) where x_j = s_j(1-t_j) + 0.5*s_j*t_j, etc.
    
    This means the constraint is quadratic, not linear. We need to linearize it.
    
    Linearization: At the current point p*, the cross product is:
    cross(p) ≈ cross(p*) + grad_cross(p*) @ (p - p*)
    
    Where p = (x_1, y_1, ..., x_N, y_N) and each coordinate is linear in (s,t).
    
    So: cross(p) ≈ cross(p*) + sum_i [dx_i/ds_i * (s_i - s_i*) * d_cross/dx_i + dx_i/dt_i * (t_i - t_i*) * d_cross/dx_i + ...]
    
    This is linear in (s_i, t_i).
    """
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # Compute cross products and signs at current solution
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Get active triples
    m_active = len(active_triples)
    if m_active == 0:
        return params, 0.0
    
    # Determine sign for each active triple
    signs = np.zeros(m_active)
    for idx, m in enumerate(active_triples):
        signs[idx] = np.sign(cross[m])
        if np.abs(cross[m]) < 1e-15:
            signs[idx] = 1.0  # Default to positive
    
    # Build linearized constraints
    # Variable vector: [s_0, t_0, s_1, t_1, ..., s_{N-1}, t_{N-1}, t_area]
    # Total variables: 2*N + 1
    
    n_vars = 2 * N + 1
    
    # Objective: maximize t_area => minimize -t_area
    c = np.zeros(n_vars)
    c[-1] = -1.0  # Minimize -t_area
    
    # Constraints: sigma_m * cross_linearized >= 2 * t_area
    # => sigma_m * (cross_m* + grad @ delta) - 2 * t_area >= 0
    # => -sigma_m * (cross_m* + grad @ delta) + 2 * t_area <= 0
    
    A_ub = np.zeros((m_active, n_vars))
    b_ub = np.zeros(m_active)
    
    for idx, m in enumerate(active_triples):
        i, j, k = _IDX_TRIPLES[m]
        
        # Current cross value
        cross_val = cross[m]
        sigma = signs[idx]
        
        # Gradient of cross with respect to each point's coordinates
        # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        
        # d_cross/dx_i = -(y_k - y_i) + (y_j - y_i)
        # d_cross/dy_i = -(x_k - x_i) + (x_j - x_i)
        # d_cross/dx_j = (y_k - y_i)
        # d_cross/dy_j = -(x_k - x_i)
        # d_cross/dx_k = -(y_j - y_i)
        # d_cross/dy_k = (x_j - x_i)
        
        dx_i = -(pk[k, 1] - pi[i, 1]) + (pj[j, 1] - pi[i, 1])
        dy_i = -(pk[k, 0] - pi[i, 0]) + (pj[j, 0] - pi[i, 0])
        dx_j = pk[k, 1] - pi[i, 1]
        dy_j = -(pk[k, 0] - pi[i, 0])
        dx_k = -(pj[j, 1] - pi[i, 1])
        dy_k = pj[j, 0] - pi[i, 0]
        
        # Now, each coordinate is a function of (s, t):
        # x = s*(1-t) + 0.5*s*t = s - 0.5*s*t
        # y = 0 + 0.5*sqrt(3)*s*t = (sqrt(3)/2)*s*t
        
        # dx/ds = (1-t) + 0.5*t = 1 - 0.5*t
        # dx/dt = -s + 0.5*s = -0.5*s
        # dy/ds = (sqrt(3)/2)*t
        # dy/dt = (sqrt(3)/2)*s
        
        s_i, t_i = params[2*i], params[2*i+1]
        s_j, t_j = params[2*j], params[2*j+1]
        s_k, t_k = params[2*k], params[2*k+1]
        
        # Gradient of cross w.r.t. s_i:
        # = dx_i * (1 - 0.5*t_i) + dy_i * (sqrt(3)/2 * t_i)
        # Gradient of cross w.r.t. t_i:
        # = dx_i * (-0.5*s_i) + dy_i * (sqrt(3)/2 * s_i)
        
        sqrt3_half = np.sqrt(3) / 2
        
        d_cross_ds_i = dx_i * (1 - 0.5 * t_i) + dy_i * (sqrt3_half * t_i)
        d_cross_dt_i = dx_i * (-0.5 * s_i) + dy_i * (sqrt3_half * s_i)
        
        d_cross_ds_j = dx_j * (1 - 0.5 * t_j) + dy_j * (sqrt3_half * t_j)
        d_cross_dt_j = dx_j * (-0.5 * s_j) + dy_j * (sqrt3_half * s_j)
        
        d_cross_ds_k = dx_k * (1 - 0.5 * t_k) + dy_k * (sqrt3_half * t_k)
        d_cross_dt_k = dx_k * (-0.5 * s_k) + dy_k * (sqrt3_half * s_k)
        
        # Linearized cross: cross_val + d_cross_ds_i*(s_i - s_i*) + ... 
        # = cross_val + d_cross_ds_i*s_i + d_cross_dt_i*t_i + ... - (d_cross_ds_i*s_i* + d_cross_dt_i*t_i* + ...)
        
        const = cross_val - (d_cross_ds_i * s_i + d_cross_dt_i * t_i + 
                            d_cross_ds_j * s_j + d_cross_dt_j * t_j +
                            d_cross_ds_k * s_k + d_cross_dt_k * t_k)
        
        # Constraint: sigma * (const + d_cross_ds_i*s_i + d_cross_dt_i*t_i + ...) >= 2*t_area
        # => -sigma * (const + d_cross_ds_i*s_i + ...) + 2*t_area <= 0
        
        A_ub[idx, 2*i] = -sigma * d_cross_ds_i
        A_ub[idx, 2*i+1] = -sigma * d_cross_dt_i
        A_ub[idx, 2*j] = -sigma * d_cross_ds_j
        A_ub[idx, 2*j+1] = -sigma * d_cross_dt_j
        A_ub[idx, 2*k] = -sigma * d_cross_ds_k
        A_ub[idx, 2*k+1] = -sigma * d_cross_dt_k
        A_ub[idx, -1] = 2.0
        b_ub[idx] = -sigma * const
    
    # Bounds: s_i, t_i in [0, 1], t_area >= 0
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, None)]
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            new_params = res.x[:2*N]
            new_t = res.x[-1]
            return new_params, new_t
        else:
            return params, 0.0
    except Exception:
        return params, 0.0


def _scp_iteration(params, max_active=30, deadline=None):
    """
    One iteration of Sequential Convex Programming.
    
    1. Evaluate all triangle areas at current params.
    2. Select the smallest 'max_active' triangles as the active set.
    3. Solve LP to maximize t subject to linearized constraints for active set.
    4. Return updated params.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    # Select active set: smallest triangles
    if len(areas) <= max_active:
        active_triples = list(range(len(areas)))
    else:
        # Get indices of smallest max_active areas
        active_indices = np.argsort(areas)[:max_active]
        active_triples = list(active_indices)
    
    new_params, t_val = _solve_lp(params, active_triples, deadline)
    return new_params

def _run_scp(params, n_iterations=50, deadline=None, max_active=30):
    """
    Run SCP for multiple iterations.
    """
    current = params.copy()
    best_params = current.copy()
    best_score = _min_triangle_area_vectorized(_param_to_points(current))
    
    for it in range(n_iterations):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        new_params = _scp_iteration(current, max_active=max_active, deadline=deadline)
        
        # Check if we improved
        score = _min_triangle_area_vectorized(_param_to_points(new_params))
        if score > best_score:
            best_score = score
            best_params = new_params.copy()
            current = new_params.copy()
        else:
            # No improvement, try with more active constraints
            current = new_params.copy()
            # If still no improvement after a few iterations, break
            if it > 5 and score < best_score * 0.99:
                break
    
    return best_params

def _soft_min_objective_and_grad(params, beta):
    """
    Objective: negative soft-min of triangle areas.
    Gradient: computed via chain rule.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    # Numerical stability for log-sum-exp
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Objective value
    obj = -(min_a - np.log(sum_exp) / beta)
    
    # Gradient
    weights = exp_shifted / sum_exp
    
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    sign_cross = np.sign(cross)
    sign_cross[np.abs(cross) < 1e-15] = 0.0
    
    d_cross_pi = np.column_stack([
        -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1]),
        -(pj[:, 0] - pi[:, 0]) + (pk[:, 0] - pi[:, 0])
    ])
    d_cross_pj = np.column_stack([
        pk[:, 1] - pi[:, 1],
        -(pk[:, 0] - pi[:, 0])
    ])
    d_cross_pk = np.column_stack([
        -(pj[:, 1] - pi[:, 1]),
        pj[:, 0] - pi[:, 0]
    ])
    
    grad_points = np.zeros((N, 2))
    w = weights * sign_cross * 0.5
    
    np.add.at(grad_points, _IDX_I, w[:, np.newaxis] * d_cross_pi)
    np.add.at(grad_points, _IDX_J, w[:, np.newaxis] * d_cross_pj)
    np.add.at(grad_points, _IDX_K, w[:, np.newaxis] * d_cross_pk)
    
    s = params[0::2]
    t = params[1::2]
    
    dx_ds = (1 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    return obj, -grad_params

def _raw_objective(params):
    """Objective for final polishing: negative min area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _generate_fps_initialization(rng, n_points=N, n_candidates=300):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def _structured_init():
    """Generate a good structured initialization."""
    pts = np.zeros((N, 2))
    pts[0] = V0 + 0.05 * (V1 + V2 - 2*V0)
    pts[1] = V1 + 0.05 * (V0 + V2 - 2*V1)
    pts[2] = V2 + 0.05 * (V0 + V1 - 2*V2)
    pts[3] = (V0 + V1) / 2.0
    pts[4] = (V1 + V2) / 2.0
    pts[5] = (V2 + V0) / 2.0
    centroid = (V0 + V1 + V2) / 3.0
    pts[6] = centroid
    pts[7] = centroid + 0.15 * (V0 - centroid)
    pts[8] = centroid + 0.15 * (V1 - centroid)
    pts[9] = centroid + 0.15 * (V2 - centroid)
    pts[10] = centroid + 0.08 * (V0 - centroid)
    return _points_to_param(pts)

def _polish(params, deadline):
    """
    Final polishing with L-BFGS-B and Nelder-Mead.
    """
    if time.time() > deadline - 5.0:
        return params
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # L-BFGS-B polish
    try:
        res = minimize(
            _raw_objective, params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.x is not None:
            params = res.x
    except Exception:
        pass
    
    # Nelder-Mead polish
    if time.time() > deadline - 2.0:
        return params
    
    try:
        res = minimize(
            _raw_objective, params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        if res.x is not None:
            params = np.clip(res.x, 0.0, 1.0)
    except Exception:
        pass
    
    return params

def _ga_step(population, rng, elite_size=10, mutation_rate=0.15, crossover_rate=0.8):
    """
    One step of genetic algorithm.
    population: list of params arrays (2N each)
    """
    n = len(population)
    
    # Evaluate fitness (min area)
    fitness = np.array([_min_triangle_area_vectorized(_param_to_points(p)) for p in population])
    
    # Sort by fitness (descending - we want max min area)
    sorted_idx = np.argsort(fitness)[::-1]
    sorted_pop = [population[i] for i in sorted_idx]
    sorted_fit = fitness[sorted_idx]
    
    # Elitism: keep top elite_size
    new_pop = [sorted_pop[i].copy() for i in range(min(elite_size, n))]
    
    # Tournament selection + crossover + mutation
    while len(new_pop) < n:
        # Select two parents via tournament
        idx1 = rng.choice(n, size=elite_size)
        idx2 = rng.choice(n, size=elite_size)
        parent1 = sorted_pop[idx1[np.argmax(fitness[idx1])]]
        parent2 = sorted_pop[idx2[np.argmax(fitness[idx2])]]
        
        # Crossover
        if rng.random() < crossover_rate:
            # Uniform crossover
            mask = rng.random(2 * N) > 0.5
            child = np.where(mask, parent1, parent2)
        else:
            child = parent1.copy()
        
        # Mutation
        if rng.random() < mutation_rate:
            # Gaussian mutation
            sigma = 0.05
            child = child + sigma * rng.randn(2 * N)
            child = np.clip(child, 0.0, 1.0)
        
        new_pop.append(child)
    
    return new_pop

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf

    rng = np.random.RandomState(42)

    # ===== Phase 1: Generate diverse initial solutions =====
    candidates = []

    # Structured init
    candidates.append(_structured_init())

    # FPS inits
    for i in range(5):
        pts = _generate_fps_initialization(rng)
        candidates.append(_points_to_param(pts))

    # Random inits
    for i in range(5):
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        candidates.append(params)

    # ===== Phase 2: Run SCP from each candidate =====
    for idx, cand in enumerate(candidates):
        if time_left() < 10.0:
            break

        # Run SCP
        scp_result = _run_scp(cand, n_iterations=30, deadline=deadline, max_active=20)
        score = _min_triangle_area_vectorized(_param_to_points(scp_result))
        if score > best_score:
            best_score = score
            best_params = scp_result.copy()

        # Also try with more active constraints
        if time_left() > 5.0:
            scp_result2 = _run_scp(cand, n_iterations=20, deadline=deadline, max_active=50)
            score2 = _min_triangle_area_vectorized(_param_to_points(scp_result2))
            if score2 > best_score:
                best_score = score2
                best_params = scp_result2.copy()

    # ===== Phase 3: GA to explore more of the space =====
    if time_left() > 30.0 and best_params is not None:
        # Use best SCP result as seed for GA
        population = [best_params.copy()]
        
        # Add some FPS and random inits
        for i in range(4):
            pts = _generate_fps_initialization(rng)
            population.append(_points_to_param(pts))
        
        for i in range(4):
            s = rng.uniform(0.1, 0.9, size=N)
            t = rng.uniform(0.1, 0.9, size=N)
            params = np.zeros(2 * N)
            params[0::2] = s
            params[1::2] = t
            population.append(params)

        n_generations = 20
        for gen in range(n_generations):
            if time_left() < 10.0:
                break

            population = _ga_step(population, rng, elite_size=8, mutation_rate=0.15, crossover_rate=0.8)

            # Track best
            for p in population:
                score = _min_triangle_area_vectorized(_param_to_points(p))
                if score > best_score:
                    best_score = score
                    best_params = p.copy()

            # Periodic SCP polish of the best
            if gen % 4 == 3 and best_params is not None:
                if time_left() > 10.0:
                    polished = _run_scp(best_params, n_iterations=20, deadline=deadline, max_active=20)
                    score = _min_triangle_area_vectorized(_param_to_points(polished))
                    if score > best_score:
                        best_score = score
                        best_params = polished.copy()
                        population[0] = best_params.copy()

    # ===== Phase 4: Final polish =====
    if time_left() > 5.0 and best_params is not None:
        # L-BFGS-B with soft-min annealing
        bounds = [(0.0, 1.0)] * (2 * N)
        betas = [10.0, 30.0, 100.0]

        current = best_params.copy()
        for beta in betas:
            if time_left() < 5.0:
                break

            def obj_grad(params):
                return _soft_min_objective_and_grad(params, beta)

            try:
                res = minimize(
                    obj_grad, current,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    current = res.x
            except Exception:
                pass

        # SCP polish
        if time_left() > 5.0:
            current = _run_scp(current, n_iterations=20, deadline=deadline, max_active=30)

        # Final polish
        if time_left() > 3.0:
            current = _polish(current, deadline)

        score = _min_triangle_area_vectorized(_param_to_points(current))
        if score > best_score:
            best_score = score
            best_params = current.copy()

    # ===== Phase 5: Additional local starts from GA population =====
    if time_left() > 15.0 and best_params is not None:
        # Take top candidates and polish each with SCP
        if 'population' in locals() and len(population) > 0:
            fitness = np.array([_min_triangle_area_vectorized(_param_to_points(p)) for p in population])
            top_idx = np.argsort(fitness)[::-1][:3]

            for idx in top_idx:
                if time_left() < 5.0:
                    break

                current = population[idx].copy()
                current = _run_scp(current, n_iterations=15, deadline=deadline, max_active=25)

                score = _min_triangle_area_vectorized(_param_to_points(current))
                if score > best_score:
                    best_score = score
                    best_params = current.copy()

    if best_params is None:
        # Fallback
        return np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, np.sqrt(3)/4], [0.75, np.sqrt(3)/4], [0.5, np.sqrt(3)/2],
            [0.5, np.sqrt(3)/8], [0.25, np.sqrt(3)/8], [0.75, np.sqrt(3)/8],
            [0.5, 3*np.sqrt(3)/8], [0.5, np.sqrt(3)/4]
        ])

    best_points = _param_to_points(best_params)

    return best_points
