"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    wall_r = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _radius_fast(centers):
    """Fast radius computation for internal use."""
    wall_r = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, beta=50.0):
    """
    Smooth approximation of the max common radius using log-sum-exp.
    Returns a smooth surrogate for the min of wall distances and half pairwise distances.
    """
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)  # shape (N, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_dists = dist / 2.0  # shape (N, N)
    
    # Combine all distances into one vector
    all_dists = np.concatenate([wall_dists.ravel(), pair_dists[np.triu_indices(N, 1)]])
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta * d))) / beta
    min_d = np.min(all_dists)
    shifted = -beta * (all_dists - min_d)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_min = min_d - log_sum_exp / beta
    
    return soft_min

def _soft_min_radius_grad(centers, beta=50.0):
    """
    Gradient of the soft-min radius with respect to centers.
    """
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)  # shape (N, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_dists = dist / 2.0  # shape (N, N)
    
    # Combine all distances
    all_dists = np.concatenate([wall_dists.ravel(), pair_dists[np.triu_indices(N, 1)]])
    
    min_d = np.min(all_dists)
    shifted = -beta * (all_dists - min_d)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Weights for soft min
    weights = exp_shifted / sum_exp
    
    # Gradient of soft_min w.r.t. each distance is -weight
    # Now compute gradient w.r.t. centers
    
    grad = np.zeros_like(centers)
    
    # Wall part: wall_dists[i,k] = 0.5 - |centers[i,k]|
    # d(soft_min)/d(centers[i,k]) = -weights_wall[i,k] * d(0.5 - |c_ik|)/d(c_ik)
    # = -weights_wall[i,k] * (-sign(c_ik))
    # = weights_wall[i,k] * sign(c_ik)
    
    wall_weights = weights[:N*2].reshape(N, 2)
    signs = np.sign(centers)
    signs[np.abs(centers) < 1e-10] = 0.0  # avoid sign(0) issues
    grad += wall_weights * signs
    
    # Pairwise part: pair_dists[i,j] = ||c_i - c_j|| / 2
    # d(soft_min)/d(c_i) = -weights_pair[i,j] * (c_i - c_j) / (2 * ||c_i - c_j||) * 2
    # Wait, let me redo this.
    # soft_min = min_d - (1/beta) * log(sum(exp(-beta*(d_k - min_d))))
    # d(soft_min)/d(d_k) = -exp(-beta*(d_k - min_d)) / sum(exp(-beta*(d_m - min_d)))
    # = -weights[k]
    
    # For pairwise: d_k = ||c_i - c_j||/2
    # d(d_k)/d(c_i) = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d(d_k)/d(c_j) = (c_j - c_i) / (2 * ||c_i - c_j||)
    
    # So d(soft_min)/d(c_i) += -weights_pair[i,j] * (c_i - c_j) / (2 * ||c_i - c_j||)
    # And d(soft_min)/d(c_j) += -weights_pair[i,j] * (c_j - c_i) / (2 * ||c_i - c_j||)
    
    # We need to map pair weights back to the pair indices
    # all_dists has: [wall_dists.ravel() (N*2 elements), pair_dists[triu] (N*(N-1)/2 elements)]
    pair_weights = weights[N*2:]
    
    # Get triu indices
    iu = np.triu_indices(N, 1)
    
    for idx in range(len(pair_weights)):
        w = pair_weights[idx]
        i, j = iu[0][idx], iu[1][idx]
        d_ij = dist[i, j]
        if d_ij > 1e-10:
            dir_vec = (centers[i] - centers[j]) / d_ij
            grad[i] -= w * dir_vec / 2.0
            grad[j] += w * dir_vec / 2.0
    
    return grad

def _slsqp_epigraph_polish(centers, r, time_limit=10.0):
    """
    SLSQP with epigraph formulation.
    x = [c_1x, c_1y, ..., c_Nx, c_Ny, t]
    Maximize t subject to:
      0.5 - |c_ik| >= t  (4N linear constraints)
      ||c_i - c_j|| >= 2t (N*(N-1)/2 nonlinear constraints)
    
    We use a smooth soft-min as the objective to guide the optimizer,
    but enforce hard constraints via SLSQP.
    """
    start_time = time.time()
    
    x0 = np.concatenate([centers.ravel(), [r]])
    
    def objective(x):
        c = x[:N*2].reshape(N, 2)
        return -_soft_min_radius(c, beta=50.0)
    
    def constraints_fun(x):
        c = x[:N*2].reshape(N, 2)
        t = x[-1]
        
        cons = []
        # Wall constraints: 0.5 - c[i,k] - t >= 0 and 0.5 + c[i,k] - t >= 0
        for i in range(N):
            for k in range(2):
                cons.append(0.5 - c[i, k] - t)
                cons.append(0.5 + c[i, k] - t)
        
        # Pairwise: ||c_i - c_j||^2 - 4t^2 >= 0
        iu = np.triu_indices(N, 1)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            cons.append(dx*dx + dy*dy - 4*t*t)
        
        return cons
    
    def constraints_jac(x):
        """
        Jacobian of constraints.
        Returns a 2D array of shape (num_constraints, len(x)).
        """
        c = x[:N*2].reshape(N, 2)
        t = x[-1]
        
        num_wall = 4 * N
        num_pair = N * (N - 1) // 2
        total = num_wall + num_pair
        
        jac = np.zeros((total, 2*N + 1))
        
        # Wall constraints
        for i in range(N):
            # 0.5 - c[i,0] - t >= 0: grad = [-1, 0, ..., 0, -1]
            row = 4*i
            jac[row, 2*i] = -1.0
            jac[row, -1] = -1.0
            
            # 0.5 + c[i,0] - t >= 0: grad = [1, 0, ..., 0, -1]
            row = 4*i + 1
            jac[row, 2*i] = 1.0
            jac[row, -1] = -1.0
            
            # 0.5 - c[i,1] - t >= 0: grad = [0, -1, ..., 0, -1]
            row = 4*i + 2
            jac[row, 2*i+1] = -1.0
            jac[row, -1] = -1.0
            
            # 0.5 + c[i,1] - t >= 0: grad = [0, 1, ..., 0, -1]
            row = 4*i + 3
            jac[row, 2*i+1] = 1.0
            jac[row, -1] = -1.0
        
        # Pairwise constraints
        iu = np.triu_indices(N, 1)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            row = num_wall + idx
            
            # d/dx_i = 2*dx, d/dy_i = 2*dy
            # d/dx_j = -2*dx, d/dy_j = -2*dy
            # d/dt = -8*t
            jac[row, 2*i] = 2*dx
            jac[row, 2*i+1] = 2*dy
            jac[row, 2*j] = -2*dx
            jac[row, 2*j+1] = -2*dy
            jac[row, -1] = -8*t
        
        return jac
    
    bounds = [(-0.49, 0.49)] * (2*N) + [(0.01, 0.5)]
    
    cons_slsqp = ({'type': 'ineq', 'fun': constraints_fun, 'jac': constraints_jac},)
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons_slsqp,
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )
        
        if res.x is not None and np.all(np.isfinite(res.x)):
            new_centers = res.x[:N*2].reshape(N, 2)
            new_r = res.x[-1]
            # Verify feasibility
            actual_r = _radius(new_centers)
            if actual_r >= new_r - 1e-6 and actual_r > r - 1e-6:
                return new_centers, actual_r
            else:
                # Return the better of the two
                if actual_r > r:
                    return new_centers, actual_r
                else:
                    return centers, r
                    
    except Exception:
        pass
        
    return centers, r

def _voronoi_style_refine(centers, r, time_limit=3.0):
    """
    Voronoi-style refinement: move each circle to the center of its Voronoi cell
    (clipped to the square and avoiding overlaps). This is a local improvement heuristic.
    """
    from scipy.spatial import Voronoi, QhullError
    
    start_time = time.time()
    current = centers.copy()
    current_r = r
    
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Voronoi diagram
        try:
            vor = Voronoi(current)
        except QhullError:
            break
        
        # For each point, find the centroid of its Voronoi cell
        # that is within the square and doesn't cause overlap
        new_centers = current.copy()
        
        for i in range(N):
            # Get the Voronoi region for point i
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0:
                continue
            
            # Get the vertices of this Voronoi cell
            vertices = vor.vertices[region]
            
            # Filter vertices that are within the square
            valid = (np.abs(vertices) <= 0.5)
            if np.sum(valid) < 3:
                # Not enough valid vertices, use a smaller move
                target = current[i]
            else:
                valid_vertices = vertices[valid]
                target = np.mean(valid_vertices, axis=0)
            
            # Clip target to stay within square with margin
            margin = current_r * 0.5
            target = np.clip(target, -0.5 + margin, 0.5 - margin)
            
            # Limit the step size
            step = target - current[i]
            step_len = np.linalg.norm(step)
            max_step = 0.02
            if step_len > max_step:
                step = step / step_len * max_step
            
            new_centers[i] = current[i] + step
        
        # Check if the new configuration is valid
        new_r = _radius(new_centers)
        
        if new_r >= current_r - 1e-8:
            current = new_centers
            current_r = new_r
        else:
            break
    
    return current, current_r

def _grow_radius(centers, r, time_limit=5.0):
    """
    Incrementally grow the radius by resolving collisions.
    """
    start_time = time.time()
    current_r = r
    current_centers = centers.copy()
    
    step_size = 0.002
    max_failures = 3
    failures = 0
    
    while time.time() - start_time < time_limit:
        new_r = current_r + step_size
        
        # Try to find a feasible configuration for new_r
        # Use a few iterations of force-directed relaxation
        trial = current_centers.copy()
        
        success = False
        for inner_iter in range(20):
            # Compute forces
            forces = np.zeros_like(trial)
            
            # Pairwise repulsion
            diff = trial[:, np.newaxis, :] - trial[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, 1.0)
            
            too_close = dist < 2 * new_r
            overlap = np.maximum(0, 2 * new_r - dist)
            norm_diff = diff / dist[:, :, np.newaxis]
            
            for i in range(N):
                for j in range(N):
                    if i != j and too_close[i, j]:
                        forces[i] += norm_diff[i, j] * overlap[i, j]
            
            # Wall repulsion
            for i in range(N):
                for k in range(2):
                    if trial[i, k] > 0.5 - new_r:
                        forces[i, k] -= (trial[i, k] - (0.5 - new_r))
                    elif trial[i, k] < -(0.5 - new_r):
                        forces[i, k] += (-(0.5 - new_r) - trial[i, k])
            
            # Move
            trial += 0.05 * forces
            trial = np.clip(trial, -0.5 + new_r, 0.5 - new_r)
            
            # Check feasibility
            diff = trial[:, np.newaxis, :] - trial[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_d = np.min(dist)
            
            if min_d >= 2 * new_r - 1e-9 and np.all(np.abs(trial) <= 0.5 - new_r + 1e-9):
                success = True
                break
        
        if success:
            current_r = new_r
            current_centers = trial
            failures = 0
        else:
            failures += 1
            if failures >= max_failures:
                break
            step_size *= 0.5
    
    return current_centers, current_r

def _make_hexagonal_grid(n_x, n_y, spacing, offset_x=0, offset_y=0):
    """Create a hexagonal lattice of points."""
    points = []
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing + offset_x
            y = j * dy + offset_y
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    return points[:N]

def _init_7x7_plus():
    """
    Initialize with a 7x7 grid plus 1 extra point, totaling 50.
    7*7 = 49, plus 1 = 50.
    """
    # 7x7 grid
    xs = np.linspace(-0.4, 0.4, 7)
    ys = np.linspace(-0.4, 0.4, 7)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    
    # Add one more point at center
    points = np.vstack([points, [[0.0, 0.0]]])
    
    return points[:N]

def _init_jittered_8x8():
    """
    Initialize with a jittered 8x8 grid, taking 50 points.
    """
    np.random.seed(123)
    xs = np.linspace(-0.45, 0.45, 8)
    ys = np.linspace(-0.45, 0.45, 8)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    
    # Add small jitter
    points += np.random.uniform(-0.01, 0.01, size=points.shape)
    
    return points[:N]

def _init_hexagonal():
    """Initialize with a hexagonal pattern."""
    c = _make_hexagonal_grid(10, 10, 0.15, offset_x=-0.5, offset_y=-0.5)
    c = c - np.mean(c, axis=0)
    max_ext = np.max(np.abs(c))
    if max_ext > 0.45:
        c = c * (0.45 / max_ext)
    return c

def _init_random():
    """Random initialization."""
    np.random.seed(42)
    return np.random.uniform(-0.45, 0.45, size=(N, 2))

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    inits.append(_init_hexagonal())
    inits.append(_init_7x7_plus())
    inits.append(_init_jittered_8x8())
    inits.append(_init_random())
    
    # Refine each initial configuration
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        time_left = DEADLINE - 10 - (time.time() - start_time)
        alloc_time = min(5.0, time_left / max(1, len(inits) - i))
        
        # Step 1: Grow radius from a small initial value
        r0 = 0.05
        refined_centers, refined_r = _grow_radius(init, r0, time_limit=alloc_time * 0.4)
        
        # Step 2: Voronoi-style refinement
        if time.time() - start_time < DEADLINE - 5:
            time_left_voronoi = min(alloc_time * 0.2, DEADLINE - 5 - (time.time() - start_time))
            if time_left_voronoi > 0:
                voronoi_centers, voronoi_r = _voronoi_style_refine(refined_centers, refined_r, time_limit=time_left_voronoi)
                if voronoi_r > refined_r:
                    refined_centers = voronoi_centers
                    refined_r = voronoi_r
        
        # Step 3: Polish with SLSQP epigraph
        if time.time() - start_time < DEADLINE - 5:
            time_left_polish = min(alloc_time * 0.4, DEADLINE - 5 - (time.time() - start_time))
            if time_left_polish > 0:
                polished_centers, polished_r = _slsqp_epigraph_polish(refined_centers, refined_r, time_limit=time_left_polish)
                refined_centers = polished_centers
                refined_r = polished_r
        
        # Verify and update best
        actual_r = _radius(refined_centers)
        if actual_r > best_r:
            best_r = actual_r
            best_centers = refined_centers.copy()
            
    # If time remains, do more random restarts
    restart_count = 0
    while time.time() - start_time < DEADLINE - 5:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left < 2: break
        
        alloc_time = min(3.0, time_left * 0.9)
        
        # Step 1: Grow radius
        refined_centers, refined_r = _grow_radius(c_rand, 0.05, time_limit=alloc_time * 0.4)
        
        # Step 2: Voronoi refinement
        if time.time() - start_time < DEADLINE - 5:
            time_left_voronoi = min(alloc_time * 0.2, DEADLINE - 5 - (time.time() - start_time))
            if time_left_voronoi > 0:
                voronoi_centers, voronoi_r = _voronoi_style_refine(refined_centers, refined_r, time_limit=time_left_voronoi)
                if voronoi_r > refined_r:
                    refined_centers = voronoi_centers
                    refined_r = voronoi_r
        
        # Step 3: SLSQP epigraph polish
        if time.time() - start_time < DEADLINE - 5:
            time_left_polish = min(alloc_time * 0.4, DEADLINE - 5 - (time.time() - start_time))
            if time_left_polish > 0:
                polished_centers, polished_r = _slsqp_epigraph_polish(refined_centers, refined_r, time_limit=time_left_polish)
                refined_centers = polished_centers
                refined_r = polished_r
        
        actual_r = _radius(refined_centers)
        if actual_r > best_r:
            best_r = actual_r
            best_centers = refined_centers.copy()
        
        restart_count += 1
    
    # Final high-precision polish if we have time
    if best_centers is not None and time.time() - start_time < DEADLINE - 3:
        time_left_final = DEADLINE - 3 - (time.time() - start_time)
        if time_left_final > 1:
            final_centers, final_r = _slsqp_epigraph_polish(best_centers, best_r, time_limit=time_left_final)
            actual_r = _radius(final_centers)
            if actual_r > best_r:
                best_r = actual_r
                best_centers = final_centers.copy()

    return best_centers
