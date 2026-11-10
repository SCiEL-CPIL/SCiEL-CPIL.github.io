import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 130.0  # Safety margin from 150s

def _compute_radius(centers):
    """
    Compute the exact minimum radius for a set of centers.
    centers: (N, 2) array.
    Returns radius and also identifies the binding constraint.
    """
    if centers.shape != (N, 2):
        return 0.0, None, None
    if not np.all(np.isfinite(centers)):
        return 0.0, None, None
    
    # Check boundaries
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    boundary_dists = np.minimum(dx, dy)
    min_boundary = np.min(boundary_dists)
    idx_boundary = np.argmin(boundary_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    flat_idx = np.argmin(dists)
    i, j = np.unravel_index(flat_idx, dists.shape)
    
    r = min(min_boundary, min_pairwise)
    
    # Determine which constraint is binding
    if min_boundary <= min_pairwise:
        binding = 'boundary'
        binding_info = idx_boundary
    else:
        binding = 'pair'
        binding_info = (i, j)
    
    return r, binding, binding_info

def _soft_min(z, beta):
    """
    Compute the soft minimum of a vector z.
    """
    z_min = np.min(z)
    z_shifted = z - z_min
    terms = np.exp(-beta * z_shifted)
    log_sum = np.log(np.sum(terms))
    return z_min - (1.0 / beta) * log_sum

def _objective(x, beta):
    """
    Objective function to minimize: -soft_min(distances).
    x: flat array of length 2*N.
    """
    centers = x.reshape(N, 2)
    
    # Distances to boundary
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    boundary_dists = np.minimum(dx, dy)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pairwise_dists = dists / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([pairwise_dists.flatten(), boundary_dists])
    
    # Soft min
    r_soft = _soft_min(all_dists, beta)
    
    return -r_soft

def _hexagonal_init(jitter=0.03):
    """
    Generate a properly scaled hexagonal lattice initialization.
    """
    # Generate hex grid points
    s = 1.0
    points = []
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    
    # Select N points closest to center
    dists_to_center = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    points = points[indices].copy()
    
    # Scale to fit in [-0.45, 0.45] (leave margin for optimization)
    max_abs = np.max(np.abs(points))
    if max_abs > 0:
        scale = 0.45 / max_abs
        points = points * scale
    
    # Jitter
    points += np.random.normal(0, jitter, size=points.shape)
    
    # Project to square
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    return points

def _random_init():
    """
    Random initialization with repulsion relaxation.
    """
    points = np.random.uniform(-0.45, 0.45, size=(N, 2))
    # Simple relaxation to avoid overlaps
    for _ in range(50):
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(i + 1, N):
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < 0.1:
                    f = d / (dist + 1e-6) * (0.1 - dist)
                    forces[i] += f
                    forces[j] -= f
        points += 0.05 * forces
        points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
        points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    return points

def _targeted_refine(centers, max_iters=2000, step_size=0.005):
    """
    Targeted repulsion refinement: identify the binding constraint
    and make precise moves to increase the radius.
    """
    current_radius, binding, binding_info = _compute_radius(centers)
    best_radius = current_radius
    best_centers = centers.copy()
    
    for _ in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
            
        # Recompute binding constraint
        current_radius, binding, binding_info = _compute_radius(centers)
        
        if binding == 'boundary':
            idx = binding_info
            # Move circle toward center
            dir_vec = -centers[idx]
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-8:
                dir_vec = dir_vec / norm
            else:
                dir_vec = np.array([1.0, 0.0])
            
            new_pos = centers[idx].copy() + dir_vec * step_size
            if np.any(np.abs(new_pos) > 0.499):
                continue
            centers[idx] = new_pos
            
        elif binding == 'pair':
            i, j = binding_info
            vec = centers[j] - centers[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                dir_i = -vec / norm
                dir_j = vec / norm
            else:
                angle = np.random.uniform(0, 2 * np.pi)
                dir_i = np.array([np.cos(angle), np.sin(angle)])
                dir_j = -dir_i
            
            new_pos_i = centers[i].copy() + dir_i * step_size
            new_pos_j = centers[j].copy() + dir_j * step_size
            
            if np.any(np.abs(new_pos_i) > 0.499) or np.any(np.abs(new_pos_j) > 0.499):
                continue
            centers[i] = new_pos_i
            centers[j] = new_pos_j
            
        else:
            break
        
        new_radius = _compute_radius(centers)[0]
        if new_radius > best_radius + 1e-12:
            best_radius = new_radius
            best_centers = centers.copy()
        else:
            # Revert and try a smaller step or random perturbation
            if binding == 'boundary':
                idx = binding_info
                centers[idx] -= dir_vec * step_size
            elif binding == 'pair':
                i, j = binding_info
                centers[i] -= dir_i * step_size
                centers[j] -= dir_j * step_size
            
            # Occasionally add small random perturbation to escape local minima
            if np.random.random() < 0.1:
                idx = np.random.randint(N)
                centers[idx] += np.random.normal(0, 0.002, size=2)
                centers[idx, 0] = np.clip(centers[idx, 0], -0.499, 0.499)
                centers[idx, 1] = np.clip(centers[idx, 1], -0.499, 0.499)
    
    return best_centers, best_radius

def _slsqp_polish(centers):
    """
    Polish with SLSQP (SQP-based) on the exact objective.
    """
    def exact_obj(x):
        c = x.reshape(N, 2)
        r, _, _ = _compute_radius(c)
        return -r
    
    def exact_grad(x):
        # Numerical gradient (expensive but N=50 is small)
        c = x.reshape(N, 2)
        r, binding, binding_info = _compute_radius(c)
        grad = np.zeros_like(x)
        
        if binding == 'boundary':
            idx = binding_info
            # Gradient of -min_boundary w.r.t. center[idx]
            # min_boundary = min(0.5-|cx|, 0.5-|cy|)
            cx, cy = c[idx, 0], c[idx, 1]
            if 0.5 - abs(cx) <= 0.5 - abs(cy):
                grad[2*idx] = -np.sign(cx) if cx != 0 else 0
            else:
                grad[2*idx+1] = -np.sign(cy) if cy != 0 else 0
        elif binding == 'pair':
            i, j = binding_info
            vec = c[j] - c[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                unit = vec / norm
                grad[2*i] = -unit[0] / 2
                grad[2*i+1] = -unit[1] / 2
                grad[2*j] = unit[0] / 2
                grad[2*j+1] = unit[1] / 2
        
        return grad
    
    x0 = centers.flatten()
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    try:
        res = opt.minimize(
            exact_obj,
            x0,
            jac=exact_grad,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        x_final = res.x
        r_final = _compute_radius(x_final.reshape(N, 2))[0]
        return x_final.reshape(N, 2), r_final
    except Exception:
        return centers, _compute_radius(centers)[0]

def construct():
    start_time = time.time()
    
    best_radius = 0.0
    best_centers = None
    
    # Generate multiple initializations
    inits = []
    inits.append(_hexagonal_init(jitter=0.02))
    inits.append(_hexagonal_init(jitter=0.05))
    inits.append(_random_init())
    
    # Optimization settings
    beta_schedule = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    for i_init, init_centers in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        x_current = init_centers.flatten()
        current_radius = _compute_radius(init_centers)[0]
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = init_centers.copy()
        
        # Run L-BFGS-B with increasing beta
        for beta in beta_schedule:
            if time.time() - start_time > DEADLINE - 15:
                break
                
            res = opt.minimize(
                _objective,
                x_current,
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (2 * N),
                args=(beta,),
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            x_current = res.x
            r = _compute_radius(x_current.reshape(N, 2))[0]
            if r > best_radius:
                best_radius = r
                best_centers = x_current.reshape(N, 2).copy()
        
        # Apply targeted refinement
        if time.time() - start_time < DEADLINE - 10:
            refined, r_refined = _targeted_refine(x_current.reshape(N, 2))
            if r_refined > best_radius:
                best_radius = r_refined
                best_centers = refined.copy()
                x_current = refined.flatten()
        
        # SLSQP polish if time permits
        if time.time() - start_time < DEADLINE - 5:
            polished, r_polished = _slsqp_polish(x_current.reshape(N, 2))
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()
    
    # Fallback
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_centers = grid

    return best_centers

if __name__ == "__main__":
    c = construct()
    print("Radius:", _compute_radius(c)[0])
