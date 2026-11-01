import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n_points, rotation=0.0):
    """Generate a hexagonal lattice of points inside the unit circle."""
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    k = int(1.0 / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if rotation != 0.0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        points = points @ R.T
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    
    points += np.random.normal(0, s * 0.01, size=points.shape)
    
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient.
    """
    C = x.reshape(N, 2)
    
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    L_safe = np.where(L > 1e-12, L, 1.0)
    x_pairs = L_safe / 2.0
    
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs_flat = x_pairs[i_idx, j_idx]
    
    x_all = np.concatenate([x_pairs_flat, x_walls])
    
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    grad = np.zeros_like(C)
    
    P_pairs = P[:len(i_idx)]
    
    D_flat = D[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    
    U = D_flat / L_flat_safe[:, np.newaxis]
    
    W = P_pairs * 0.5
    
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    P_walls = P[len(i_idx):]
    
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    
    grad += P_walls[:, np.newaxis] * U_walls
    
    return -val, -grad

def _optimize_lbfgs(points, beta, max_iter=300, time_limit=10.0):
    """Run L-BFGS-B optimization with a time limit."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return _soft_min_obj_and_grad(x, beta)
    
    try:
        res = minimize(obj_and_grad, points.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-12})
        if res.success or res.status in [1, 2, 4]:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return points

def _force_directed_refine(points, n_iter=500, time_limit=15.0):
    """
    Force-directed refinement: push apart overlapping circles and pull towards center.
    This is a different optimization paradigm than gradient-based soft-min.
    """
    current = points.copy()
    start_time = time.time()
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(current)
        
        # Pairwise repulsion for close pairs
        D = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        L = np.linalg.norm(D, axis=2)
        np.fill_diagonal(L, np.inf)
        
        # Only consider pairs that are too close (distance < some threshold)
        threshold = 0.15
        close_mask = (L < threshold) & (L > 1e-12)
        
        if np.any(close_mask):
            # Repulsive force proportional to 1/d^2 for close pairs
            L_safe = np.where(close_mask, L, 1.0)
            U = D / L_safe[:, :, np.newaxis]
            F = np.where(close_mask[:, :, np.newaxis], 1.0 / L_safe[:, :, np.newaxis], 0.0)
            
            for i in range(N):
                forces[i] += np.sum(F[i] * U[i], axis=0)
                forces[i] -= np.sum(F[:, i] * U[:, i], axis=0)
        
        # Wall constraint: push inward if outside
        norms = np.linalg.norm(current, axis=1)
        wall_mask = norms > 0.95
        if np.any(wall_mask):
            U_walls = -current[wall_mask] / norms[wall_mask, np.newaxis]
            forces[wall_mask] += 0.5 * U_walls
        
        # Damping and step size
        step = 0.001
        current += step * forces
        
        # Clamp to unit circle
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            current[mask] = current[mask] * (0.999 / norms[mask])[:, np.newaxis]
    
    return current

def _simulated_annealing_refine(points, n_iter=1000, time_limit=20.0):
    """
    Simulated annealing: random perturbations accepted with probability based on score change.
    Different from deterministic gradient methods.
    """
    current = points.copy()
    best = current.copy()
    best_score = _compute_true_score(best)
    start_time = time.time()
    
    T = 0.01  # Initial temperature
    cooling = 0.995
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation
        proposal = current + np.random.normal(0, 0.002, size=current.shape)
        
        # Clamp to unit circle
        norms = np.linalg.norm(proposal, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            proposal[mask] = proposal[mask] * (0.999 / norms[mask])[:, np.newaxis]
        
        new_score = _compute_true_score(proposal)
        delta = new_score - best_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current = proposal
            if new_score > best_score:
                best_score = new_score
                best = proposal.copy()
        
        T *= cooling
    
    return best

def _anneal_beta(points, beta_start, beta_end, n_steps, time_limit, max_iter_per_step=100):
    """Progressively increase beta and optimize at each step."""
    current = points.copy()
    for i in range(n_steps):
        beta = beta_start * (beta_end / beta_start) ** (i / max(n_steps - 1, 1))
        current = _optimize_lbfgs(current, beta, max_iter=max_iter_per_step, time_limit=time_limit)
    return current

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin before 150s kill
    
    inits = []
    
    for rot in [0.0, np.pi / 6, np.pi / 3, np.pi / 4]:
        inits.append(_hex_lattice_init(N, rotation=rot))
    
    r = np.sqrt(np.random.random(N))
    a = np.random.random(N) * 2 * np.pi
    inits.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    
    best_points = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() - start_time > time_limit:
            break
        
        current = init.copy()
        
        # Phase 1: Low beta to establish rough structure
        current = _optimize_lbfgs(current, 20.0, max_iter=150, time_limit=8.0)
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
        
        # Phase 2: Medium beta
        if time.time() - start_time < time_limit:
            current = _optimize_lbfgs(current, 100.0, max_iter=200, time_limit=10.0)
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
        
        # Phase 3: Force-directed refinement (new structural approach)
        if time.time() - start_time < time_limit:
            current = _force_directed_refine(current, n_iter=300, time_limit=12.0)
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
        
        # Phase 4: High beta gradient refinement
        if time.time() - start_time < time_limit:
            current = _optimize_lbfgs(current, 500.0, max_iter=300, time_limit=12.0)
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    # Second pass: refine the best solution further
    if best_points is not None and time.time() - start_time < time_limit:
        current = best_points.copy()
        for beta in [1000.0, 2000.0, 5000.0]:
            if time.time() - start_time > time_limit:
                break
            current = _optimize_lbfgs(current, beta, max_iter=500, time_limit=15.0)
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
        
        # Force-directed polish
        if time.time() - start_time < time_limit:
            current = _force_directed_refine(current, n_iter=200, time_limit=10.0)
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    # Simulated annealing for escaping local optima
    if best_points is not None and time.time() - start_time < time_limit - 10:
        current = _simulated_annealing_refine(best_points, n_iter=500, time_limit=15.0)
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() - start_time < time_limit - 5:
        current = best_points.copy()
        current = _optimize_lbfgs(current, 10000.0, max_iter=1000, time_limit=15.0)
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
    
    # Ensure all points are inside the unit circle
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            best_points[mask] = best_points[mask] * (0.9999 / norms[mask])[:, np.newaxis]
    
    return best_points
