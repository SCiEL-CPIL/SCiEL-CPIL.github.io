import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _score_vec(points):
    """Vectorized score computation."""
    points = points.reshape(-1, 2)
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r

def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.05, 0.25, 50):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points

def _ring_init(n):
    """Generate a ring-based initial configuration."""
    # Center circle
    points = [[0.0, 0.0]]
    # First ring
    ring1_r = 0.25
    for k in range(6):
        angle = 2 * np.pi * k / 6
        points.append([ring1_r * np.cos(angle), ring1_r * np.sin(angle)])
    # Second ring
    ring2_r = 0.5
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.pi / 12
        points.append([ring2_r * np.cos(angle), ring2_r * np.sin(angle)])
    # Third ring
    ring3_r = 0.75
    for k in range(18):
        angle = 2 * np.pi * k / 18
        points.append([ring3_r * np.cos(angle), ring3_r * np.sin(angle)])
    # Fourth ring
    ring4_r = 0.9
    for k in range(24):
        angle = 2 * np.pi * k / 24 + np.pi / 24
        points.append([ring4_r * np.cos(angle), ring4_r * np.sin(angle)])
    
    points = np.array(points)
    if len(points) >= n:
        # Select best n points
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
    return np.random.randn(n, 2) * 0.5

def _compute_r_and_grad(points):
    """
    Compute r and its gradient w.r.t. all point coordinates.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|))
    Returns (r, grad) where grad has shape (N, 2).
    """
    points = points.reshape(-1, 2)
    n = len(points)
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary = 1.0 - norms
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    
    r = min(np.min(boundary), np.min(pair_dists))
    
    # Gradient
    grad = np.zeros((n, 2))
    
    # Boundary contribution
    min_boundary = np.min(boundary)
    if abs(min_boundary - r) < 1e-12:
        for i in range(n):
            if abs(boundary[i] - min_boundary) < 1e-12:
                if norms[i] > 1e-12:
                    grad[i] -= points[i] / norms[i]
    
    # Pair contribution
    min_pair = np.min(pair_dists)
    if abs(min_pair - r) < 1e-12:
        for idx in range(len(pair_dists)):
            if abs(pair_dists[idx] - min_pair) < 1e-12:
                i, j = iu[0][idx], iu[1][idx]
                d_vec = points[i] - points[j]
                d_norm = np.linalg.norm(d_vec)
                if d_norm > 1e-12:
                    grad[i] += d_vec / (2 * d_norm)
                    grad[j] -= d_vec / (2 * d_norm)
    
    return r, grad

def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))

def _contact_graph(points, threshold=None):
    """
    Build a contact graph where edges represent nearest-neighbor contacts.
    Returns (edges, edge_lengths) where edges is (M, 2) and edge_lengths is (M,).
    """
    points = points.reshape(-1, 2)
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu]
    
    if threshold is None:
        # Use a threshold based on the current minimum distance
        threshold = np.min(pair_dists) * 1.5
    
    mask = pair_dists <= threshold
    edges = np.column_stack((iu[0][mask], iu[1][mask]))
    edge_lengths = pair_dists[mask]
    
    return edges, edge_lengths

def _graph_potential(points, beta=100.0):
    """
    Graph-based potential that enforces equal edge lengths and boundary distances.
    Uses a smooth surrogate for the min objective.
    """
    points = points.reshape(-1, 2)
    n = len(points)
    
    # Get contact graph
    edges, edge_lengths = _contact_graph(points)
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary = 1.0 - norms
    
    # Target radius estimate
    r_est = min(np.min(boundary), np.min(edge_lengths) / 2) if len(edge_lengths) > 0 else np.min(boundary)
    
    # Potential: sum of squared deviations from target r
    pot = 0.0
    
    # Pair potentials: enforce |c_i - c_j| = 2*r_est
    if len(edges) > 0:
        ci = points[edges[:, 0]]
        cj = points[edges[:, 1]]
        d = np.sqrt(np.sum((ci - cj) ** 2, axis=1))
        pot += np.sum((d - 2 * r_est) ** 2)
    
    # Boundary potentials: enforce |c_i| = 1 - r_est
    pot += np.sum((norms - (1 - r_est)) ** 2)
    
    return pot, r_est

def _graph_potential_grad(points, beta=100.0):
    """
    Gradient of the graph potential.
    """
    points = points.reshape(-1, 2)
    n = len(points)
    
    # Get contact graph
    edges, edge_lengths = _contact_graph(points)
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary = 1.0 - norms
    
    # Target radius estimate
    r_est = min(np.min(boundary), np.min(edge_lengths) / 2) if len(edge_lengths) > 0 else np.min(boundary)
    
    grad = np.zeros((n, 2))
    
    # Pair gradient
    if len(edges) > 0:
        ci = points[edges[:, 0]]
        cj = points[edges[:, 1]]
        d_vec = ci - cj
        d = np.sqrt(np.sum(d_vec ** 2, axis=1))
        
        # d/d(c_i) of (d - 2*r_est)^2 = 2*(d - 2*r_est) * (d_vec / d)
        coeff = 2 * (d - 2 * r_est) / np.where(d > 1e-12, d, 1e-12)
        
        for idx in range(len(edges)):
            i, j = edges[idx, 0], edges[idx, 1]
            if d[idx] > 1e-12:
                unit = d_vec[idx] / d[idx]
                grad[i] += coeff[idx] * unit
                grad[j] -= coeff[idx] * unit
    
    # Boundary gradient: d/d(c_i) of (|c_i| - (1-r_est))^2 = 2*(|c_i| - (1-r_est)) * (c_i / |c_i|)
    for i in range(n):
        if norms[i] > 1e-12:
            coeff = 2 * (norms[i] - (1 - r_est)) / norms[i]
            grad[i] += coeff * points[i]
    
    return grad

def _local_optimize_graph(c, time_limit=10.0):
    """Local optimization using the graph potential with L-BFGS-B."""
    t_start = time.time()
    
    def obj(x):
        c = x.reshape(N, 2)
        pot, _ = _graph_potential(c)
        return pot
    
    def obj_grad(x):
        c = x.reshape(N, 2)
        return _graph_potential_grad(c).flatten()
    
    try:
        res = minimize(
            obj,
            c.flatten(),
            jac=obj_grad,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success or res.fun < obj(c.flatten()):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    
    return c

def _soft_min_optimize(c, time_limit=10.0):
    """Local optimization using soft-min objective with L-BFGS-B."""
    pair_idx = _pairwise_indices(N)
    
    def neg_soft_min(x):
        c = x.reshape(N, 2)
        ci = c[pair_idx[:, 0]]
        cj = c[pair_idx[:, 1]]
        d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
        b = 1 - np.hypot(c[:, 0], c[:, 1])
        all_vals = np.concatenate([d, b])
        beta = 500.0
        m = np.min(all_vals)
        return m + np.log(np.sum(np.exp(-beta * (all_vals - m)))) / beta
    
    def neg_soft_min_grad(x):
        c = x.reshape(N, 2)
        ci = c[pair_idx[:, 0]]
        cj = c[pair_idx[:, 1]]
        d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
        b = 1 - np.hypot(c[:, 0], c[:, 1])
        all_vals = np.concatenate([d, b])
        beta = 500.0
        m = np.min(all_vals)
        weights = np.exp(-beta * (all_vals - m))
        weights /= np.sum(weights)
        
        grad_vals = -weights
        grad_c = np.zeros((N, 2))
        
        for idx in range(len(d)):
            if abs(grad_vals[idx]) > 1e-15:
                i, j = pair_idx[idx, 0], pair_idx[idx, 1]
                d_vec = c[i] - c[j]
                d_norm = np.linalg.norm(d_vec)
                if d_norm > 1e-12:
                    unit = d_vec / d_norm
                    grad_c[i] += grad_vals[idx] * unit / 2
                    grad_c[j] -= grad_vals[idx] * unit / 2
        
        for idx in range(N):
            if abs(grad_vals[len(d) + idx]) > 1e-15:
                norm = np.linalg.norm(c[idx])
                if norm > 1e-12:
                    grad_c[idx] -= grad_vals[len(d) + idx] * c[idx] / norm
        
        return grad_c.flatten()
    
    t_start = time.time()
    try:
        res = minimize(
            neg_soft_min,
            c.flatten(),
            jac=neg_soft_min_grad,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success or res.fun < neg_soft_min(c.flatten()):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    
    return c

def _polish_gradient_descent(points, time_limit=5.0):
    """
    High-precision gradient descent polish using exact subgradient of the min function.
    """
    pts = points.copy().flatten()
    t_start = time.time()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    
    lr = 0.005
    momentum = 0.8
    v = np.zeros_like(pts)
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() - t_start < time_limit:
        c = pts.reshape(N, 2)
        r, grad = _compute_r_and_grad(c)
        
        if r > best_score + 1e-15:
            best_score = r
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                lr *= 0.5
                v *= 0.5
                no_improve_count = 0
                if lr < 1e-8:
                    break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        else:
            break
        
        v = momentum * v + lr * grad
        pts = pts + v
        
        c_new = pts.reshape(N, 2)
        norms = np.linalg.norm(c_new, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                scale = (1.0 - 1e-10) / norms[i]
                c_new[i] *= scale
                pts[i*2:(i+1)*2] = c_new[i]
    
    return best_pts.reshape(N, 2), best_score

def _anneal(points, time_limit=20.0):
    """Simulated annealing to escape local minima."""
    pts = points.copy().flatten()
    t_start = time.time()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    
    T = 0.05  # Initial temperature
    cooling = 0.995
    n_steps = 0
    
    while time.time() - t_start < time_limit and T > 1e-6:
        # Generate a random perturbation
        delta = np.random.randn(N, 2) * T
        new_pts = pts + delta.flatten()
        new_pts = new_pts.reshape(N, 2)
        
        # Project to feasible region
        norms = np.linalg.norm(new_pts, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                scale = (1.0 - 1e-10) / norms[i]
                new_pts[i] *= scale
        
        new_score = _score_vec(new_pts)
        old_score = _score_vec(pts.reshape(N, 2))
        
        # Accept if better or with probability exp((new-old)/T)
        if new_score > old_score or np.random.random() < np.exp((new_score - old_score) / T):
            pts = new_pts.flatten()
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        
        T *= cooling
        n_steps += 1
    
    return best_pts.reshape(N, 2), best_score

def construct():
    t_start = time.time()
    
    # 1. Multiple initializations
    init1 = _hexagonal_init(N)
    init2 = _ring_init(N)
    init3 = np.random.randn(N, 2) * 0.4
    
    best_points = None
    best_score = 0.0
    
    for init in [init1, init2, init3]:
        r = _score_vec(init)
        if r > best_score:
            best_score = r
            best_points = init.copy()
    
    # 2. Graph-based optimization
    if time.time() < DEADLINE - 50:
        c_opt = _local_optimize_graph(best_points, time_limit=min(20.0, DEADLINE - (time.time() - t_start) - 50))
        r_opt = _score_vec(c_opt)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # 3. Soft-min optimization
    if time.time() < DEADLINE - 35:
        c_opt = _soft_min_optimize(best_points, time_limit=min(15.0, DEADLINE - (time.time() - t_start) - 35))
        r_opt = _score_vec(c_opt)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # 4. Simulated annealing
    if time.time() < DEADLINE - 20:
        annealed, r_anneal = _anneal(best_points, time_limit=min(12.0, DEADLINE - (time.time() - t_start) - 20))
        if r_anneal > best_score:
            best_score = r_anneal
            best_points = annealed.copy()
    
    # 5. Fine polish with gradient descent
    if time.time() < DEADLINE - 10:
        polished, r_polish = _polish_gradient_descent(best_points, time_limit=min(8.0, DEADLINE - (time.time() - t_start) - 10))
        if r_polish > best_score:
            best_score = r_polish
            best_points = polished.copy()
    
    # 6. Multi-Start Perturb and Polish
    n_restarts = 0
    while time.time() < DEADLINE - 8 and n_restarts < 6:
        perturbation_scale = 0.03 / (1.0 + 0.3 * n_restarts)
        x0 = best_points + np.random.randn(N, 2) * perturbation_scale
        
        # Graph-based optimization
        c_opt = _local_optimize_graph(x0, time_limit=min(10.0, DEADLINE - (time.time() - t_start) - 8))
        r_opt = _score_vec(c_opt)
        
        # Fine polish
        if time.time() < DEADLINE - 3:
            polished, r_polish = _polish_gradient_descent(c_opt, time_limit=min(5.0, DEADLINE - (time.time() - t_start) - 3))
            r_opt = max(r_opt, r_polish)
            if r_polish > r_opt:
                c_opt = polished
        
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
        
        n_restarts += 1
    
    # 7. Final Polish
    if time.time() < DEADLINE - 2:
        polished, r_final = _polish_gradient_descent(best_points, time_limit=min(5.0, DEADLINE - (time.time() - t_start) - 2))
        if r_final > best_score:
            best_points = polished
            best_score = r_final
    
    # Ensure validity: clamp points that might be slightly outside
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            scale = (1 - best_score) / r2[i]
            c_final[i] *= scale
    
    # Final verification
    r_check = _score_vec(c_final)
    if r_check < best_score - 1e-12:
        c_final *= (1.0 - 1e-10)
    
    return c_final
