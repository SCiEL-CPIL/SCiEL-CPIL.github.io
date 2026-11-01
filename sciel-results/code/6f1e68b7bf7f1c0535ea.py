"""Packing 50 equal circles in a unit disk: optimize via smooth surrogate + L-BFGS-B."""
import numpy as np
import time

N = 50

def _score(pts):
    """Compute the valid radius for a configuration of N points."""
    # Check finiteness
    if not np.all(np.isfinite(pts)):
        return 0.0
    if pts.shape != (N, 2):
        return 0.0
    # Check inside container
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    if np.any(norms > 1.0):
        return 0.0
    # Check pairwise distances
    min_pair_dist = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((pts[i] - pts[j]) ** 2))
            if d < min_pair_dist:
                min_pair_dist = d
    min_boundary = 1.0 - np.max(norms)
    r = min(min_pair_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0


def _soft_min_obj(pts, beta=500.0):
    """
    Smooth surrogate for -min(min_pair_dist/2, 1 - max_norm).
    We want to maximize r = min(d_min/2, b_min).
    Equivalently minimize -r.
    
    Using log-sum-exp style: for a set of constraints g_i >= 0, 
    the soft constraint is -log(sum(exp(-beta*g_i)))/beta which approximates min(g_i).
    
    Here constraints are: for each pair (i,j): d_ij/2 - r >= 0, and for each i: 1 - |c_i| - r >= 0.
    But we don't have r as a variable. Instead, we minimize a function that penalizes violations.
    
    Alternative: minimize f = -softmin(constraints) where constraints are the "slack" values.
    
    Let's use: maximize min over all constraints. Use:
    f = - (1/beta) * log( sum_i exp(-beta * s_i) )
    where s_i are the slacks (d_ij/2 for pairs, 1-|c_i| for boundaries).
    This approximates -min(s_i) as beta -> inf.
    We want to maximize this, so minimize -f = (1/beta)*log(sum(exp(-beta*s_i))).
    """
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Boundary distances
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - norms
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # For numerical stability, subtract max
    max_d = np.max(all_dists)
    shifted = all_dists - max_d
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = max_d - np.log(np.sum(np.exp(-beta * (all_dists - max_d)))) / beta
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val


def _soft_min_grad(pts, beta=500.0):
    """Gradient of the soft-min objective."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - norms
    
    all_dists = np.concatenate([pair_dists, boundary_dists])
    max_d = np.max(all_dists)
    shifted = all_dists - max_d
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    # d(soft_min)/d(all_dists_k) = exp(-beta*shifted_k) / sum_exp
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(pts)
    
    # Pairwise part
    # For pair (i,j), dist = |c_i - c_j|, d(dist)/d(c_i) = (c_i - c_j)/|c_i - c_j|
    # soft_min derivative wrt c_i: sum over j of weight_{ij} * (c_i - c_j)/|c_i - c_j|
    # But we're minimizing -soft_min, so grad of objective = -d(soft_min)/d(c)
    
    # Actually, let me recompute. 
    # soft_min = max_d - log(sum_exp)/beta
    # d(soft_min)/d(all_dists_k) = 1 - (1/beta)*(-beta)*exp(-beta*shifted_k)/sum_exp = 1 + exp(-beta*shifted_k)/sum_exp
    # Wait, that's not right. Let me redo.
    
    # soft_min = max_d - (1/beta)*log(sum_k exp(-beta*(d_k - max_d)))
    # d(soft_min)/d(d_k) = 0 - (1/beta)*(-beta)*exp(-beta*(d_k - max_d))/sum_exp = exp(-beta*(d_k-max_d))/sum_exp
    # So d(soft_min)/d(d_k) = w_k where w_k = exp(-beta*(d_k-max_d))/sum_exp
    
    # Now for pairwise: d_k = |c_i - c_j|/2
    # d(d_k)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(d_k)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    
    # For boundary: d_k = 1 - |c_i|
    # d(d_k)/d(c_i) = -c_i / |c_i|
    
    # grad of objective = -d(soft_min)/d(pts)
    
    # Pairwise contributions
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = weights[idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        direction = (pts[i] - pts[j]) / d
        # d(soft_min)/d(c_i) += w * direction / 2
        # d(soft_min)/d(c_j) += w * (-direction) / 2
        grad[i] -= w * direction / 2.0
        grad[j] += w * direction / 2.0
    
    # Boundary contributions
    n_pairs = len(iu[0])
    for i in range(N):
        w = weights[n_pairs + i]
        norm = norms[i]
        if norm < 1e-10:
            continue
        direction = pts[i] / norm
        # d(soft_min)/d(c_i) += w * (-direction)
        grad[i] += w * direction
    
    return grad


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s
    
    # Initial configuration: hexagonal lattice
    best_pts = None
    best_score = 0.0
    
    # Try a few initializations
    initializations = []
    
    # 1. Hexagonal lattice
    spacing = 0.24
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        if len(points) >= N:
            # Sort by radius to prefer center
            idx = np.argsort(radius, kind='stable')[:N]
            initializations.append(points[idx].copy())
            break
        spacing *= 0.97
    
    # 2. Ring-based initialization
    rng = np.random.default_rng(42)
    n_rings = 7
    ring_pts = []
    # Center point
    ring_pts.append(np.array([0.0, 0.0]))
    for ring in range(1, n_rings):
        r = ring * 0.14
        n_in_ring = int(2 * np.pi * r / 0.22)
        if n_in_ring < 1:
            n_in_ring = 1
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False) + rng.uniform(0, 2*np.pi/n_in_ring)
        for a in angles:
            ring_pts.append(np.array([r * np.cos(a), r * np.sin(a)]))
    if len(ring_pts) >= N:
        initializations.append(np.array(ring_pts[:N]))
    
    # 3. Random initialization
    for _ in range(3):
        pts = rng.random((N, 2)) * 2 - 1
        # Scale to fit
        norms = np.sqrt(np.sum(pts**2, axis=1))
        max_norm = np.max(norms)
        if max_norm > 0.9:
            pts *= 0.9 / max_norm
        initializations.append(pts)
    
    for init_pts in initializations:
        if time.time() > deadline - 20:
            break
        
        # Force-directed relaxation to resolve overlaps
        pts = init_pts.copy()
        for it in range(500):
            if time.time() > deadline - 15:
                break
            force = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    d = pts[i] - pts[j]
                    dist = np.sqrt(np.sum(d**2))
                    if dist < 1e-10:
                        d = rng.random(2) * 0.01
                        dist = np.sqrt(np.sum(d**2))
                    if dist < 0.25:
                        f = (0.25 - dist) / dist
                        force[i] += f * d
                        force[j] -= f * d
                # Boundary repulsion
                norm = np.sqrt(np.sum(pts[i]**2))
                if norm > 0.95:
                    force[i] -= 10 * (norm - 0.95) * pts[i] / norm
            pts += 0.01 * force
            # Project back
            norms = np.sqrt(np.sum(pts**2, axis=1))
            mask = norms > 0.99
            pts[mask] *= 0.99 / norms[mask]
        
        # L-BFGS-B optimization with increasing beta
        for beta in [100, 500, 2000, 10000]:
            if time.time() > deadline - 5:
                break
            
            from scipy.optimize import minimize
            
            def obj(x):
                return _soft_min_obj(x.reshape(N, 2), beta=beta)
            
            def grad(x):
                return _soft_min_grad(x.reshape(N, 2), beta=beta).flatten()
            
            x0 = pts.flatten()
            result = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                            options={'maxiter': 2000, 'gtol': 1e-8, 'ftol': 1e-12})
            pts = result.x.reshape(N, 2)
            
            # Project to valid region
            norms = np.sqrt(np.sum(pts**2, axis=1))
            mask = norms > 0.999
            pts[mask] *= 0.999 / norms[mask]
        
        # Final polish with high beta
        if time.time() < deadline - 3:
            from scipy.optimize import minimize
            beta_final = 50000
            x0 = pts.flatten()
            
            def obj_final(x):
                return _soft_min_obj(x.reshape(N, 2), beta=beta_final)
            
            def grad_final(x):
                return _soft_min_grad(x.reshape(N, 2), beta=beta_final).flatten()
            
            result = minimize(obj_final, x0, jac=grad_final, method='L-BFGS-B',
                            options={'maxiter': 5000, 'gtol': 1e-10, 'ftol': 1e-14})
            pts = result.x.reshape(N, 2)
            norms = np.sqrt(np.sum(pts**2, axis=1))
            mask = norms > 0.9999
            pts[mask] *= 0.9999 / norms[mask]
        
        # Evaluate
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None or best_score <= 0:
        # Fallback: return the hexagonal lattice result
        return initializations[0]
    
    return best_pts
