"""Rewritten program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
import math

N = 130

# EVOLVE-BLOCK-START

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) == 3:
            v0, v1, v2 = verts
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        elif len(verts) == 4:
            v0, v1, v2 = verts[0], verts[1], verts[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        else:
            continue
        
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len
        
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        for v in verts:
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_rad:
                max_rad = ang
    
    return math.degrees(max_rad)

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _soft_max_loss(P_flat, samples, lam=8.0):
    """
    Soft-max covering loss: sum over samples of exp(lam * min_dist_to_points).
    Returns (loss, grad_flat).
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms

    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)

    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)

    loss = np.sum(np.exp(lam * min_dists))

    weights = lam * np.exp(lam * min_dists)
    dots_nearest = dots[np.arange(len(samples)), nearest]
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
    grad_factor = -weights / denom

    grad_P = np.zeros_like(Pn)
    np.add.at(grad_P, nearest, grad_factor[:, None] * samples)

    grad_dot = np.sum(grad_P * Pn, axis=1, keepdims=True)
    grad_tan = grad_P - grad_dot * Pn

    grad_flat = (grad_tan / norms).flatten()

    return loss, grad_flat

def _repulsion_loss(P_flat, samples, lam=8.0, repulse_strength=0.5):
    """
    Combined loss: soft-max covering + pairwise repulsion to prevent clustering.
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms

    # Covering part
    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)
    cover_loss = np.sum(np.exp(lam * min_dists))

    # Repulsion part: for each pair, add exp(-k * dist) to encourage separation
    # Use a subset of pairs for efficiency
    n = len(Pn)
    # All pairwise distances
    all_dots = Pn @ Pn.T
    np.fill_diagonal(all_dots, 2.0)  # avoid self
    all_dots = np.clip(all_dots, -1.0, 1.0)
    all_dists = np.arccos(all_dots)
    
    # Repulsion: sum of exp(-k * dist) for all pairs
    k_rep = 5.0
    repulse_matrix = np.exp(-k_rep * all_dists)
    np.fill_diagonal(repulse_matrix, 0.0)
    repulse_loss = repulse_strength * np.sum(repulse_matrix)
    
    total_loss = cover_loss + repulse_loss

    # Gradient for covering part
    weights = lam * np.exp(lam * min_dists)
    dots_nearest = dots[np.arange(len(samples)), nearest]
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
    grad_factor = -weights / denom

    grad_P = np.zeros_like(Pn)
    np.add.at(grad_P, nearest, grad_factor[:, None] * samples)

    # Gradient for repulsion part
    # d/dP_i of exp(-k * arccos(P_i . P_j)) = exp(-k*dist) * (-k) * d(arccos)/dP_i
    # d(arccos(x))/dx = -1/sqrt(1-x^2)
    # d(arccos(P_i.P_j))/dP_i = -(P_j - (P_i.P_j)*P_i) / sqrt(1-(P_i.P_j)^2)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot_ij = all_dots[i, j]
            dist_ij = all_dists[i, j]
            rep_val = np.exp(-k_rep * dist_ij)
            denom_rep = np.sqrt(np.maximum(1e-12, 1.0 - dot_ij**2))
            # Gradient contribution to P_i
            proj = dot_ij * Pn[i]
            tangent_dir = Pn[j] - proj
            grad_rep_i = repulse_strength * rep_val * (-k_rep) * (-tangent_dir / denom_rep)
            grad_P[i] += grad_rep_i

    # Project to tangent
    grad_dot = np.sum(grad_P * Pn, axis=1, keepdims=True)
    grad_tan = grad_P - grad_dot * Pn

    grad_flat = (grad_tan / norms).flatten()

    return total_loss, grad_flat

def _local_search_greedy(P, samples, max_iters=300, deadline=None):
    """
    Greedy local search: repeatedly move the point responsible for the worst
    sample toward that sample with adaptive step size.
    """
    n_samples = len(samples)
    rng = np.random.default_rng(42)
    
    def get_max_dist(P):
        dots = samples @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest = np.argmin(dists, axis=1)
        return np.max(min_dists), min_dists, nearest

    current_max, min_dists, nearest = get_max_dist(P)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        p_idx = nearest[worst_idx]
        p_old = P[p_idx].copy()
        
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-8:
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm < 1e-8:
                continue
            grad_dir = rand / rand_norm
        else:
            grad_dir = grad_dir / grad_norm
        
        best_step = 0.0
        best_max = current_max
        best_p = None
        
        for step in [0.15, 0.10, 0.05, 0.02, 0.01]:
            new_p = p_old + step * grad_dir
            new_p = _normalize(new_p.reshape(1, 3))[0]
            P[p_idx] = new_p
            
            new_max, _, _ = get_max_dist(P)
            
            if new_max < best_max:
                best_max = new_max
                best_step = step
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_p is not None:
            P[p_idx] = best_p
            current_max = best_max
            min_dists, nearest = get_max_dist(P)[1:]
        else:
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm > 1e-8:
                rand /= rand_norm
            P[p_idx] = _normalize((p_old + 0.05 * rand).reshape(1, 3))[0]
            min_dists, nearest = get_max_dist(P)[1:]
    
    return P, current_max

def _repulsion_local_search(P, samples, deadline):
    """
    Local search that combines covering optimization with repulsion.
    For each point, compute a force that balances pulling toward the worst
    uncovered sample and pushing away from nearby points.
    """
    n = len(P)
    rng = np.random.default_rng(42)
    
    def get_max_dist(P):
        dots = samples @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest = np.argmin(dists, axis=1)
        return np.max(min_dists), min_dists, nearest

    current_max, min_dists, nearest = get_max_dist(P)
    max_iters = 150
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find worst sample
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        p_idx = nearest[worst_idx]
        p_old = P[p_idx].copy()
        
        # Compute repulsion from nearby points
        dots_all = P @ p_old
        dots_all = np.clip(dots_all, -1.0, 1.0)
        dists_all = np.arccos(dots_all)
        # Exclude self
        dists_all[p_idx] = np.pi
        
        # Repulsion force: sum of exp(-k*dist) * direction
        k_rep = 5.0
        rep_weights = np.exp(-k_rep * dists_all)
        rep_weights[p_idx] = 0.0
        
        # Direction of repulsion from each point
        # For point j, direction to push p_old away from P[j]
        # is in the tangent plane: P[j] - (P[j].p_old)*p_old
        rep_force = np.zeros(3)
        for j in range(n):
            if j == p_idx:
                continue
            proj = dots_all[j] * p_old
            tangent_dir = P[j] - proj
            tan_norm = np.linalg.norm(tangent_dir)
            if tan_norm > 1e-10:
                tangent_dir /= tan_norm
            rep_force += rep_weights[j] * tangent_dir
        
        # Attraction toward worst sample
        proj_worst = np.dot(p_old, worst_sample)
        att_dir = worst_sample - proj_worst * p_old
        att_norm = np.linalg.norm(att_dir)
        if att_norm > 1e-8:
            att_dir /= att_norm
        else:
            att_dir = np.zeros(3)
        
        # Combine forces
        balance = 0.3  # repulsion strength relative to attraction
        combined = att_dir - balance * rep_force
        comb_norm = np.linalg.norm(combined)
        if comb_norm < 1e-10:
            combined = att_dir if att_norm > 1e-8 else rng.standard_normal(3)
            comb_norm = np.linalg.norm(combined)
        combined /= comb_norm
        
        # Adaptive step
        best_step = 0.0
        best_max = current_max
        best_p = None
        
        for step in [0.12, 0.08, 0.04, 0.02, 0.01]:
            new_p = p_old + step * combined
            new_p = _normalize(new_p.reshape(1, 3))[0]
            P[p_idx] = new_p
            
            new_max, _, _ = get_max_dist(P)
            
            if new_max < best_max:
                best_max = new_max
                best_step = step
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_p is not None:
            P[p_idx] = best_p
            current_max = best_max
            min_dists, nearest = get_max_dist(P)[1:]
        else:
            # Perturb
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm > 1e-8:
                rand /= rand_norm
            P[p_idx] = _normalize((p_old + 0.03 * rand).reshape(1, 3))[0]
            min_dists, nearest = get_max_dist(P)[1:]
    
    return P, current_max

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Initialize with Fibonacci sphere + small noise
    P = _fibonacci_sphere(N, seed_noise=0.03, seed=42)
    
    # 2. Generate samples for optimization
    S_opt = 10000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # 3. First L-BFGS-B optimization with soft-max loss
    time_check = [False]
    def time_callback(xk):
        if time.time() > deadline - 50.0:
            time_check[0] = True
            raise ValueError("Time limit approaching")
    
    try:
        res = minimize(
            lambda x: _soft_max_loss(x, samples_opt, lam=8.0),
            P.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
            callback=time_callback
        )
        if res.x is not None:
            P = _normalize(res.x.reshape(-1, 3))
    except Exception:
        pass
    
    # 4. Second optimization with repulsion-aware loss
    if time.time() < deadline - 70.0:
        S_opt2 = 12000
        samples_opt2 = _fibonacci_sphere(S_opt2)
        
        def time_callback2(xk):
            if time.time() > deadline - 40.0:
                raise ValueError("Time limit approaching")
        
        try:
            res2 = minimize(
                lambda x: _soft_max_loss(x, samples_opt2, lam=10.0),
                P.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 60, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
                callback=time_callback2
            )
            if res2.x is not None:
                P = _normalize(res2.x.reshape(-1, 3))
        except Exception:
            pass
    
    # 5. Repulsion-based local search
    if time.time() < deadline - 25.0:
        S_fine = 25000
        samples_fine = _fibonacci_sphere(S_fine)
        P, _ = _repulsion_local_search(P, samples_fine, deadline - 20.0)
    
    # 6. Final greedy local search
    if time.time() < deadline - 8.0:
        S_fine2 = 30000
        samples_fine2 = _fibonacci_sphere(S_fine2)
        P, _ = _local_search_greedy(P, samples_fine2, max_iters=100, 
                                     deadline=deadline - 5.0)
    
    # Final normalization
    P = _normalize(P)
    
    return P

# EVOLVE-BLOCK-END
