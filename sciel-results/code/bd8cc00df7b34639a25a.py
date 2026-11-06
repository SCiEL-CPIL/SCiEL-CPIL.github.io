"""Combined program for covering the sphere with 130 equal caps."""
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

def _repulsion_init(n, iterations=3000, dt=0.01, seed=42):
    """
    Initialize points using a Thomson-like repulsion process.
    Points repel each other with 1/r^2 force on the sphere.
    This creates a more uniform initial distribution than Fibonacci.
    """
    rng = np.random.default_rng(seed)
    # Start with random points on sphere
    P = rng.standard_normal((n, 3))
    P = _normalize(P)
    
    for it in range(iterations):
        # Compute pairwise distances
        dots = P @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # Repulsion force: 1/r^2 in tangent direction
        forces = np.zeros_like(P)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r = dists[i, j]
                if r < 1e-10:
                    continue
                # Force magnitude
                f_mag = 1.0 / (r * r + 1e-6)
                # Tangent direction from i to j
                proj = dots[i, j] * P[i]
                tangent = P[j] - proj
                tan_norm = np.linalg.norm(tangent)
                if tan_norm > 1e-10:
                    tangent /= tan_norm
                forces[i] += f_mag * tangent
        
        # Update positions with cooling
        temperature = dt * (1.0 - it / iterations)
        P_new = P + temperature * forces
        P = _normalize(P_new)
    
    return P

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

def _log_sum_exp_max_loss(P_flat, samples, beta=50.0):
    """
    Smooth approximation of max distance using log-sum-exp.
    This is a smooth surrogate for the max covering radius.
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms

    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)

    # Log-sum-exp for stability
    max_dists = np.max(dists, axis=1)
    shifted = dists - max_dists[:, None]
    log_sum_exp = max_dists + np.log(np.sum(np.exp(beta * shifted), axis=1)) / beta
    
    loss = np.mean(log_sum_exp)
    
    # Gradient
    weights = np.exp(beta * (dists - max_dists[:, None]))
    weights /= np.sum(weights, axis=1, keepdims=True)
    
    grad_P = np.zeros_like(Pn)
    for i in range(len(samples)):
        for j in range(len(Pn)):
            if weights[i, j] < 1e-15:
                continue
            dot_ij = dots[i, j]
            denom = np.sqrt(max(1e-12, 1.0 - dot_ij ** 2))
            grad_P[j] += -weights[i, j] / denom * samples[i]
    
    # Project to tangent space
    grad_dot = np.sum(grad_P * Pn, axis=1, keepdims=True)
    grad_tan = grad_P - grad_dot * Pn
    grad_flat = (grad_tan / norms).flatten()
    
    return loss, grad_flat

def _combined_loss(P_flat, samples, beta=50.0, spectral_weight=0.15):
    """
    Combined loss: Log-sum-exp covering + pairwise distance variance.
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms
    n = len(Pn)

    target_dist = math.sqrt(4 * np.pi / n)

    # 1. Log-sum-exp covering loss
    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    max_dists = np.max(dists, axis=1)
    shifted = dists - max_dists[:, None]
    log_sum_exp = max_dists + np.log(np.sum(np.exp(beta * shifted), axis=1)) / beta
    cover_loss = np.mean(log_sum_exp)
    
    weights = np.exp(beta * (dists - max_dists[:, None]))
    weights /= np.sum(weights, axis=1, keepdims=True)
    
    grad_P = np.zeros_like(Pn)
    for i in range(len(samples)):
        for j in range(len(Pn)):
            if weights[i, j] < 1e-15:
                continue
            dot_ij = dots[i, j]
            denom = np.sqrt(max(1e-12, 1.0 - dot_ij ** 2))
            grad_P[j] += -weights[i, j] / denom * samples[i]

    # 2. Pairwise distance variance
    dots_all = Pn @ Pn.T
    np.fill_diagonal(dots_all, 2.0)
    dots_all = np.clip(dots_all, -1.0, 1.0)
    dists_all = np.arccos(dots_all)
    
    iu = np.triu_indices(n, 1)
    pair_dists = dists_all[iu]
    spec_loss = np.sum((pair_dists - target_dist) ** 2)
    
    # Gradient for spectral loss
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        dot_ij = dots_all[i, j]
        dist_ij = dists_all[i, j]
        denom_rep = np.sqrt(max(1e-12, 1.0 - dot_ij ** 2))
        factor = 2 * (dist_ij - target_dist) * (-1.0 / denom_rep)
        
        tangent_i = Pn[j] - dot_ij * Pn[i]
        tangent_j = Pn[i] - dot_ij * Pn[j]
        
        grad_P[i] += spectral_weight * factor * tangent_i
        grad_P[j] += spectral_weight * factor * tangent_j
    
    total_loss = cover_loss + spectral_weight * spec_loss
    
    # Project gradient to tangent space
    grad_dot = np.sum(grad_P * Pn, axis=1, keepdims=True)
    grad_tan = grad_P - grad_dot * Pn
    grad_flat = (grad_tan / norms).flatten()
    
    return total_loss, grad_flat

def _fast_local_search(P, samples, deadline, max_iters=300):
    """
    Fast local search: Move the point responsible for the worst sample.
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
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        p_idx = nearest[worst_idx]
        p_old = P[p_idx].copy()
        
        # Direction to move: toward worst_sample, projected to tangent space
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
        
        steps = [0.15, 0.10, 0.05, 0.02, 0.01]
        
        for step in steps:
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
            _, min_dists, nearest = get_max_dist(P)
        else:
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm > 1e-8:
                rand /= rand_norm
            P[p_idx] = _normalize((p_old + 0.03 * rand).reshape(1, 3))[0]
            _, min_dists, nearest = get_max_dist(P)
            
    return P, current_max

def _repulsion_local_search(P, samples, deadline, max_iters=150):
    """Local search combining covering attraction with pairwise repulsion."""
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
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        p_idx = nearest[worst_idx]
        p_old = P[p_idx].copy()
        
        dots_all = P @ p_old
        dots_all = np.clip(dots_all, -1.0, 1.0)
        dists_all = np.arccos(dots_all)
        dists_all[p_idx] = np.pi
        
        k_rep = 5.0
        rep_weights = np.exp(-k_rep * dists_all)
        rep_weights[p_idx] = 0.0
        
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
        
        proj_worst = np.dot(p_old, worst_sample)
        att_dir = worst_sample - proj_worst * p_old
        att_norm = np.linalg.norm(att_dir)
        if att_norm > 1e-8:
            att_dir /= att_norm
        else:
            att_dir = np.zeros(3)
        
        balance = 0.3
        combined = att_dir - balance * rep_force
        comb_norm = np.linalg.norm(combined)
        if comb_norm < 1e-10:
            combined = att_dir if att_norm > 1e-8 else rng.standard_normal(3)
            comb_norm = np.linalg.norm(combined)
        combined /= comb_norm
        
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
    deadline = start_time + 145.0
    
    best_P = None
    best_score = float('inf')
    
    # Multiple restarts with different initializations
    restarts = [
        ("repulsion", None, 42),
        ("fibonacci", 0.03, 42),
        ("fibonacci", 0.05, 123),
        ("fibonacci", 0.08, 999),
        ("fibonacci", 0.0, 777),
    ]
    
    # Allocate time budget
    total_time = deadline - start_time
    time_per_restart = total_time / len(restarts)
    
    for r_idx, (init_type, noise, seed) in enumerate(restarts):
        current_deadline = start_time + (r_idx + 1) * time_per_restart
        current_deadline = min(current_deadline, deadline - 5.0)
        
        if time.time() > current_deadline - 2.0:
            break
        
        # 1. Initialize
        if init_type == "repulsion":
            P = _repulsion_init(N, iterations=2000, dt=0.01, seed=seed)
        else:
            P = _fibonacci_sphere(N, seed_noise=noise, seed=seed)
        
        # 2. Generate samples
        S_opt = 10000
        samples_opt = _fibonacci_sphere(S_opt, seed=seed + 100)
        
        # 3. L-BFGS optimization with combined loss
        opt_deadline = current_deadline - 10.0
        
        def time_callback(xk):
            if time.time() > opt_deadline:
                raise ValueError("Time limit approaching")
        
        try:
            res = minimize(
                lambda x: _combined_loss(x, samples_opt, beta=50.0, spectral_weight=0.15),
                P.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 40, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
                callback=time_callback
            )
            if res.x is not None:
                P = _normalize(res.x.reshape(-1, 3))
        except Exception:
            pass
        
        # 4. Local search refinement
        if time.time() < current_deadline - 5.0:
            S_fine = 20000
            samples_fine = _fibonacci_sphere(S_fine, seed=seed + 200)
            P, _ = _fast_local_search(P, samples_fine, current_deadline - 3.0, max_iters=150)
        
        # 5. Evaluate
        score = _covering_radius_deg(P)
        if score < best_score:
            best_score = score
            best_P = P.copy()
    
    # Final refinement with best configuration
    if best_P is not None and time.time() < deadline - 8.0:
        S_final = 25000
        samples_final = _fibonacci_sphere(S_final, seed=9999)
        best_P, _ = _repulsion_local_search(best_P, samples_final, deadline - 3.0, max_iters=80)
    
    return _normalize(best_P if best_P is not None else _fibonacci_sphere(N))

# EVOLVE-BLOCK-END
