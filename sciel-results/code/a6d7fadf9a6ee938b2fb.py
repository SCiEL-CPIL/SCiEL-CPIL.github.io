"""Program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
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

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) >= 3:
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

def _get_min_dists(P, samples):
    dots = samples @ P.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    nearest_idx = np.argmin(dists, axis=1)
    return min_dists, nearest_idx

def _thomson_init(n, seed=42):
    """Thomson (electrostatic) initialization via gradient descent on repulsion energy."""
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n, 3))
    P = _normalize(P)
    
    def energy_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        
        dots = P @ P.T
        dist_sq = 2.0 - 2.0 * dots
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_sq = 1.0 / np.maximum(dist_sq, 1e-10)
        np.fill_diagonal(inv_dist_sq, 0.0)
        energy = np.sum(inv_dist_sq)
        
        grad = np.zeros_like(P)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d_sq = dist_sq[i, j]
                if d_sq < 1e-10:
                    continue
                grad[i] += 4.0 * P[j] / (d_sq * d_sq)
        
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        
        return energy, grad_tangent.flatten()
    
    try:
        res = minimize(
            energy_and_grad,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    return P

def _smooth_max_surrogate(P, samples, temperature=0.05):
    """
    Compute a smooth approximation of the max distance using log-sum-exp.
    Returns (value, gradient) where gradient is w.r.t. the flattened P.
    """
    dots = samples @ P.T
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    
    t = 1.0 / temperature
    max_d = np.max(min_dists)
    exp_shifted = np.exp(t * (min_dists - max_d))
    log_sum = np.log(np.sum(exp_shifted))
    value = max_d + log_sum / t
    
    weights = exp_shifted / np.sum(exp_shifted)
    
    nearest_idx = np.argmin(dists, axis=1)
    grad_P = np.zeros((N, 3))
    
    for s in range(len(nearest_idx)):
        j = nearest_idx[s]
        w = weights[s]
        if w < 1e-15:
            continue
        d_val = dots[s, j]
        denom = np.sqrt(max(1.0 - d_val * d_val, 1e-15))
        grad_P[j] -= w * samples[s] / denom
    
    return value, grad_P

def _spectral_uniformity_energy(P, max_l=10):
    """
    Compute spectral energy: variance of pair correlation function via spherical harmonics.
    This promotes uniformity by penalizing large-scale clustering.
    """
    P = _normalize(P)
    
    from scipy.special import lpmv
    
    energy = 0.0
    for l in range(1, max_l + 1):
        dots = P @ P.T
        P_l = lpmv(0, l, dots)
        
        mult_sum = np.sum(P_l)
        C_l = (2 * l + 1) / (4 * np.pi) * mult_sum
        
        energy += C_l * C_l
    
    return energy

def _polish_greedy(P, samples, deadline, max_iters=500, rng_seed=42):
    """Greedy local search: move the point nearest to the worst sample toward it."""
    min_dists, nearest_idx = _get_min_dists(P, samples)
    current_max = np.max(min_dists)
    rng = np.random.default_rng(rng_seed)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        min_dists, nearest_idx = _get_min_dists(P, samples)
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        
        p_idx = nearest_idx[worst_idx]
        p_old = P[p_idx].copy()
        
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-8:
            continue
        
        grad_dir = grad_dir / grad_norm
        
        # Try several step sizes
        best_step = 0.0
        best_new_max = current_max
        best_p = p_old.copy()
        
        for step_size in [0.08, 0.04, 0.02, 0.01, 0.005, 0.002]:
            new_p = p_old + step_size * grad_dir
            new_p = _normalize(new_p.reshape(1, 3))[0]
            
            P[p_idx] = new_p
            min_dists_temp, _ = _get_min_dists(P, samples)
            new_max = np.max(min_dists_temp)
            
            if new_max < best_new_max - 1e-12:
                best_new_max = new_max
                best_step = step_size
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_step > 0:
            P[p_idx] = best_p
            current_max = best_new_max
        else:
            # Try a random perturbation to escape local minimum
            random_dir = rng.standard_normal(3)
            random_dir = random_dir - np.dot(random_dir, p_old) * p_old
            random_norm = np.linalg.norm(random_dir)
            if random_norm > 1e-6:
                random_dir = random_dir / random_norm
            else:
                random_dir = np.zeros(3)
            
            for step_size in [0.02, 0.01, 0.005]:
                new_p = p_old + step_size * random_dir
                new_p = _normalize(new_p.reshape(1, 3))[0]
                
                P[p_idx] = new_p
                min_dists_temp, _ = _get_min_dists(P, samples)
                new_max = np.max(min_dists_temp)
                
                if new_max < best_new_max - 1e-12:
                    best_new_max = new_max
                    best_step = step_size
                    best_p = new_p.copy()
                else:
                    P[p_idx] = p_old
            
            if best_step > 0:
                P[p_idx] = best_p
                current_max = best_new_max
            else:
                P[p_idx] = p_old
    
    return P, current_max

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initialization: Thomson (electrostatic) init
    P = _thomson_init(N, seed=42)
    
    # 2. Generate samples for optimization
    S_opt = 20000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # Phase 1: Riemannian optimization on the smooth max surrogate
    def objective_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        val, grad = _smooth_max_surrogate(P, samples_opt, temperature=0.03)
        
        # Project gradient to tangent space
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        
        return val, grad_tangent.flatten()
    
    try:
        res = minimize(
            objective_and_grad,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 40.0:
        S_polish = 25000
        samples_polish = _fibonacci_sphere(S_polish)
        P, _ = _polish_greedy(P, samples_polish, deadline - 5.0, max_iters=100)
        return P
    
    # Phase 2: Second round of smooth max optimization with different temperature
    samples_opt2 = _fibonacci_sphere(15000, seed_noise=0.05, seed=123)
    
    def objective_and_grad2(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        val, grad = _smooth_max_surrogate(P, samples_opt2, temperature=0.02)
        
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        
        return val, grad_tangent.flatten()
    
    try:
        res = minimize(
            objective_and_grad2,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    # Phase 3: High-resolution greedy polish
    S_polish = 40000
    samples_polish = _fibonacci_sphere(S_polish)
    
    P, current_max = _polish_greedy(P, samples_polish, deadline - 5.0, max_iters=300)
    
    return P
# EVOLVE-BLOCK-END
