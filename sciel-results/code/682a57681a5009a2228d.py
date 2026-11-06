"""Program for covering the sphere with 130 equal caps."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    # 1. Initialization
    P = _fibonacci_sphere(N, seed_noise=0.005)
    
    # 2. Prepare samples for optimization
    S_opt = 8000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # 3. Model-and-optimize approach: 
    # We parameterize the points directly and use a robust objective.
    # The key idea: minimize the sum of powers of the min-distance to samples.
    # Using p-norm with high p approximates min-max but is smoother than direct max.
    
    def make_objective(P_current):
        def objective(P_flat):
            P = P_flat.reshape(-1, 3)
            norms = np.linalg.norm(P, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            P = P / norms
            
            dots = samples_opt @ P.T
            dots = np.clip(dots, -1.0, 1.0)
            # Use chordal distance squared as proxy, which is smoother
            # d^2 = 2 - 2*cos(theta)
            # We want to minimize the max of theta, which is monotonic with d^2
            dist_sq = 2.0 - 2.0 * dots  # (S, N)
            min_dist_sq = np.min(dist_sq, axis=1)  # (S,)
            
            # Use a high power to emphasize the maximum
            # Sum of (min_dist_sq)^p
            p_power = 20
            loss = np.sum(np.power(np.maximum(min_dist_sq, 1e-10), p_power))
            
            return loss
        
        def gradient(P_flat):
            P = P_flat.reshape(-1, 3)
            norms = np.linalg.norm(P, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            P = P / norms
            
            dots = samples_opt @ P.T
            dots = np.clip(dots, -1.0, 1.0)
            dist_sq = 2.0 - 2.0 * dots  # (S, N)
            
            min_idx = np.argmin(dist_sq, axis=1)  # (S,)
            min_dist_sq = dist_sq[np.arange(len(min_idx)), min_idx]
            min_dist_sq_clipped = np.maximum(min_dist_sq, 1e-10)
            
            # d(loss)/d(dist_sq_j) = p * (dist_sq_j)^(p-1) for the min one
            grad_dist_sq = np.zeros_like(dist_sq)
            grad_dist_sq[np.arange(len(min_idx)), min_idx] = 20.0 * np.power(min_dist_sq_clipped, 19.0)
            
            # d(dist_sq)/d(P) = -2 * samples
            # grad_P_ambient[s, i] = sum over j where j is min for sample s of grad_dist_sq[s, j] * (-2) * samples[s, i]
            # This is a sparse gradient
            grad_P_ambient = np.zeros((N, 3))
            for s in range(len(min_idx)):
                j = min_idx[s]
                grad_P_ambient[j] -= 2.0 * grad_dist_sq[s, j] * samples_opt[s]
            
            # Project to tangent space
            grad_dot_P = np.sum(grad_P_ambient * P, axis=1, keepdims=True)
            grad_tangent = grad_P_ambient - grad_dot_P * P
            
            norms_flat = np.linalg.norm(P_flat.reshape(-1, 3), axis=1, keepdims=True)
            norms_flat[norms_flat < 1e-15] = 1.0
            grad_P_raw = grad_tangent / norms_flat
            
            return grad_P_raw.flatten()
        
        return objective, gradient
    
    # Phase 1: BFGS optimization
    try:
        obj, grad = make_objective(P)
        res = minimize(
            obj,
            grad,
            P.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30.0:
        # Do a quick polish
        S_polish = 20000
        samples_polish = _fibonacci_sphere(S_polish)
        min_dists, nearest_idx = _get_min_dists(P, samples_polish)
        current_max = np.max(min_dists)
        
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            worst_idx = np.argmax(min_dists)
            worst_sample = samples_polish[worst_idx]
            p_idx = nearest_idx[worst_idx]
            p_old = P[p_idx].copy()
            
            proj = np.dot(p_old, worst_sample)
            grad_dir = worst_sample - proj * p_old
            grad_norm = np.linalg.norm(grad_dir)
            if grad_norm < 1e-6:
                continue
            grad_dir = grad_dir / grad_norm
            
            best_step = 0.0
            best_new_max = current_max
            best_p = p_old.copy()
            
            for step_size in [0.05, 0.02, 0.01]:
                new_p = p_old + step_size * grad_dir
                new_p = _normalize(new_p.reshape(1, 3))[0]
                P[p_idx] = new_p
                min_dists_temp, _ = _get_min_dists(P, samples_polish)
                new_max = np.max(min_dists_temp)
                if new_max < best_new_max:
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
            
            min_dists, nearest_idx = _get_min_dists(P, samples_polish)
        
        return P
    
    # 4. Polish with high-resolution greedy local search
    S_polish = 30000
    samples_polish = _fibonacci_sphere(S_polish)
    
    min_dists, nearest_idx = _get_min_dists(P, samples_polish)
    current_max_dist = np.max(min_dists)
    
    rng = np.random.default_rng(42)
    
    # Greedy polish loop
    for _ in range(200):
        if time.time() > deadline - 10.0:
            break
            
        min_dists, nearest_idx = _get_min_dists(P, samples_polish)
        worst_idx = np.argmax(min_dists)
        worst_sample = samples_polish[worst_idx]
        
        p_idx = nearest_idx[worst_idx]
        p_old = P[p_idx].copy()
        
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-6:
            continue
            
        grad_dir = grad_dir / grad_norm
        
        # Add small random noise to escape local minima
        random_dir = rng.standard_normal(3)
        random_dir = random_dir - np.dot(random_dir, p_old) * p_old
        random_norm = np.linalg.norm(random_dir)
        if random_norm > 1e-6:
            random_dir = random_dir / random_norm
        else:
            random_dir = np.zeros(3)
            
        best_step = 0.0
        best_new_max = current_max_dist
        best_p = p_old.copy()
        
        for step_size in [0.05, 0.02, 0.01, 0.005]:
            new_p = p_old + step_size * (0.9 * grad_dir + 0.1 * random_dir)
            new_p = _normalize(new_p.reshape(1, 3))[0]
            
            P[p_idx] = new_p
            min_dists_temp, _ = _get_min_dists(P, samples_polish)
            new_max = np.max(min_dists_temp)
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_step = step_size
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_step > 0:
            P[p_idx] = best_p
            current_max_dist = best_new_max
        else:
            P[p_idx] = p_old
    
    return P
# EVOLVE-BLOCK-END
