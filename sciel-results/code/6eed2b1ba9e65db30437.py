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
    
    # Minimize -sum of log distances (or sum of 1/r^2) to spread points
    # Use repulsion energy: E = -sum_{i!=j} log(||p_i - p_j||)
    # or simpler: minimize sum_{i!=j} 1/||p_i - p_j||^2
    
    def energy_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        
        # Compute pairwise distances
        dots = P @ P.T  # (n, n)
        # Distance squared: 2 - 2*cos(theta)
        dist_sq = 2.0 - 2.0 * dots
        np.fill_diagonal(dist_sq, 1.0)
        
        # Energy: sum of 1/dist_sq (repulsion)
        # Avoid division by zero
        inv_dist_sq = 1.0 / np.maximum(dist_sq, 1e-10)
        np.fill_diagonal(inv_dist_sq, 0.0)
        energy = np.sum(inv_dist_sq)
        
        # Gradient: d(1/d^2)/d(p_i) where d^2 = 2 - 2*p_i.p_j
        # = -2/d^4 * d(d^2)/d(p_i) = -2/d^4 * (-2*p_j) = 4*p_j/d^4
        # But we need to project to tangent space
        grad = np.zeros_like(P)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d_sq = dist_sq[i, j]
                if d_sq < 1e-10:
                    continue
                grad[i] += 4.0 * P[j] / (d_sq * d_sq)
        
        # Project to tangent space
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        
        return energy, grad_tangent.flatten()
    
    try:
        res = minimize(
            energy_and_grad,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
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
    dists = np.arccos(dots)  # (S, N)
    min_dists = np.min(dists, axis=1)  # (S,)
    
    # Log-sum-exp for smooth max
    t = 1.0 / temperature
    max_d = np.max(min_dists)
    exp_shifted = np.exp(t * (min_dists - max_d))
    log_sum = np.log(np.sum(exp_shifted))
    value = max_d + log_sum / t
    
    # Gradient w.r.t. min_dists
    weights = exp_shifted / np.sum(exp_shifted)  # (S,)
    
    # Gradient w.r.t. P: for each sample s, the nearest point j
    nearest_idx = np.argmin(dists, axis=1)  # (S,)
    grad_P = np.zeros((N, 3))
    
    # d(min_dists)/d(P[j]) for the nearest point j to sample s
    # min_dists[s] = arccos(dots[s, j]) where j = nearest_idx[s]
    # d(arccos(x))/dx = -1/sqrt(1-x^2)
    # d(dots[s,j])/d(P[j]) = samples[s]
    # So d(min_dists[s])/d(P[j]) = -samples[s] / sqrt(1 - dots[s,j]^2)
    
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
    
    # Compute the pair correlation function via the power spectrum
    # For a set of points on the sphere, the power spectrum at degree l is:
    # C_l = (1/N^2) * |sum_i sum_j Y_l^m(p_i) Y_l^m(p_j)^*|^2 type quantity
    # Simpler: use the fact that sum_{l,m} (2l+1)/(4pi) |Y_lm(f)|^2 = |f|^2
    # where f is the density. For point measure, we look at the multipole moments.
    
    # Alternative simpler approach: compute the variance of the local density
    # by binning the sphere into caps and looking at the variance of counts.
    # But the directive says to use spherical harmonics.
    
    # Let's compute the power spectrum of the point distribution.
    # The density is rho(p) = sum_i delta(p - p_i)
    # The power spectrum is C_l = (1/(2l+1)) sum_m |Y_lm(rho)|^2
    # For a discrete set, Y_lm(rho) = sum_i Y_lm(p_i)
    
    # We'll compute the variance of the density over the sphere using the
    # first few multipole moments. A uniform distribution has C_l = 0 for l > 0.
    
    # Use associated Legendre polynomials to compute Y_lm
    # For simplicity, compute the "dipole" and "quadrupole" moments as proxies.
    
    # Actually, let's use a more direct approach: compute the variance of the
    # number of neighbors within a fixed angular radius.
    
    # For now, let's use the sum of squared multipole moments as a proxy for uniformity.
    # M_l = |sum_i Y_lm(p_i)|^2 for various l, m
    
    # We'll compute the first few l values
    energy = 0.0
    for l in range(1, max_l + 1):
        # For each l, sum over m of |sum_i Y_lm(p_i)|^2
        # Use the addition theorem: sum_m Y_lm(p) Y_lm(q)^* = (2l+1)/(4pi) P_l(p.q)
        # So sum_m |sum_i Y_lm(p_i)|^2 = sum_m sum_i sum_j Y_lm(p_i) Y_lm(p_j)^*
        # = sum_i sum_j sum_m Y_lm(p_i) Y_lm^*(p_j)
        # = sum_i sum_j (2l+1)/(4pi) P_l(p_i . p_j)
        
        dots = P @ P.T  # (N, N)
        # Compute P_l for each pair
        # Use numpy's polynomial or scipy
        from scipy.special import lpmv
        # lpmv(m, l, x) gives associated Legendre P_l^m(x)
        # We need P_l(x) = lpmv(0, l, x)
        P_l = lpmv(0, l, dots)  # (N, N)
        
        # sum_m |sum_i Y_lm(p_i)|^2 = (2l+1)/(4pi) * sum_{i,j} P_l(p_i . p_j)
        mult_sum = np.sum(P_l)
        C_l = (2 * l + 1) / (4 * np.pi) * mult_sum
        
        # For a uniform distribution, C_l should be 0 for l > 0
        # We want to minimize the sum of C_l^2 for l >= 1
        energy += C_l * C_l
    
    return energy

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    # 1. Initialization: Use Thomson (electrostatic) init for better starting point
    P = _thomson_init(N, seed=42)
    
    # 2. Generate samples for optimization
    S_opt = 15000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # Phase 1: Riemannian optimization on the smooth max surrogate
    # Use log-sum-exp for smooth max of min-distances
    
    def objective_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        val, grad = _smooth_max_surrogate(P, samples_opt, temperature=0.05)
        
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
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
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
    
    # Phase 2: Spectral uniformity refinement
    # Use the spectral energy to encourage more uniform distribution
    # This is a different objective than the covering radius, so it can help
    # escape local minima of the covering radius optimization.
    
    def spectral_obj_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)
        
        # Compute spectral energy
        energy = _spectral_uniformity_energy(P, max_l=6)
        
        # For gradient, we need d(energy)/d(P)
        # This is complex, so we'll use numerical gradient or skip
        # For now, just return the value and zero gradient (will use finite diff)
        return energy
    
    # Use the spectral energy as a secondary objective in a combined optimization
    # Or use it to perturb the configuration to escape local minima
    
    # Phase 3: Polish with high-resolution greedy local search
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
