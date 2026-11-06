"""Combined program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
import math
from scipy.spatial import ConvexHull

N = 130
DEADLINE = 125.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use pairwise distances
        d = 1.0 - pts @ pts.T
        np.fill_diagonal(d, 0.0)
        d = np.maximum(d, 0.0)
        return np.degrees(np.arccos(np.clip(1.0 - np.max(d), -1.0, 1.0)))
    
    max_angle = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        for sign in [1.0, -1.0]:
            x = sign * n
            da = np.dot(x, a)
            db = np.dot(x, b)
            dc = np.dot(x, c)
            if abs(da - db) < 1e-8 and abs(da - dc) < 1e-8:
                angle = np.arccos(np.clip(da, -1.0, 1.0))
                if angle > max_angle:
                    max_angle = angle
                break
    
    return np.degrees(max_angle)


def _fast_covering_radius(pts):
    """Fast approximation of covering radius using nearest-neighbor distances."""
    pts = _normalize(pts)
    d = 1.0 - pts @ pts.T
    np.fill_diagonal(d, 0.0)
    d = np.maximum(d, 0.0)
    min_d = np.min(d, axis=1)
    max_d = np.max(min_d)
    return np.degrees(np.arccos(np.clip(1.0 - max_d, -1.0, 1.0)))


def _fibonacci_sphere(n, phase=0.0, rng=None, noise=0.0):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + phase
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    if noise > 0:
        if rng is None:
            rng = np.random.default_rng(42)
        pts += rng.normal(0, noise, (n, 3))
        pts = _normalize(pts)
    
    return pts


def _thomson_init(n, iters=500, rng=None):
    """Thomson electrostatic initialization for uniform points on sphere."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    pts = rng.normal(size=(n, 3))
    pts = _normalize(pts)
    
    for it in range(iters):
        # Compute forces
        dots = pts @ pts.T
        np.fill_diagonal(dots, 0.0)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.sqrt(2.0 - 2.0 * dots)
        dists = np.maximum(dists, 1e-10)
        
        # Force: 1/r^2 repulsion
        force_mat = np.zeros_like(dists)
        np.fill_diagonal(force_mat, 0.0)
        force_mat = 1.0 / (dists**2)
        
        # Sum forces for each point
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i != j:
                    forces[i] += force_mat[i, j] * (pts[j] - pts[i])
        
        # Move points
        step = 0.01 * (1.0 - it / iters)
        pts += step * forces
        pts = _normalize(pts)
    
    return pts


def _combined_loss(P_flat, samples, lam=8.0, spectral_weight=0.2, target_dist=None):
    """
    Combined loss: Soft-max covering + pairwise distance variance.
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms
    n = len(Pn)

    if target_dist is None:
        target_dist = math.sqrt(4 * np.pi / n)

    # 1. Soft-max covering loss
    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)
    
    cover_loss = np.sum(np.exp(lam * min_dists))
    
    weights = lam * np.exp(lam * min_dists)
    dots_nearest = dots[np.arange(len(samples)), nearest]
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
    grad_factor = -weights / denom
    
    grad_P = np.zeros_like(Pn)
    np.add.at(grad_P, nearest, grad_factor[:, None] * samples)

    # 2. Spectral/Uniformity loss (Pairwise distance variance)
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


def _fast_local_search(P, samples, deadline, max_iters=200):
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


def construct():
    t_start = time.time()
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting configurations
    starts = []
    
    # Fibonacci spirals with different phases
    for phase in [0.0, 0.5, 1.0, 1.5, 2.0]:
        pts = _fibonacci_sphere(N, phase=phase, rng=rng, noise=0.03)
        starts.append(pts)
    
    # Thomson initialization
    try:
        thomson_pts = _thomson_init(N, iters=300, rng=rng)
        starts.append(thomson_pts)
    except Exception:
        pass
    
    # Random starts
    for _ in range(3):
        pts = rng.normal(size=(N, 3))
        starts.append(_normalize(pts))
    
    best_pts = None
    best_score = np.inf
    
    for s_idx, start_pts in enumerate(starts):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 25:
            break
        
        # Phase 1: L-BFGS optimization
        try:
            P_flat = start_pts.flatten()
            
            # Use a moderate number of samples for optimization
            S_opt = 10000
            samples_opt = _fibonacci_sphere(S_opt, seed=42 + s_idx * 100)
            
            opt_deadline = t_start + DEADLINE - 15 - (len(starts) - s_idx - 1) * 10
            
            def time_callback(xk):
                if time.time() > opt_deadline:
                    raise ValueError("Time limit exceeded")
            
            result = minimize(
                _combined_loss,
                P_flat,
                args=(samples_opt,),
                method='L-BFGS-B',
                jac=True,
                callback=time_callback,
                options={'maxiter': 100, 'gtol': 1e-4}
            )
            
            opt_pts = _normalize(result.x.reshape(-1, 3))
        except Exception:
            opt_pts = start_pts
        
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 15:
            break
        
        # Phase 2: Fast local search
        try:
            samples_ls = _fibonacci_sphere(5000, seed=42 + s_idx * 200)
            ls_deadline = min(t_start + DEADLINE - 5, time.time() + 10)
            ls_pts, _ = _fast_local_search(opt_pts, samples_ls, ls_deadline, max_iters=100)
        except Exception:
            ls_pts = opt_pts
        
        # Phase 3: Final local search polish
        try:
            samples_polish = _fibonacci_sphere(3000, seed=42 + s_idx * 300)
            polish_deadline = min(t_start + DEADLINE - 2, time.time() + 5)
            final_pts, _ = _fast_local_search(ls_pts, samples_polish, polish_deadline, max_iters=50)
        except Exception:
            final_pts = ls_pts
        
        # Evaluate
        try:
            score = _covering_radius_deg(final_pts)
        except Exception:
            score = np.inf
        
        if score < best_score:
            best_score = score
            best_pts = final_pts.copy()
        
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 10:
            break
    
    # If we didn't find anything, use the first Fibonacci spiral
    if best_pts is None:
        best_pts = starts[0]
    
    # Final normalization and deduplication
    best_pts = _normalize(best_pts)
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                best_pts[j] += rng.normal(size=3) * 1e-4
                best_pts[j] = _normalize(best_pts[j].reshape(1, 3))[0]
    
    return best_pts

# EVOLVE-BLOCK-END
