import time
import numpy as np
import math
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def _compute_covering_radius(pts):
    """
    Compute the covering radius in radians.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return math.pi
        
    simplices = hull.simplices
    max_rad = 0.0
    
    for simplex in simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        if np.dot(n, v0) < 0:
            n = -n
            
        d0 = np.dot(v0, n)
        d1 = np.dot(v1, n)
        d2 = np.dot(v2, n)
        
        min_d = np.min([d0, d1, d2])
        min_d = np.clip(min_d, -1.0, 1.0)
        rad = np.arccos(min_d)
        
        if rad > max_rad:
            max_rad = rad
            
    return max_rad

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

def _compute_covering_radius(pts):
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return math.pi
        
    simplices = hull.simplices
    max_rad = 0.0
    
    for simplex in simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        if np.dot(n, v0) < 0:
            n = -n
            
        d0 = np.dot(v0, n)
        d1 = np.dot(v1, n)
        d2 = np.dot(v2, n)
        
        min_d = np.min([d0, d1, d2])
        min_d = np.clip(min_d, -1.0, 1.0)
        rad = np.arccos(min_d)
        
        if rad > max_rad:
            max_rad = rad
            
    return max_rad

def _compute_covering_radius_and_grads(pts):
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return math.pi, np.zeros_like(pts)

    simplices = hull.simplices
    n_faces = len(simplices)
    
    centers = np.zeros((n_faces, 3))
    min_dots = np.zeros((n_faces, 3))
    radii = np.zeros(n_faces)
    
    for i, simplex in enumerate(simplices):
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            radii[i] = 0.0
            continue
        n = n / n_norm
        
        if np.dot(n, v0) < 0:
            n = -n
            
        centers[i] = n
        
        d0 = np.dot(v0, n)
        d1 = np.dot(v1, n)
        d2 = np.dot(v2, n)
        
        min_dots[i, 0] = d0
        min_dots[i, 1] = d1
        min_dots[i, 2] = d2
        
        min_d = np.min(min_dots[i])
        min_d = np.clip(min_d, -1.0, 1.0)
        radii[i] = np.arccos(min_d)
        
    max_radius = np.max(radii)
    
    threshold = max_radius - 0.001 
    active_mask = radii > threshold
    
    if not np.any(active_mask):
        return max_radius, np.zeros_like(pts)
    
    grads = np.zeros_like(pts)
    
    active_indices = np.where(active_mask)[0]
    
    for i in active_indices:
        simplex = simplices[i]
        n = centers[i]
        
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        g0 = n - np.dot(n, v0) * v0
        g1 = n - np.dot(n, v1) * v1
        g2 = n - np.dot(n, v2) * v2
        
        g0_norm = np.linalg.norm(g0)
        g1_norm = np.linalg.norm(g1)
        g2_norm = np.linalg.norm(g2)
        
        if g0_norm > 1e-9: g0 /= g0_norm
        if g1_norm > 1e-9: g1 /= g1_norm
        if g2_norm > 1e-9: g2 /= g2_norm
        
        grads[simplex[0]] += g0
        grads[simplex[1]] += g1
        grads[simplex[2]] += g2
        
    return max_radius, grads

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Global Phase: Thomson Problem Minimization
    def thomson_energy_and_grad(P):
        P = _normalize(P)
        U = 2.0 - 2.0 * (P @ P.T)
        U_diag_mask = ~np.eye(N, dtype=bool)
        U_safe = np.where(U_diag_mask, U, 1.0)
        
        E_terms = np.zeros_like(U)
        E_terms[U_diag_mask] = U_safe[U_diag_mask] ** -3
        E = np.sum(E_terms)
        
        W = np.zeros_like(U)
        W[U_diag_mask] = 6.0 * U_safe[U_diag_mask] ** -4
        grad = W @ P
        
        return E, grad

    def _thomson_optimize(P, max_iters=500, lr=0.01):
        P = _normalize(P)
        for i in range(max_iters):
            if time.time() > deadline - 60.0:
                break
            E, grad = thomson_energy_and_grad(P)
            grad = grad - np.sum(grad * P, axis=1, keepdims=True) * P
            P = P - lr * grad
            P = _normalize(P)
        return P

    # Initialize with Fibonacci
    P = _fibonacci_sphere(N, seed_noise=0.1, seed=42)
    
    # Phase 1: Thomson
    P = _thomson_optimize(P)
    
    best_P = P.copy()
    best_radius = _compute_covering_radius(P)
    
    # Phase 2: Local Refinement (Annealing on Covering Radius)
    rng = np.random.default_rng(42)
    
    max_iterations = 2000
    step_size = 0.05
    cool_rate = 0.995
    acceptance_temp = 0.001
    
    temp = acceptance_temp
    for it in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
            
        radius, grads = _compute_covering_radius_and_grads(P)
        
        if radius < best_radius:
            best_radius = radius
            best_P = P.copy()
            
        noise = rng.normal(0, 0.01, (N, 3))
        noise = noise - np.sum(noise * P, axis=1, keepdims=True) * P
        grads += noise * temp
        
        new_P = P + step_size * grads
        new_P = _normalize(new_P)
        
        new_radius, _ = _compute_covering_radius_and_grads(new_P)
        
        delta_r = new_radius - radius
        if delta_r < 0:
            P = new_P
        else:
            if rng.random() < math.exp(-delta_r / temp):
                P = new_P
        
        temp *= cool_rate
        step_size *= 0.999
            
    return best_P
# EVOLVE-BLOCK-END
