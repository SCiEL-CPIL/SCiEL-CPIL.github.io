"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import sys


def _covering_radius(pts):
    """Compute the covering radius (in radians) of a set of points on the unit sphere."""
    pts = np.asarray(pts)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
        
        # Find circumcenter of the spherical triangle (a,b,c)
        u = np.cross(a - b, a - c)
        u_norm = np.linalg.norm(u)
        
        if u_norm < 1e-10:
            continue
            
        u = u / u_norm
        
        # Ensure u is on the same side as the facet normal (outward)
        normal = np.cross(b - a, c - a)
        if np.dot(u, normal) < 0:
            u = -u
        
        # Compute circumradius
        dot_val = np.clip(np.dot(u, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        
        if r > max_circumradius:
            max_circumradius = r
    
    return max_circumradius


def _covering_gradient(pts, power=20.0):
    """Compute the gradient of the sum of circumradii^power for all hull facets."""
    pts = np.asarray(pts)
    n = len(pts)
    grad = np.zeros_like(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return grad

    for simplex in hull.simplices:
        a_idx, b_idx, c_idx = simplex
        a, b, c = pts[a_idx], pts[b_idx], pts[c_idx]
        
        # Circumcenter
        u = np.cross(a - b, a - c)
        u_norm = np.linalg.norm(u)
        
        if u_norm < 1e-10:
            continue
            
        u = u / u_norm
        
        normal = np.cross(b - a, c - a)
        if np.dot(u, normal) < 0:
            u = -u
        
        # Circumradius
        dot_val = np.clip(np.dot(u, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        
        if r < 1e-8:
            continue
            
        # d(r^power)/dr = power * r^(power-1)
        dval_dr = power * (r ** (power - 1))
        
        # dr/dp for each vertex
        # r = arccos(u.p)
        # dr/dp = -1/sin(r) * (u - (u.p)p)
        sin_r = np.sin(r)
        if sin_r < 1e-8:
            continue
            
        coeff = dval_dr / sin_r
        
        # Gradient contribution for a
        # grad_a += coeff * ( (u.p)a - u )  ... wait
        # d(arccos(u.p))/dp = -1/sin(r) * u
        # But p is on sphere, we need tangent projection.
        # The gradient in R3 is -u/sin(r).
        # Projected to tangent space at a: -u/sin(r) - ((-u/sin(r)).a) a
        # = -1/sin(r) * (u - (u.a)a)
        
        u_dot_a = np.dot(u, a)
        grad_a = coeff * (u_dot_a * a - u)
        
        u_dot_b = np.dot(u, b)
        grad_b = coeff * (u_dot_b * b - u)
        
        u_dot_c = np.dot(u, c)
        grad_c = coeff * (u_dot_c * c - u)
        
        grad[a_idx] += grad_a
        grad[b_idx] += grad_b
        grad[c_idx] += grad_c
        
    return grad


def _repulsive_gradient(pts, power=2.0):
    """Compute the gradient of the repulsive energy sum(1/d^power) projected onto tangent spaces."""
    pts = np.asarray(pts)
    n = len(pts)
    grad = np.zeros_like(pts)
    
    dots = np.dot(pts, pts.T)
    mask = ~np.eye(n, dtype=bool)
    
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    dists = np.arccos(dots)
    sins = np.sin(dists)
    
    eps = 1e-10
    safe_dists = np.where(dists < eps, eps, dists)
    safe_sins = np.where(sins < eps, eps, sins)
    
    weights = np.zeros_like(dists)
    np.fill_diagonal(weights, 0.0)
    valid = mask
    weights[valid] = 1.0 / (safe_dists[valid] ** (power + 1) * safe_sins[valid])
    
    rep_grad = power * (weights @ pts)
    
    for i in range(n):
        radial_comp = np.dot(rep_grad[i], pts[i])
        grad[i] = rep_grad[i] - radial_comp * pts[i]
        
    return grad


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    best_cr = np.inf
    best_pts = None
    
    # Parameters
    alpha_cover = 1.0
    alpha_repel = 0.1
    power_cover = 20.0
    power_repel = 2.0
    
    for restart in range(10):
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Random initialization
            np.random.seed(42 + restart)
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Optimize
        lr = 0.01
        n_iter = 1500
        momentum = 0.9
        v = np.zeros_like(pts)
        
        prev_cr = np.inf
        patience = 100
        no_improve = 0
        
        for it in range(n_iter):
            # Compute gradients
            grad_cover = _covering_gradient(pts, power=power_cover)
            grad_repel = _repulsive_gradient(pts, power=power_repel)
            
            grad = alpha_cover * grad_cover + alpha_repel * grad_repel
            
            # Momentum update
            v = momentum * v - lr * grad
            pts_new = pts + v
            
            # Normalize to unit sphere
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            pts_new = pts_new / norms
            
            pts = pts_new
            
            # Check improvement periodically
            if it % 10 == 0:
                cr = _covering_radius(pts)
                if cr < best_cr:
                    best_cr = cr
                    best_pts = pts.copy()
                
                if cr < prev_cr - 1e-6:
                    prev_cr = cr
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if no_improve > patience:
                    break
                if cr < 0.32: # ~18.3 degrees
                    break
                    
    # Final refinement
    if best_pts is not None:
        pts = best_pts.copy()
        lr = 0.001
        for it in range(500):
            grad_cover = _covering_gradient(pts, power=power_cover)
            grad_repel = _repulsive_gradient(pts, power=power_repel)
            grad = alpha_cover * grad_cover + alpha_repel * grad_repel
            
            pts_new = pts - lr * grad
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            pts_new = pts_new / norms
            pts = pts_new
            
            if it % 50 == 0:
                cr = _covering_radius(pts)
                if cr < best_cr:
                    best_cr = cr
                    best_pts = pts.copy()

    return best_pts
# EVOLVE-BLOCK-END
