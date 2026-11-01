"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _compute_radius(centers):
    """Compute the valid packing radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall constraint: min distance from any center to any wall
    wall_dist = np.min(0.5 - np.abs(centers))
    if wall_dist < -1e-9:
        return 0.0
    # Pairwise distances
    n = centers.shape[0]
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    r = min(min_pair / 2.0, wall_dist)
    return max(r, 0.0)


def _soft_min_obj_and_grad(centers, beta):
    """
    Objective: -soft_min of all pairwise distances and wall distances.
    We want to maximize the minimum of {d_ij/2 for i<j} U {0.5-|x_ik| for all i,k}.
    
    Using log-sum-exp soft-min: soft_min(v) = -(1/beta) * log(sum(exp(-beta*v)))
    We minimize -soft_min, i.e., maximize soft_min.
    
    Actually, let's define:
    values = [d_ij/2 for all pairs] + [0.5 - |x_ik| for all i,k]
    We want to maximize min(values).
    
    soft_min(v) = -(1/beta) * log(sum(exp(-beta * v)))
    This is a smooth lower bound on min(v).
    
    Objective to minimize: -soft_min(values) = (1/beta) * log(sum(exp(-beta * v)))
    
    Gradient: d/dx [(1/beta)*log(sum(exp(-beta*v)))]
             = (1/beta) * (1/sum(exp(-beta*v))) * sum(-beta * exp(-beta*v) * dv/dx)
             = -sum(exp(-beta*v) * dv/dx) / sum(exp(-beta*v))
             = -sum(w_k * dv_k/dx) where w_k = exp(-beta*v_k)/sum(exp(-beta*v_j))
    """
    n = centers.shape[0]
    centers = centers.reshape(n, 2)
    
    # Collect all values and their gradients
    # We'll compute log-sum-exp in a numerically stable way
    
    # First pass: compute all values
    vals = []
    # Pairwise distances / 2
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            vals.append(d / 2.0)
    # Wall distances
    for i in range(n):
        for k in range(2):
            vals.append(0.5 - abs(centers[i, k]))
    
    vals = np.array(vals)
    
    # Numerically stable log-sum-exp of -beta * vals
    beta_vals = -beta * vals
    max_bv = np.max(beta_vals)
    if max_bv > 700:
        # Avoid overflow
        exp_vals = np.exp(beta_vals - max_bv)
        log_sum = max_bv + np.log(np.sum(exp_vals))
    else:
        exp_vals = np.exp(beta_vals)
        log_sum = np.log(np.sum(exp_vals))
    
    obj = log_sum / beta
    
    # Weights for gradient
    w = exp_vals / np.sum(exp_vals)
    
    # Gradient computation
    grad = np.zeros_like(centers)
    
    # Pairwise gradients
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                # Degenerate case
                continue
            # d(v)/d(centers[i]) = diff/d, d(v)/d(centers[j]) = -diff/d
            # v = d/2, so dv/d(centers[i]) = diff/(2*d), dv/d(centers[j]) = -diff/(2*d)
            wi = w[idx]
            grad[i] += wi * diff / (2.0 * d) * (-1)  # because obj gradient is -sum(w*dv/dx)
            grad[j] += wi * (-diff) / (2.0 * d) * (-1)
            # Wait, let me re-derive.
            # obj = (1/beta)*log(sum(exp(-beta*v_k)))
            # d(obj)/d(x) = (1/beta) * (1/S) * sum(-beta * exp(-beta*v_k) * dv_k/dx)
            #             = -sum(exp(-beta*v_k)/S * dv_k/dx)
            #             = -sum(w_k * dv_k/dx)
            # where w_k = exp(-beta*v_k)/S
            #
            # For pair (i,j): v = |c_i - c_j|/2
            # dv/d(c_i) = (c_i - c_j) / (2|c_i - c_j|)
            # dv/d(c_j) = -(c_i - c_j) / (2|c_i - c_j|)
            #
            # So contribution to grad from this pair:
            # grad[i] += -w * (c_i - c_j) / (2*d)
            # grad[j] += -w * (-(c_i - c_j)) / (2*d) = +w * (c_i - c_j) / (2*d)
            
            idx += 1
    
    # Let me redo this more carefully
    grad = np.zeros_like(centers)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                idx += 1
                continue
            wi = w[idx]
            # dv/d(c_i) = diff/(2d), dv/d(c_j) = -diff/(2d)
            # grad contribution = -w * dv/dx
            grad[i] -= wi * diff / (2.0 * d)
            grad[j] += wi * diff / (2.0 * d)
            idx += 1
    
    # Wall gradients
    for i in range(n):
        for k in range(2):
            v = 0.5 - abs(centers[i, k])
            wi = w[idx]
            # dv/d(centers[i,k]) = -sign(centers[i,k])
            s = np.sign(centers[i, k])
            if s == 0:
                s = 1.0
            grad[i, k] -= wi * (-s)  # -w * dv/dx = -w * (-s) = w*s
            idx += 1
    
    return obj, grad


def _construct_hexagonal(n_circles):
    """Create a hexagonal close-packed initial configuration."""
    # Start with a hexagonal lattice, then trim to n_circles
    # Hexagonal lattice with spacing 1
    pts = []
    # Generate a large hexagonal grid and center it
    R = int(np.ceil(np.sqrt(n_circles) * 2))
    for row in range(-R, R + 1):
        for col in range(-R, R + 1):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    
    # Scale and center
    # We want the points to fit roughly in [-0.5, 0.5]^2
    # First, find the extent
    min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    pts = pts - center
    
    # Scale to fit in unit square
    extent = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
    if extent > 0:
        scale = 0.9 / extent  # Leave some margin
        pts *= scale
    
    # Sort by distance from center to pick the closest n_circles
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n_circles]]
    
    return pts


def _gradient_polish(centers, max_time=10.0):
    """Polish the configuration using gradient descent on the soft-min objective."""
    start = time.time()
    beta = 50.0
    centers = centers.copy()
    
    for _ in range(200):
        if time.time() - start > max_time:
            break
        obj, grad = _soft_min_obj_and_grad(centers, beta)
        
        # Simple gradient step with line search
        lr = 0.01
        for _ in range(50):
            new_centers = centers - lr * grad
            new_obj, _ = _soft_min_obj_and_grad(new_centers, beta)
            if new_obj < obj:
                centers = new_centers
                obj = new_obj
                break
            lr *= 0.5
        else:
            break
        
        # Increase beta gradually
        beta *= 1.05
    
    return centers


def construct():
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with hexagonal initial guess
    centers = _construct_hexagonal(N)
    
    # Compute initial radius
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Gradient polish with increasing beta
    for beta_start in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline:
            break
        centers = _gradient_polish(centers, max_time=min(15.0, deadline - time.time()))
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        centers = _gradient_polish(best_centers, max_time=min(20.0, deadline - time.time() - 2.0))
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
