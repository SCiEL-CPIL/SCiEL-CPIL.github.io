"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    x = centers[:, 0]
    y = centers[:, 1]
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    dists = np.hypot(dx, dy)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate a hexagonal lattice packing inside the unit circle."""
    best_points = None
    best_radius = 0.0
    
    for spacing in np.linspace(0.15, 0.25, 20):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            points = points[np.argsort(radius, kind='stable')[:N]]
            r = _compute_radius(points)
            if r > best_radius:
                best_radius = r
                best_points = points.copy()
    
    if best_points is not None:
        return best_points
    
    spacing = 0.2
    k = int(1.0 / spacing) + 2
    rows = [(i * spacing, j * spacing) for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    radius = radius[mask]
    if len(points) >= N:
        return points[np.argsort(radius, kind='stable')[:N]]
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

def _softmin_obj_and_grad(flat_centers, beta=50.0):
    """
    Objective: minimize -softmin of (pairwise dists/2, boundary dists)
    Gradient computed analytically.
    
    flat_centers: shape (2*N,)
    """
    centers = flat_centers.reshape(N, 2)
    
    # Boundary distances
    r_norm = np.hypot(centers[:, 0], centers[:, 1])
    boundary_dists = 1.0 - r_norm
    
    # Pairwise distances / 2
    x = centers[:, 0]
    y = centers[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.hypot(dx, dy)
    np.fill_diagonal(dists, 0.0)
    pair_dists = dists / 2.0
    
    # Collect all constraints (boundary + pairwise upper triangle)
    # Boundary: N values
    # Pairwise: N*(N-1)/2 values
    all_dists = np.concatenate([
        boundary_dists,
        pair_dists[np.triu_indices(N, k=1)]
    ])
    
    # Soft-min: -log(sum(exp(-beta * d))) / beta
    # We want to maximize the min, so minimize -softmin
    # softmin(d) = -log(sum(exp(-beta*d)))/beta
    # -softmin(d) = log(sum(exp(-beta*d)))/beta
    
    # For numerical stability, subtract max
    shifted = -beta * all_dists
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    obj = (log_sum - np.max(shifted)) / beta  # This is -softmin + max/beta, but we just need the gradient direction
    
    # Actually, let's compute properly:
    # softmin(d) = -1/beta * log(sum(exp(-beta*d)))
    # We minimize f = -softmin(d) = 1/beta * log(sum(exp(-beta*d)))
    
    shifted2 = -beta * all_dists
    shifted2 -= np.max(shifted2)
    exp_shifted = np.exp(shifted2)
    sum_exp = np.sum(exp_shifted)
    log_sum_exp = np.log(sum_exp)
    
    # f = 1/beta * (log_sum_exp + max_shifted) where max_shifted = max(-beta*d) = -beta*min(d)
    # But for gradient, we work with the softmax weights
    weights = exp_shifted / sum_exp  # shape (N + N*(N-1)/2,)
    
    # Gradient with respect to all_dists
    # df/d(all_dists) = -weights
    grad_all_dists = -weights
    
    # Now map back to centers
    grad_centers = np.zeros((N, 2))
    
    # Boundary part: grad_all_dists[:N]
    # boundary_dists[i] = 1 - hypot(centers[i])
    # d(boundary_dists[i])/d(centers[i]) = -centers[i] / hypot(centers[i])
    for i in range(N):
        if r_norm[i] > 1e-12:
            grad_centers[i] += grad_all_dists[i] * (-centers[i] / r_norm[i])
    
    # Pairwise part: grad_all_dists[N:]
    # pair_dists[i,j] = hypot(centers[i]-centers[j])/2
    # d(pair_dists[i,j])/d(centers[i]) = (centers[i]-centers[j]) / (2*hypot(...))
    # d(pair_dists[i,j])/d(centers[j]) = (centers[j]-centers[i]) / (2*hypot(...))
    
    idx_i, idx_j = np.triu_indices(N, k=1)
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        d = dists[i, j]
        if d < 1e-12:
            continue
        w = grad_all_dists[N + k]
        # d(pair_dists[i,j])/d(centers[i]) = (centers[i]-centers[j]) / (2*d)
        diff = centers[i] - centers[j]
        grad_centers[i] += w * diff / (2.0 * d)
        # d(pair_dists[i,j])/d(centers[j]) = (centers[j]-centers[i]) / (2*d)
        grad_centers[j] += w * (-diff) / (2.0 * d)
    
    # Flatten gradient
    grad_flat = grad_centers.flatten()
    
    return obj, grad_flat

def _gradient_polish(centers, max_time=10.0):
    """Use L-BFGS-B to polish the packing."""
    start = time.time()
    
    def objective(flat):
        if time.time() - start > max_time:
            return 0.0, np.zeros(2*N)
        obj, grad = _softmin_obj_and_grad(flat, beta=50.0)
        return obj, grad
    
    flat_init = centers.flatten()
    
    result = minimize(
        objective,
        flat_init,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
    )
    
    if result.success or result.fun < 0:
        polished = result.x.reshape(N, 2)
        if np.all(np.isfinite(polished)):
            return polished
    
    return centers

def construct():
    """Construct the packing of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Start with hexagonal packing
    centers = _hexagonal_packing()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Gradient polish
    if time.time() < deadline - 20:
        polished = _gradient_polish(best_centers, max_time=min(30.0, deadline - time.time() - 10))
        polished_radius = _compute_radius(polished)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished.copy()
    
    # If time remains, try gradient polish from slightly perturbed versions
    if time.time() < deadline - 15:
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            perturbed = best_centers + np.random.normal(0, 0.005, (N, 2))
            polished = _gradient_polish(perturbed, max_time=min(20.0, deadline - time.time() - 5))
            polished_radius = _compute_radius(polished)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centers = polished.copy()
    
    # Final validation
    final_radius = _compute_radius(best_centers)
    if final_radius > 0:
        return best_centers
    
    return _hexagonal_packing()
# EVOLVE-BLOCK-END
