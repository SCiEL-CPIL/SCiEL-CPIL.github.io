import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200

def _compute_radius(centers):
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    dist_matrix = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix)
    r = min(min_wall, min_pair / 2.0)
    return max(0.0, r)

def _soft_min(values, beta):
    """Smooth approximation of minimum using log-sum-exp trick."""
    m = np.min(values)
    return m - np.log(np.sum(np.exp(-beta * (values - m)))) / beta

def _objective_and_grad(x, beta=100.0):
    """
    Smooth approximation of negative minimum clearance.
    Uses soft-min of all pairwise and wall clearances.
    """
    centers = x.reshape(-1, 2)
    
    # Wall clearances: 4 per point (left, right, bottom, top)
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    # Pairwise clearances: d_ij / 2 for all i < j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    pair_clearances = dist_matrix / 2.0
    
    # Collect all clearances into one array
    # Wall: N*4 values, Pair: N*(N-1)/2 values
    wall_all = np.concatenate([d_left, d_right, d_bottom, d_top])
    # Upper triangle indices for pairs
    iu = np.triu_indices(N, k=1)
    pair_all = pair_clearances[iu]
    
    all_clearances = np.concatenate([wall_all, pair_all])
    
    # Soft min of all clearances
    r_smooth = _soft_min(all_clearances, beta)
    
    # Objective: minimize -r_smooth
    obj = -r_smooth
    
    # Gradient computation
    # d(r_smooth)/d(c_k) = -exp(-beta*c_k) / sum(exp(-beta*c_j))
    # This is a probability distribution over clearances
    exp_neg = np.exp(-beta * (all_clearances - np.min(all_clearances)))
    Z = np.sum(exp_neg)
    weights = exp_neg / Z  # weight for each clearance term
    
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    # For each point i, wall clearances are at indices [i], [N+i], [2N+i], [3N+i] in wall_all
    for i in range(N):
        w_left = weights[i]
        w_right = weights[N + i]
        w_bottom = weights[2*N + i]
        w_top = weights[3*N + i]
        
        # d(c_left)/dx_i = 1, d(c_right)/dx_i = -1, etc.
        grad[i*2] += beta * (w_left - w_right)  # d(-r_smooth)/dx = -d(r_smooth)/dx = -beta*sum(w_k * dc_k/dx)
        # Wait: r_smooth = min - log(sum(exp(-beta*(c-min))))/beta
        # d(r_smooth)/dc_k = -exp(-beta*c_k)/sum(exp(-beta*c_j)) = -w_k
        # d(-r_smooth)/dc_k = w_k
        # d(-r_smooth)/dx = sum_k w_k * dc_k/dx
        
        # Let me redo this carefully.
        # r_smooth = m - (1/beta)*log(sum(exp(-beta*(c_k - m))))
        # d(r_smooth)/dc_k = (1/beta) * exp(-beta*(c_k-m)) / sum(exp(-beta*(c_j-m))) = (1/beta) * w_k
        # where w_k = exp(-beta*(c_k-m))/sum(exp(-beta*(c_j-m)))
        # So d(-r_smooth)/dc_k = -w_k/beta
        
        # For wall:
        # c_left = x_i + 0.5, dc_left/dx_i = 1
        # c_right = 0.5 - x_i, dc_right/dx_i = -1
        # c_bottom = y_i + 0.5, dc_bottom/dy_i = 1
        # c_top = 0.5 - y_i, dc_top/dy_i = -1
        
        # d(-r_smooth)/dx_i = sum over wall terms: -w_k/beta * dc_k/dx_i
        # = -w_left/beta * 1 + -w_right/beta * (-1) + ... (y terms don't contribute to x)
        # = (-w_left + w_right)/beta
        
        # Hmm, let me just use finite difference for gradient to be safe,
        # or compute analytically more carefully.
    
    # Actually, let me use a simpler gradient-free approach with Nelder-Mead
    # and focus on getting a good initial configuration.
    
    # For now, return just the objective (Nelder-Mead doesn't need gradient)
    return obj

def _hex_initial(n_circles=N):
    """Generate hexagonal packing initial configuration."""
    # Start with hexagonal grid
    d = 1.0 / 15.0  # approximate spacing
    row_spacing = d * math.sqrt(3) / 2
    points = []
    
    y = -0.5 + d/2
    row = 0
    while len(points) < n_circles and y < 0.5 - d/2:
        x_start = -0.5 + d/2
        if row % 2 == 1:
            x_start += d/2
        x = x_start
        while x < 0.5 - d/2 and len(points) < n_circles:
            points.append([x, y])
            x += d
        y += row_spacing
        row += 1
    
    if len(points) < n_circles:
        # Pad with random points
        while len(points) < n_circles:
            points.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
    
    return np.array(points[:n_circles])

def _square_initial(n_circles=N):
    """Generate square grid initial configuration."""
    m = int(math.ceil(math.sqrt(n_circles)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) < n_circles:
        while len(grid) < n_circles:
            grid = np.vstack([grid, [np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)]])
    return grid[:n_circles]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    best_centers = None
    best_radius = 0.0
    
    # Generate multiple initial configurations
    inits = []
    
    # Hexagonal
    if time_left() > 60:
        inits.append(_hex_initial())
    
    # Square grid
    if time_left() > 60:
        inits.append(_square_initial())
    
    # Perturbed hexagonal
    if time_left() > 60:
        base = _hex_initial()
        for _ in range(3):
            if time_left() < 30:
                break
            perturbed = base + np.random.randn(N, 2) * 0.01
            perturbed = np.clip(perturbed, -0.49, 0.49)
            inits.append(perturbed)
    
    # Random
    if time_left() > 60:
        for _ in range(2):
            if time_left() < 30:
                break
            rand_pts = np.random.uniform(-0.48, 0.48, (N, 2))
            inits.append(rand_pts)
    
    # Optimize each initial configuration
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for init in inits:
        if time_left() < 15:
            break
            
        x0 = init.flatten()
        
        try:
            # Use Nelder-Mead for robustness
            res = minimize(
                lambda x: _objective_and_grad(x, beta=50.0),
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if res.x is not None and np.all(np.isfinite(res.x)):
                centers = res.x.reshape(-1, 2)
                centers = np.clip(centers, -0.5, 0.5)
                r = _compute_radius(centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = centers.copy()
                    
        except Exception:
            continue
        
        # Check if we should continue
        if time_left() < 20:
            break
    
    # Final polish if we have time
    if best_centers is not None and time_left() > 10:
        x0 = best_centers.flatten()
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, beta=200.0),
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                centers = res.x.reshape(-1, 2)
                centers = np.clip(centers, -0.5, 0.5)
                r = _compute_radius(centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = centers.copy()
        except Exception:
            pass
    
    # Fallback
    if best_centers is None:
        best_centers = _square_initial()
    
    return best_centers
