"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Uses a hybrid approach: initial placement via grid, then local optimization using
    gradient-free methods to push circles apart and grow radii.
    """
    n = N
    # Initial placement: use a grid-like structure to avoid immediate overlaps
    # We'll use a simple heuristic initial placement
    side = int(np.ceil(np.sqrt(n)))
    # Create grid positions
    positions = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y])
                # Initial radius: small enough to not overlap with neighbors
                r = 0.5 / side * 0.9
                radii.append(r)
    
    # Ensure we have exactly n circles
    while len(positions) < n:
        positions.append([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
        radii.append(0.05)
    
    positions = np.array(positions[:n])
    radii = np.array(radii[:n])
    
    # Optimization: use a penalty-based approach with local search
    # We'll define a cost function that penalizes overlaps and boundary violations
    # and rewards larger radii
    
    def cost_and_grad(params):
        """
        params: array of shape (2*n,) where first n are x-coords, next n are y-coords
        We'll optimize positions and use a separate radius growth strategy.
        """
        x = params[:n]
        y = params[n:2*n]
        
        # Boundary penalty: circles must be inside [0,1]^2
        # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        # For now, we'll use a fixed radius estimate and optimize positions
        # Then we'll compute max possible radii given positions
        
        # Compute pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # For each pair, the constraint is r_i + r_j <= dist_ij
        # We want to maximize sum of radii subject to these constraints
        # and boundary constraints
        
        # Use a simple approach: assume equal radii first, then refine
        # For equal radii r: 2r <= min_dist, and r <= min(x, 1-x, y, 1-y)
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                if dist[i, j] < min_dist:
                    min_dist = dist[i, j]
        
        # Boundary constraint
        boundary_min = np.min(np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
        
        # Max equal radius
        r_equal = min(min_dist / 2, boundary_min)
        r_equal = max(r_equal, 1e-6)
        
        # Penalty for overlaps if we try to use r_equal
        # Actually, let's just return the positions and compute radii separately
        
        # For optimization, we want to maximize the minimum distance between circles
        # and keep them away from boundaries
        # Let's define a cost that we minimize:
        # cost = -sum of log(min_dist_pair) + boundary_penalty
        
        # Simpler: minimize sum of overlap penalties
        cost = 0.0
        for i in range(n):
            for j in range(i+1, n):
                # If circles overlap, penalize
                # We'll use a soft penalty
                d = dist[i, j]
                # Target: d >= 2*r, but r is unknown
                # Let's use a heuristic: penalize if d < 0.1 (too close)
                if d < 0.05:
                    cost += (0.05 - d)**2
                elif d < 0.1:
                    cost += (0.1 - d)**2 * 0.1
        
        # Boundary penalty
        for i in range(n):
            if x[i] < 0.05:
                cost += (0.05 - x[i])**2
            if x[i] > 0.95:
                cost += (x[i] - 0.95)**2
            if y[i] < 0.05:
                cost += (0.05 - y[i])**2
            if y[i] > 0.95:
                cost += (y[i] - 0.95)**2
        
        return cost
    
    # Use Nelder-Mead or Powell to optimize positions
    # We'll do a few rounds of optimization
    
    best_params = np.concatenate([positions[:, 0], positions[:, 1]])
    best_cost = cost_and_grad(best_params)
    
    # Try multiple restarts
    for restart in range(5):
        # Perturb initial positions
        if restart == 0:
            params = best_params.copy()
        else:
            params = best_params + np.random.randn(2*n) * 0.1
            params = np.clip(params, 0.01, 0.99)
        
        result = minimize(cost_and_grad, params, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x
    
    x = best_params[:n]
    y = best_params[n:2*n]
    
    # Now compute optimal radii given the positions
    # This is a linear programming problem: maximize sum(r_i) subject to:
    # r_i + r_j <= dist_ij for all i < j
    # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # r_i >= 0
    
    # Use a simple iterative approach: start with small radii and grow them
    # until constraints are violated
    
    radii = np.ones(n) * 0.01
    # Compute distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary constraints
    boundary_r = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    
    # Iteratively grow radii
    for iteration in range(100):
        updated = False
        for i in range(n):
            # Max radius for circle i given other circles
            max_r = boundary_r[i]
            for j in range(n):
                if i != j:
                    max_r = min(max_r, (dist[i, j] - radii[j]))
            max_r = max(max_r, 0.0)
            
            # Try to increase radius
            new_r = min(radii[i] * 1.1, max_r)
            if new_r > radii[i]:
                radii[i] = new_r
                updated = True
        
        if not updated:
            break
    
    # Final check: ensure no overlaps
    final_radii = np.zeros(n)
    for i in range(n):
        max_r = boundary_r[i]
        for j in range(n):
            if i != j:
                max_r = min(max_r, (dist[i, j] - final_radii[j]) / 2)
        final_radii[i] = max(max_r, 0.0)
    
    # Verify and adjust
    for i in range(n):
        for j in range(i+1, n):
            d = dist[i, j]
            if final_radii[i] + final_radii[j] > d:
                # Scale down
                scale = d / (final_radii[i] + final_radii[j]) * 0.999
                final_radii[i] *= scale
                final_radii[j] *= scale
    
    # Clip to boundary
    for i in range(n):
        final_radii[i] = min(final_radii[i], boundary_r[i])
    
    return np.column_stack([x, y, final_radii])
# EVOLVE-BLOCK-END
