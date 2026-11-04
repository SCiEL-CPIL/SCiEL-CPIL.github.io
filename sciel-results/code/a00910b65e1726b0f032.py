import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _soft_min(x, beta=50.0):
    """
    Smooth approximation of the minimum of x.
    x: array of values
    beta: temperature parameter (higher is sharper)
    """
    x_min = np.min(x)
    # Shift to avoid overflow/underflow
    x_shifted = x - x_min
    # If x_shifted is large, exp(-beta * x_shifted) underflows to 0, which is fine.
    # If x_shifted is 0, exp(0) = 1.
    # Sum of exps is at least 1.
    log_sum = np.log(np.sum(np.exp(-beta * x_shifted)))
    return x_min - (1.0 / beta) * log_sum

def _compute_distances(pos, w, h):
    """
    Compute all pairwise distances and distances to walls.
    Returns:
    d_pairs: (N, N) matrix of distances between circle centers
    d_walls: (N, 4) matrix of distances to each wall
    """
    N = len(pos)
    # Pairwise distances
    # pos is (N, 2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d_pairs = np.sqrt(np.sum(diff**2, axis=2))
    
    # Distances to walls
    d_walls = np.column_stack([
        pos[:, 0],          # Left
        w - pos[:, 0],      # Right
        pos[:, 1],          # Bottom
        h - pos[:, 1]       # Top
    ])
    return d_pairs, d_walls

def _objective_and_grad(pos, w, h, beta=50.0):
    """
    Objective: -Sum(soft_min(clearances))
    Gradient: -Sum(d(soft_min)/dx)
    """
    N = len(pos)
    d_pairs, d_walls = _compute_distances(pos, w, h)
    
    # Construct the vector of clearances for each circle
    # For circle i, clearances are d_pairs[i, j]/2 for j!=i, and d_walls[i, :]
    # Total clearances per circle: (N-1) + 4 = N+3
    
    # We want to compute sum_i soft_min(clearances_i)
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # grad soft_min(x) w.r.t x_k = exp(-beta * x_k) / sum(exp(-beta * x))
    
    # To be efficient, we can compute the weights for each clearance.
    # Let C_i be the vector of clearances for circle i.
    # We need sum_i [ -1/beta * log(sum_k exp(-beta * C_{i,k})) ]
    # And gradient w.r.t pos.
    
    obj_val = 0.0
    grad = np.zeros_like(pos)
    
    # Precompute exponentials for stability
    # We iterate over circles to accumulate gradients, but vectorize the inner operations where possible.
    # Since N=21, looping over N is fine.
    
    for i in range(N):
        # Clearances for circle i
        # d_pairs[i, :]/2 (excluding i)
        d_pairs_i = d_pairs[i, :].copy()
        d_pairs_i[i] = np.inf # Exclude self
        
        # Clearances vector for circle i
        # It consists of d_pairs_i/2 (N-1 elements) and d_walls[i] (4 elements)
        c_pairs = d_pairs_i / 2.0
        c_walls = d_walls[i, :]
        c_all = np.concatenate([c_pairs, c_walls])
        
        # Soft min
        c_min = np.min(c_all)
        c_shifted = c_all - c_min
        exp_terms = np.exp(-beta * c_shifted)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0: # Should not happen
            continue
            
        log_sum = np.log(sum_exp)
        obj_val -= (-1.0 / beta) * log_sum # We are minimizing -sum(soft_min), so objective is -soft_min
        # Wait, objective in minimize is f(x). We want to maximize sum(soft_min).
        # So f(x) = - sum(soft_min).
        # soft_min = c_min - 1/beta * log_sum
        # f(x) = - sum (c_min - 1/beta * log_sum) = - sum(c_min) + 1/beta * sum(log_sum)
        
        # Gradient of soft_min w.r.t c_all
        # d(soft_min)/d(c_k) = exp(-beta * c_k) / sum_exp
        grad_c = exp_terms / sum_exp
        
        # Gradient w.r.t pos[i]
        # c_k comes from d_pairs(i,j)/2 or d_walls(i, wall)
        
        # 1. From pairs
        # c_j = d_ij / 2 = 0.5 * ||p_i - p_j||
        # d(c_j)/d(p_i) = 0.5 * (p_i - p_j) / ||p_i - p_j||
        # Contribution to grad[pos[i]] += sum_j grad_c[j] * d(c_j)/d(p_i)
        
        # grad_c is (N+3,). First N-1 are for pairs, last 4 for walls.
        # Note: d_pairs_i has N elements, index i is inf.
        # c_pairs has N elements, index i is inf.
        # We need to map grad_c indices to actual j indices.
        
        # Let's just loop over j for clarity and safety, N is small.
        g_i = np.zeros(2)
        
        # Pairs
        for j in range(N):
            if i == j:
                continue
            # Find index in c_pairs
            # c_pairs is d_pairs_i / 2. d_pairs_i has original indices.
            # So c_pairs[j] corresponds to circle j.
            # grad_c index for pair j is just j? 
            # c_all = [c_pairs[0], ..., c_pairs[N-1], c_walls[0], ...]
            # But we skipped i in c_pairs? No, we set d_pairs_i[i] = inf.
            # So c_pairs[i] is inf. exp(-beta*inf) = 0. So grad_c[i] will be 0.
            # So we can just use index j directly from 0 to N-1.
            
            w_j = grad_c[j]
            if w_j > 1e-12:
                diff_ij = pos[i] - pos[j]
                d_ij = np.linalg.norm(diff_ij)
                if d_ij > 1e-9:
                    g_i += w_j * 0.5 * diff_ij / d_ij
                else:
                    # Degenerate case, push apart randomly or use subgradient
                    # Use a small random vector to break symmetry
                    g_i += w_j * 0.5 * np.array([1.0, 0.0])
            
        # Walls
        # c_walls = [x_i, w-x_i, y_i, h-y_i]
        # Indices in c_all: N, N+1, N+2, N+3
        # grad_c[N] corresponds to x_i (left wall)
        # grad_c[N+1] corresponds to w-x_i (right wall)
        # grad_c[N+2] corresponds to y_i (bottom wall)
        # grad_c[N+3] corresponds to h-y_i (top wall)
        
        w_left = grad_c[N]
        w_right = grad_c[N+1]
        w_bottom = grad_c[N+2]
        w_top = grad_c[N+3]
        
        # d(x_i)/dx_i = 1, d(w-x_i)/dx_i = -1
        # d(y_i)/dy_i = 1, d(h-y_i)/dy_i = -1
        
        g_i[0] += w_left * 1.0
        g_i[0] += w_right * (-1.0)
        g_i[1] += w_bottom * 1.0
        g_i[1] += w_top * (-1.0)
        
        grad[i] = g_i
        
    return obj_val, grad

def _force_directed_layout(pos, w, h, iterations=500):
    """
    Simple force-directed layout to separate overlapping circles.
    Assumes initial radii are estimated from grid spacing.
    """
    N = len(pos)
    # Estimate initial radii
    # Use minimum distance to walls and neighbors
    radii = np.zeros(N)
    for i in range(N):
        min_d = min(pos[i][0], w - pos[i][0], pos[i][1], h - pos[i][1])
        for j in range(N):
            if i != j:
                d = np.linalg.norm(pos[i] - pos[j])
                min_d = min(min_d, d / 2.0)
        radii[i] = max(0.0, min_d * 0.9)
        
    for _ in range(iterations):
        forces = np.zeros_like(pos)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 1e-9:
                    forces[i] += np.array([1.0, 0.0])
                    continue
                if d < radii[i] + radii[j]:
                    # Repulsive force
                    f = (radii[i] + radii[j] - d) / d
                    forces[i] += f * (pos[i] - pos[j])
            # Wall repulsion
            for wall in range(4):
                if wall == 0: # Left
                    d = pos[i][0]
                    if d < radii[i]:
                        forces[i][0] += (radii[i] - d)
                elif wall == 1: # Right
                    d = w - pos[i][0]
                    if d < radii[i]:
                        forces[i][0] -= (radii[i] - d)
                elif wall == 2: # Bottom
                    d = pos[i][1]
                    if d < radii[i]:
                        forces[i][1] += (radii[i] - d)
                elif wall == 3: # Top
                    d = h - pos[i][1]
                    if d < radii[i]:
                        forces[i][1] -= (radii[i] - d)
        
        # Update positions with damping
        pos += 0.05 * forces
        # Clip to bounds
        pos[:, 0] = np.clip(pos[:, 0], 0, w)
        pos[:, 1] = np.clip(pos[:, 1], 0, h)
        
    return pos

def _pack_for_aspect(w, h, n_circles=N, deadline=None):
    """
    Pack circles in w x h rectangle.
    """
    # 1. Initialize with hexagonal grid
    # Estimate number of rows/cols
    # Area per circle ~ pi * r^2. Total area w*h.
    # Rough estimate r ~ sqrt(w*h / (pi * n))
    r_est = np.sqrt(w * h / (np.pi * n_circles))
    
    # Hexagonal packing density ~ 0.9069
    # Effective area per circle ~ pi * r^2 / 0.9069
    # Spacing s ~ sqrt(pi * r^2 / 0.9069) = r * sqrt(pi/0.9069) ~ 1.86 * r
    # Let's just use a grid and let optimization fix it.
    
    # Try a few grid configurations
    best_pos = None
    best_score = -np.inf
    
    for cols in range(3, int(np.sqrt(n_circles * w / h)) + 3):
        if cols > n_circles: break
        rows = int(np.ceil(n_circles / cols))
        
        # Create grid
        pos = np.zeros((n_circles, 2))
        for i in range(n_circles):
            col = i % cols
            row = i // cols
            
            # Center of cell
            x = (col + 0.5) * w / cols
            y = (row + 0.5) * h / rows
            
            # Add slight hexagonal offset for odd rows
            if row % 2 == 1 and cols > 1:
                x += 0.5 * w / cols
                
            pos[i] = [x, y]
        
        # Force directed relaxation
        pos = _force_directed_layout(pos, w, h, iterations=200)
        
        # Optimization
        x0 = pos.flatten()
        
        # Adaptive beta: start low, increase
        # We'll do a few steps with increasing beta
        
        current_pos = pos.copy()
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if deadline and time.time() > deadline:
                break
                
            def obj_wrapper(x):
                p = x.reshape(n_circles, 2)
                val, _ = _objective_and_grad(p, w, h, beta)
                return val
            
            def grad_wrapper(x):
                p = x.reshape(n_circles, 2)
                _, g = _objective_and_grad(p, w, h, beta)
                return g.flatten()
            
            bounds = [(0, w)] * n_circles + [(0, h)] * n_circles
            
            try:
                res = minimize(obj_wrapper, current_pos.flatten(), method='L-BFGS-B', 
                               jac=grad_wrapper, bounds=bounds, 
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                current_pos = res.x.reshape(n_circles, 2)
            except Exception:
                pass
            
        # Evaluate final score
        d_pairs, d_walls = _compute_distances(current_pos, w, h)
        radii = np.zeros(n_circles)
        for i in range(n_circles):
            min_d = min(d_walls[i])
            for j in range(n_circles):
                if i != j:
                    min_d = min(min_d, d_pairs[i, j] / 2.0)
            radii[i] = max(0.0, min_d)
        
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_pos = current_pos.copy()
            best_radii = radii.copy()
            
    if best_pos is None:
        return None, None, 0.0
        
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_score = 0
    best_result = None
    
    # Try various aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h=2
    # w = ar * h => h(ar+1)=2 => h = 2/(ar+1), w = 2*ar/(ar+1)
    
    # Sample aspect ratios
    aspect_ratios = np.linspace(0.4, 2.5, 15)
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, deadline=deadline)
        
        if pos is not None and score > best_score:
            best_score = score
            best_result = np.column_stack([pos, radii])
            
    # Fallback
    if best_result is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            radii[i] = 0.5 / side - 1e-3
        best_result = np.column_stack([pos, radii])
        
    return best_result
