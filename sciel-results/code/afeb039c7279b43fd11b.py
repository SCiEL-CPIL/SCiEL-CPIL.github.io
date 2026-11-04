import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Generate a hex lattice with spacing small enough to have > N points in unit disk
    # Target density: ~300 points in area pi. Area per point ~ pi/300 ~ 0.01.
    # Spacing s ~ sqrt(0.01) ~ 0.1.
    # Hex lattice spacing a: area per point = a^2 * sqrt(3)/2.
    # a = sqrt(2 * Area / sqrt(3)) = sqrt(2 * 0.01 / 1.732) ~ sqrt(0.0115) ~ 0.107
    spacing = 0.11
    points_list = []
    # Iterate over a grid large enough to cover the unit circle
    # Radius 1. Need to go from -1.5 to 1.5
    x_range = np.arange(-1.5, 1.5, spacing)
    y_range = np.arange(-1.5, 1.5, spacing * np.sqrt(3) / 2)
    
    for i, x in enumerate(x_range):
        # Offset every other row
        offset = (spacing / 2.0) if i % 2 else 0.0
        for y in y_range:
            px = x + offset
            py = y
            if np.hypot(px, py) < 0.95: # Keep well inside
                points_list.append([px, py])
    
    points = np.array(points_list)
    
    if len(points) < N:
        # Fallback: random if lattice failed (unlikely)
        points = np.random.rand(N, 2) * 2 - 1
        # Ensure inside
        norms = np.linalg.norm(points, axis=1)
        points = points * (0.9 / norms)[:, None]
    else:
        # Select N points. 
        # To get a good start, we want points that are spread out.
        # Taking the N closest to the origin creates a dense core which is good for packing.
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:N]]
        
    # Add small random jitter to break symmetry and help optimization
    points += np.random.normal(0, 0.01, points.shape)
    
    # Ensure initial points are inside
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 0.9
    points[mask] = points[mask] * (0.9 / norms[mask])[:, None]
    
    best_points = points.copy()
    best_radius = -np.inf
    
    # Objective function: Minimize negative soft-min of distances
    # Soft min of values v: -1/beta * log(sum(exp(-beta * v)))
    # We want to maximize min(v), so we minimize -softmin(v)
    # v consists of pairwise distances and distance to boundary (1 - norm)
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 2)
        
        # Pairwise distances
        # cdist is efficient
        dists = cdist(x, x)
        # Get upper triangle indices
        iu = np.triu_indices(N, k=1)
        d_pairs = dists[iu]
        
        # Boundary distances
        norms = np.linalg.norm(x, axis=1)
        d_boundary = 1.0 - norms
        
        # Combine all constraints into one array
        # We want to maximize the minimum of these.
        # To avoid log(0) or issues, clip distances to a small positive value
        eps = 1e-6
        d_pairs = np.maximum(d_pairs, eps)
        d_boundary = np.maximum(d_boundary, eps)
        
        all_dists = np.concatenate([d_pairs, d_boundary])
        
        # Soft min
        # beta is controlled from outside
        nonlocal beta
        # Compute log-sum-exp for stability
        # log(sum(exp(-beta * v))) = -beta * max(v) + log(sum(exp(-beta * (v - max(v)))))
        # Actually, softmin is -1/beta * log(sum(exp(-beta * v)))
        # Let u = -beta * v
        # logsumexp(u)
        # We can use scipy.special.logsumexp if available, or manual
        
        # Manual logsumexp for -beta * v
        u = -beta * all_dists
        max_u = np.max(u)
        log_sum_exp = max_u + np.log(np.sum(np.exp(u - max_u)))
        
        val = -log_sum_exp / beta
        
        # Gradient
        # d/dx (-1/beta * log(sum(exp(-beta * v_i))))
        # = -1/beta * 1/sum(exp(-beta * v_i)) * sum(exp(-beta * v_i) * (-beta * dv_i/dx))
        # = sum(exp(-beta * v_i) * dv_i/dx) / sum(exp(-beta * v_i))
        # Let p_i = exp(-beta * v_i) / sum(exp(-beta * v_j))
        # Grad = sum(p_i * dv_i/dx)
        
        exp_u = np.exp(u - max_u) # Unnormalized probs
        probs = exp_u / np.sum(exp_u)
        
        grad = np.zeros_like(x)
        
        # Gradient of pairwise distances
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij
        # We need to accumulate: sum_{j != i} p_{ij} * (x_i - x_j) / d_ij
        # Note: p_{ij} corresponds to the term for pair (i,j) in the sum.
        # The sum in the objective is over all pairs and boundary terms.
        # The probabilities probs are aligned with all_dists.
        
        # Map probs back to pairs
        p_pairs = probs[:len(d_pairs)]
        p_boundary = probs[len(d_pairs):]
        
        # Vectorized gradient accumulation for pairs
        # We have indices iu (i, j)
        i_idx, j_idx = iu
        
        # For each pair (i, j), the force on i is p_{ij} * (x_i - x_j) / d_ij
        # The force on j is p_{ij} * (x_j - x_i) / d_ij = - force on i
        
        # Create a matrix of forces? N=300, 300x300 is fine.
        # Or accumulate directly.
        
        # d_pairs is the distance for each pair in iu
        # p_pairs is the weight
        
        # Unit vectors
        # diff = x[i] - x[j]
        # We can use advanced indexing
        
        # Accumulate into grad
        # grad[i] += p * (x[i] - x[j]) / d
        # grad[j] -= p * (x[i] - x[j]) / d
        
        # To do this efficiently:
        # Create a matrix of weights W where W[i,j] = p_{ij} / d_{ij}
        # Then grad = (W * (x[:,None,:] - x[None,:,:])) summed? No, that's O(N^3).
        
        # Better:
        # For each pair, compute vector v = (x[i] - x[j]) / d
        # Multiply by p
        # Add to grad[i], subtract from grad[j]
        
        # Vectorized:
        # x_i = x[i_idx], x_j = x[j_idx]
        x_i = x[i_idx]
        x_j = x[j_idx]
        vecs = x_i - x_j
        # Normalize
        vecs /= d_pairs[:, None]
        # Weight
        vecs *= p_pairs[:, None]
        
        # Accumulate
        # np.add.at
        np.add.at(grad, i_idx, vecs)
        np.add.at(grad, j_idx, -vecs)
        
        # Boundary gradient
        # d(1 - ||x_i||) / dx_i = - x_i / ||x_i||
        # Force is p_boundary * (- x_i / ||x_i||)
        # Wait, objective is max min(d). 
        # If we are minimizing -softmin, the gradient points in the direction of increasing the soft-min.
        # d(softmin)/dx = sum p_i dv_i/dx.
        # For boundary term v_i = 1 - ||x_i||.
        # dv_i/dx_i = - x_i / ||x_i||.
        # So we add p_boundary * (- x_i / ||x_i||) to grad.
        
        # Handle zero norms
        safe_norms = np.where(norms < eps, 1.0, norms)
        boundary_vecs = -x / safe_norms[:, None]
        boundary_vecs *= p_boundary[:, None]
        
        grad += boundary_vecs
        
        # We are minimizing, so return -val and -grad?
        # No, scipy minimize minimizes f.
        # We defined f = -softmin.
        # So we return val (which is -softmin) and its gradient.
        # Wait.
        # Objective in code:
        # val = -log_sum_exp / beta. This is the soft-min value.
        # We want to MAXIMIZE soft-min.
        # So we MINIMIZE -soft-min.
        # So f(x) = -val.
        # grad_f = -grad_softmin.
        
        return -val, -grad

    beta = 5.0
    # Annealing schedule
    # Start smooth, get sharper
    
    while time.time() - start_time < time_limit - 5:
        x_flat = points.flatten()
        
        try:
            res = minimize(
                objective_and_grad, 
                x_flat, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 50, 'gtol': 1e-5, 'ftol': 1e-5}
            )
        except Exception:
            break
            
        new_points = res.x.reshape(N, 2)
        
        # Evaluate true radius
        dists = cdist(new_points, new_points)
        iu = np.triu_indices(N, k=1)
        min_pair_dist = np.min(dists[iu])
        norms = np.linalg.norm(new_points, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        current_radius = min(min_pair_dist, min_boundary_dist) / 2.0
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points.copy()
            
        # Increase beta to sharpen the minimum
        beta *= 1.2
        
        # If we are stuck or improving very slowly, we might want to restart or just continue
        # Given time budget, increasing beta is the main driver.
        
    return best_points

# EVOLVE-BLOCK-END
