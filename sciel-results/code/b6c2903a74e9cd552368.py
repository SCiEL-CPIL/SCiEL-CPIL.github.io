"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    min_dist_to_wall = np.min(0.5 - np.abs(centers))
    
    # Minimum pairwise distance / 2
    # Use cKDTree for efficiency
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            min_pair_dist = min(min_pair_dist, d)
    
    return min(min_dist_to_wall, min_pair_dist / 2.0)

def _initial_cubic_grid():
    """Create initial cubic grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _soft_min_gradient_descent(centers, t_init, max_iters=5000, beta=100.0, lr=0.005):
    """
    Optimize centers and t using gradient descent on a soft-min surrogate.
    Objective: maximize t subject to constraints.
    Surrogate: max_t sum(soft_min(constraints - t)) where soft_min uses log-sum-exp.
    Actually, we want to maximize the minimum constraint.
    Let constraints C_i be distances (pairwise/2 and wall).
    We want to maximize t such that t <= C_i for all i.
    We can maximize a soft version: max_t -sum(log(sum(exp(-beta*(C_i - t))))) 
    Or simply minimize sum(exp(-beta*(C_i - t))) which pushes t up and C_i up.
    Let's minimize: E = sum( max(0, t - C_i)^2 ) + lambda * sum( (t - C_i) ) ? No, that's not smooth.
    
    Better: Minimize F(t, c) = sum_i exp(-beta * (C_i(c) - t))
    Gradient wrt t: -beta * sum exp(-beta(C_i - t))
    Gradient wrt c: beta * sum exp(-beta(C_i - t)) * grad_c C_i
    
    We want to minimize F. 
    If t is large and C_i is small, C_i - t is negative, exp is large. 
    So minimizing F pushes t down or C_i up. 
    We want to maximize t. 
    Let's maximize G = -F = -sum exp(-beta(C_i - t)).
    Or just minimize F and let t float? 
    If we minimize F, t will go to -infinity if not constrained. 
    We need to maximize t. 
    So we minimize H = sum exp(-beta(C_i - t)) - alpha * t.
    For large alpha, t will be pushed up as much as possible until C_i - t becomes the bottleneck.
    
    Let's use: Minimize L = sum_i exp(-beta * (C_i - t)) - alpha * t
    alpha should be large enough to push t up.
    """
    centers = centers.copy()
    t = t_init
    best_centers = centers.copy()
    best_t = t
    best_score = _compute_radius(best_centers)
    
    alpha = 50.0 # Push t up
    
    for it in range(max_iters):
        if time.time() > 140:
            break
        
        # Compute distances
        # Wall distances
        wall_dists = 0.5 - np.abs(centers) # (N, 3)
        wall_dists_flat = wall_dists.flatten() # (3N,)
        
        # Pairwise distances / 2
        # We need gradients.
        # For efficiency, we can compute all pairs if N is small (50 is small).
        # Pairs: i < j
        # C_pair_ij = ||c_i - c_j|| / 2
        
        # To save time, we can compute forces directly.
        # grad_c C_pair_ij = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # grad_c C_wall_ik = -sign(c_ik)
        
        # Let's accumulate gradients.
        grad_centers = np.zeros_like(centers)
        grad_t = 0.0
        
        # Wall terms
        # C_i = 0.5 - |c_i|
        # term = exp(-beta * (C_i - t))
        # dL/dt = -alpha - sum(-beta * term) = -alpha + beta * sum(term)
        # dL/dc = sum(term * beta * dC/dc)
        
        wall_terms = np.exp(-beta * (wall_dists_flat - t))
        sum_wall_terms = np.sum(wall_terms)
        
        grad_t += -alpha + beta * sum_wall_terms
        
        # Grad wrt centers for walls
        # dC/dc = -sign(c)
        # dL/dc = sum(term * beta * (-sign(c)))
        # Reshape wall_terms to (N, 3)
        wall_terms_3d = wall_terms.reshape(N, 3)
        grad_centers += -beta * wall_terms_3d * np.sign(centers)
        
        # Pairwise terms
        # This is O(N^2). 50^2 = 2500. Fast enough.
        sum_pair_terms = 0.0
        # We can vectorize the pair computation to some extent or just loop.
        # Loop is fine for N=50.
        
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                C_pair = dist / 2.0
                term = np.exp(-beta * (C_pair - t))
                
                sum_pair_terms += term
                
                # Grad wrt t
                grad_t += -beta * term
                
                # Grad wrt centers
                # dC_pair/dc_i = 0.5 * diff / dist
                # dC_pair/dc_j = -0.5 * diff / dist
                # dL/dc_i += term * beta * (0.5 * diff / dist)
                # dL/dc_j += term * beta * (-0.5 * diff / dist)
                
                factor = 0.5 * beta * term / dist
                grad_centers[i] += factor * diff
                grad_centers[j] -= factor * diff
                
        # Update
        # Centers: c -= lr * grad_centers
        # t: t -= lr_t * grad_t
        
        # Adaptive learning rate?
        # Let's use a fixed small lr for stability.
        centers -= lr * grad_centers
        t -= lr * 0.1 * grad_t # Scale t update to be smaller to avoid instability
        
        # Clamp centers to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        # Compute actual radius for tracking
        if it % 10 == 0:
            current_score = _compute_radius(centers)
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
                best_t = t
        
    return best_centers, best_score

def _local_polish(centers, max_iters=2000):
    """
    Final high-precision polish using coordinate descent on the exact min objective.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    step = 0.01
    for it in range(max_iters):
        if time.time() > 145:
            break
            
        improved = False
        # Try moving each center in each dimension
        for i in range(N):
            for d in range(3):
                # Try small steps
                for delta in [step, -step, step*0.5, -step*0.5, step*0.1, -step*0.1]:
                    new_c = centers.copy()
                    new_c[i, d] += delta
                    if new_c[i, d] < -0.5 or new_c[i, d] > 0.5:
                        continue
                    new_r = _compute_radius(new_c)
                    if new_r > best_radius + 1e-12:
                        centers = new_c
                        best_radius = new_r
                        best_centers = centers.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    return best_centers, best_radius

def construct():
    """Construct 50 sphere centers in unit cube to maximize minimum radius."""
    start_time = time.time()
    deadline = 145.0
    
    # Start with cubic grid
    best_centers = _initial_cubic_grid()
    best_radius = _compute_radius(best_centers)
    
    # 1. Soft-min gradient descent
    # Initialize t slightly below current radius to be safe
    t_init = best_radius * 0.9
    
    # Run gradient descent
    gd_centers, gd_radius = _soft_min_gradient_descent(best_centers, t_init, max_iters=3000, beta=50.0, lr=0.01)
    if gd_radius > best_radius:
        best_centers = gd_centers
        best_radius = gd_radius
        
    # 2. Local polish
    if time.time() < deadline - 5:
        polished_centers, polished_radius = _local_polish(best_centers, max_iters=1000)
        if polished_radius > best_radius:
            best_centers = polished_centers
            best_radius = polished_radius
            
    # 3. If time remains, try a second round with different beta or restart
    if time.time() < deadline - 10:
        # Perturb slightly and run GD again
        perturbed = best_centers + np.random.normal(0, 0.005, (N, 3))
        perturbed = np.clip(perturbed, -0.5, 0.5)
        gd_centers2, gd_radius2 = _soft_min_gradient_descent(perturbed, best_radius * 0.95, max_iters=2000, beta=100.0, lr=0.005)
        if gd_radius2 > best_radius:
            best_centers = gd_centers2
            best_radius = gd_radius2
            if time.time() < deadline - 5:
                polished_centers2, polished_radius2 = _local_polish(best_centers, max_iters=500)
                if polished_radius2 > best_radius:
                    best_centers = polished_centers2
                    best_radius = polished_radius2

    # Final verification
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        return _initial_cubic_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
