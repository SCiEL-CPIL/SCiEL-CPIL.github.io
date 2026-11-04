"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

def _compute_radii_and_grad(centers):
    """
    Computes radii and gradient of sum(radii) with respect to centers.
    Uses Voronoi diagram for neighbor identification.
    Objective: sum(r_i)
    r_i = min( dist_to_boundary_i, min_{j in neighbors(i)} dist(i, j) / 2 )
    """
    n = len(centers)
    if n < 3:
        radii = np.zeros(n)
        grad = np.zeros((n, 2))
        for i in range(n):
            r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    r = min(r, d / 2.0)
            radii[i] = r
        return radii, grad

    try:
        vor = Voronoi(centers)
        # Map points to regions
        # scipy.spatial.Voronoi has .point_region
        point_region = vor.point_region
    except Exception:
        # Fallback to O(N^2) if Voronoi fails
        radii = np.zeros(n)
        grad = np.zeros((n, 2))
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        
        for i in range(n):
            # Boundary distance
            dist_b = np.array([centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1]])
            min_b = np.min(dist_b)
            
            # Neighbor distance
            min_n = np.min(dist_matrix[i])
            j = np.argmin(dist_matrix[i])
            
            r = min(min_b, min_n / 2.0)
            radii[i] = r
            
            # Gradient
            # If limited by boundary
            if min_b <= min_n / 2.0:
                # grad of dist_b
                if dist_b[0] == min_b: grad[i,0] += -1.0 # x - 0 -> grad x is 1? No, r = x. grad r/grad x = 1.
                # Wait, r = min(x, 1-x...). 
                # If r = x, d(r)/d(x) = 1.
                # If r = 1-x, d(r)/d(x) = -1.
                # If r = y, d(r)/d(y) = 1.
                # If r = 1-y, d(r)/d(y) = -1.
                
                if dist_b[0] == min_b: grad[i,0] += 1.0
                elif dist_b[1] == min_b: grad[i,0] += -1.0
                elif dist_b[2] == min_b: grad[i,1] += 1.0
                else: grad[i,1] += -1.0
            else:
                # Limited by neighbor j
                # r = dist(i,j)/2
                # d(r)/d(x_i) = (x_i - x_j) / (2 * dist)
                # d(r)/d(y_i) = (y_i - y_j) / (2 * dist)
                # Also, moving j changes r_i.
                # d(r_i)/d(x_j) = (x_j - x_i) / (2 * dist)
                # d(r_i)/d(y_j) = (y_j - y_i) / (2 * dist)
                
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                d = dist_matrix[i, j]
                
                if d > 1e-9:
                    grad[i,0] += dx / d
                    grad[i,1] += dy / d
                    grad[j,0] += -dx / d
                    grad[j,1] += -dy / d
        return radii, grad

    radii = np.zeros(n)
    grad = np.zeros((n, 2))
    
    # Precompute distances to boundaries
    dist_to_box = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                             np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # We need neighbor distances.
    # For each point, find the nearest neighbor.
    # Voronoi regions contain vertices. The distance from point to region boundary is r_voronoi.
    # r_i = min(dist_to_box_i, r_voronoi_i)
    
    # Calculate r_voronoi for each point
    # voronoi.vertices are the vertices of the cells.
    # For each point i, get its region indices.
    # For each vertex in the region, if it's valid (index != -1), compute distance to center i.
    # The minimum such distance is the distance to the Voronoi cell boundary.
    
    # Note: This is an approximation if the nearest neighbor's Voronoi edge is not a vertex 
    # of the cell (e.g., if the cell is unbounded in a way that the closest point is on an edge 
    # defined by an unbounded ray). However, for packing inside a box, boundaries are usually 
    # constrained or we can check edge midpoints.
    # Actually, a safer bet for gradient is to use the O(N^2) neighbor check for the active constraint 
    # but use Voronoi to identify *which* neighbors are close enough to potentially be active.
    # Or just use O(N^2) for N=26 as it is fast enough for L-BFGS iterations?
    # L-BFGS might do 100s of iterations. 100 * 26^2 * 2 (forward+grad) is ~ 100k ops. Very fast.
    # Let's stick to O(N^2) for robustness and simplicity of gradient, but optimize the inner loop.
    
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    
    for i in range(n):
        # 1. Check boundary constraint
        r_bound = dist_to_box[i]
        
        # 2. Check neighbor constraints
        # Find the closest neighbor
        min_dist = np.min(dist_matrix[i])
        j = np.argmin(dist_matrix[i])
        r_neigh = min_dist / 2.0
        
        # The active constraint is the minimum of the two
        r_i = min(r_bound, r_neigh)
        radii[i] = r_i
        
        # Gradient contribution
        if r_bound <= r_neigh:
            # Limited by box
            if centers[i, 0] <= 1.0 - centers[i, 0]:
                grad[i, 0] += 1.0
            else:
                grad[i, 0] += -1.0
            
            if centers[i, 1] <= 1.0 - centers[i, 1]:
                grad[i, 1] += 1.0
            else:
                grad[i, 1] += -1.0
        else:
            # Limited by neighbor j
            # r_i = 0.5 * ||c_i - c_j||
            # grad r_i wrt c_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
            # grad r_i wrt c_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
            
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = min_dist
            
            if d > 1e-9:
                inv_d = 1.0 / d
                grad[i, 0] += 0.5 * dx * inv_d
                grad[i, 1] += 0.5 * dy * inv_d
                grad[j, 0] += -0.5 * dx * inv_d
                grad[j, 1] += -0.5 * dy * inv_d
    
    return radii, grad

def _objective_and_grad(centers):
    """
    Objective: -sum(radii) (to minimize)
    """
    radii, grad = _compute_radii_and_grad(centers)
    return -np.sum(radii), -grad.flatten()

def _is_valid(centers, radii):
    """
    Check if the configuration is valid (disjoint, inside box).
    """
    # Check box
    for i in range(len(centers)):
        x, y = centers[i]
        r = radii[i]
        if x < r or x > 1.0 - r or y < r or y > 1.0 - r:
            return False
    # Check disjoint
    # Radii are computed as min(dist/2, box), so if we use these radii, they are disjoint by definition?
    # Yes, r_i <= d_ij / 2 for all j. So r_i + r_j <= d_ij?
    # Wait. r_i = d_ij / 2. r_j might be larger than d_ij / 2? No, r_j is also limited by its neighbors.
    # Is it possible that r_i + r_j > d_ij?
    # r_i = min( box_i, min_k d_ik/2 )
    # r_j = min( box_j, min_k d_jk/2 )
    # If i and j are neighbors, d_ij is the min for i? Not necessarily.
    # If d_ij is the min for i, r_i = d_ij/2.
    # If d_jk < d_ij for some k, then r_j = d_jk/2 < d_ij/2 = r_i.
    # Then r_i + r_j = d_ij/2 + d_jk/2.
    # Is d_ij/2 + d_jk/2 <= d_ij?
    # d_jk/2 <= d_ij/2 => d_jk <= d_ij.
    # So r_i + r_j <= d_ij.
    # Yes, the radii computed by this method always produce a valid packing.
    return True

def _grow_radii(centers, max_iter=100):
    """
    Polish step: Try to increase radii by moving circles away from constraints.
    Actually, the optimizer already maximizes sum of radii.
    This step might be redundant if the optimizer converged well.
    However, sometimes the optimizer gets stuck in a local min where a circle is "pinned" 
    by a non-active constraint that becomes active if we move it.
    A simple local search with random perturbations can help escape.
    """
    best_score, best_centers = -np.inf, centers.copy()
    step = 0.01
    
    for _ in range(max_iter):
        radii, _ = _compute_radii_and_grad(centers)
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
        
        # Random perturbation
        i = np.random.randint(len(centers))
        angle = np.random.uniform(0, 2*np.pi)
        new_c = centers.copy()
        new_c[i, 0] += step * np.cos(angle)
        new_c[i, 1] += step * np.sin(angle)
        
        # Keep in box
        new_c[i, 0] = np.clip(new_c[i, 0], 0.01, 0.99)
        new_c[i, 1] = np.clip(new_c[i, 1], 0.01, 0.99)
        
        new_radii, _ = _compute_radii_and_grad(new_c)
        new_score = np.sum(new_radii)
        
        if new_score > score:
            centers = new_c
        else:
            step *= 0.9
    
    return best_centers, best_score

def _hex_packing_init():
    centers = np.zeros((N, 2))
    side = int(np.ceil(np.sqrt(N)))
    for i in range(N):
        gx = i % side
        gy = i // side
        offset_x = 0.5 / side if gy % 2 == 1 else 0.0
        x = (gx + 0.5) / side + offset_x
        y = (gy + 0.5) / side
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        centers[i] = [x, y]
    return centers

def _random_init():
    return np.random.uniform(0.1, 0.9, (N, 2))

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_centers = None
    best_score = -np.inf
    
    np.random.seed(42)
    
    # Multi-start with different initializations
    inits = [_hex_packing_init()]
    for _ in range(10):
        inits.append(_random_init())
    
    for trial, init_centers in enumerate(inits):
        if time.time() > deadline:
            break
            
        centers = init_centers.copy()
        
        # L-BFGS-B optimization
        # Flatten centers to 1D array
        x0 = centers.flatten()
        
        try:
            res = minimize(
                _objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_centers = res.x.reshape(-1, 2)
            
            # Ensure centers are within [0,1]
            opt_centers = np.clip(opt_centers, 0.0, 1.0)
            
            radii, _ = _compute_radii_and_grad(opt_centers)
            score = np.sum(radii)
            
            if score > best_score:
                best_score = score
                best_centers = opt_centers.copy()
                
        except Exception:
            continue
    
    if best_centers is None:
        best_centers = _hex_packing_init()
        radii, _ = _compute_radii_and_grad(best_centers)
        best_score = np.sum(radii)
        
    # Polish with local search if time permits
    if time.time() < deadline - 10:
        polished_centers, polished_score = _grow_radii(best_centers)
        if polished_score > best_score:
            best_centers = polished_centers
            best_score = polished_score
            
    radii, _ = _compute_radii_and_grad(best_centers)
    
    # Final validation and clipping
    result = np.column_stack([best_centers, radii])
    for i in range(N):
        x, y, ri = result[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    return result
# EVOLVE-BLOCK-END
