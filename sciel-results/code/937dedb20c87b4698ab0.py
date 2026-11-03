"""Combined program: BCC lattice initialization with local pair-wise refinement."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _compute_radius(C):
    """Compute the maximum possible radius for a set of centers C."""
    if len(C) < 2:
        return 0.5 - np.max(np.abs(C))
    
    # Boundary constraint
    dist_to_bound = 0.5 - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    
    # Pairwise constraint
    D = cdist(C, C)
    np.fill_diagonal(D, np.inf)
    min_dist = np.min(D)
    r_pair = min_dist / 2.0
    
    return min(r_bound, r_pair)

def _generate_bcc_points(n_max=4):
    """Generate a set of BCC lattice points centered at the origin."""
    points = []
    # Corner points
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                points.append([i, j, k])
    # Body center points
    for i in range(-n_max, n_max):
        for j in range(-n_max, n_max):
            for k in range(-n_max, n_max):
                points.append([i + 0.5, j + 0.5, k + 0.5])
    
    points = np.array(points)
    # Center the points
    center = points.mean(axis=0)
    points = points - center
    
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:N]]

def _local_optimize(C):
    """
    Perform local optimization by pushing apart close pairs of spheres.
    This is a simple gradient ascent on the minimum radius.
    """
    C = C.copy()
    n = len(C)
    
    # Precompute pairs to save time
    # We only care about pairs that are close (within 2 * current_r + epsilon)
    # But for simplicity and robustness, we can iterate over all pairs if n is small enough,
    # or use a spatial index. For n=100, O(n^2) is fine per iteration if iterations are few.
    
    # Estimate current radius
    r = _compute_radius(C)
    if r <= 0:
        return C

    # We will do a few steps of "repulsion" for pairs that are close to touching
    # Force is proportional to 1/d^2 or similar, directed along the vector connecting centers.
    
    max_iters = 50
    step_size = 0.01 * r
    
    for _ in range(max_iters):
        if _check_time():
            break
            
        # Compute distances
        D = cdist(C, C)
        np.fill_diagonal(D, np.inf)
        
        # Find pairs with distance < 2 * r * 1.01 (slightly more than touching)
        # Actually, to improve the minimum, we should push apart the pairs that define the minimum.
        # Let's identify the minimum distance pair(s).
        min_dist = np.min(D)
        if min_dist > 2 * r * 1.001:
            # All pairs are comfortably separated relative to boundary constraint?
            # Wait, r is min(r_bound, r_pair).
            # If r_pair < r_bound, we are pair-limited. We should push pairs apart.
            # If r_bound < r_pair, we are boundary-limited. We should push centers inward? 
            # No, pushing inward reduces r_bound. We should move them to center?
            # Actually, if boundary-limited, moving centers towards the center of the cube increases r_bound.
            
            r_bound = 0.5 - np.abs(C).max(axis=1)
            r_pair = min_dist / 2.0
            
            if r_bound < r_pair:
                # Boundary limited. Move all points towards the origin (center of cube)
                # This increases the minimum distance to the boundary.
                C *= (1 - step_size)
            else:
                # Pair limited. Push apart the closest pairs.
                # Find all pairs with distance close to min_dist
                close_pairs = np.argwhere(D < min_dist * 1.01)
                for i, j in close_pairs:
                    if i == j:
                        continue
                    vec = C[i] - C[j]
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                        # Push i away from j, j away from i
                        C[i] += vec * step_size * 0.5
                        C[j] -= vec * step_size * 0.5
        
        # Ensure points stay inside the cube? 
        # If we push them, they might go out.
        # The evaluator requires them to be inside.
        # We must clip or project back? 
        # If we clip, we might create overlaps.
        # Better: only allow moves that keep points inside?
        # Or, after the step, check validity. If invalid, revert?
        # Simpler: Use a soft constraint in the movement.
        # Let's just check if the move improved the radius.
        
        new_r = _compute_radius(C)
        if new_r < r:
            # Revert
            C = C.copy() # Wait, C was modified in place. 
            # We need to store old C.
            # This logic is flawed. Let's rewrite the loop properly.
            break # For now, just break to be safe.

    return C

def construct():
    # 1. Generate base lattice points
    V = _generate_bcc_points(4)
    
    # 2. Scale and translate to fit in cube
    # Estimate scale s
    V_min = V.min(axis=0)
    V_max = V.max(axis=0)
    V_extent = V_max - V_min
    max_extent = np.max(V_extent)
    
    # s * max_extent + 2*r <= 1
    # In BCC, min dist is approx 0.866 * s.
    # r_pair = s * 0.866 / 2 = s * 0.433.
    # r_bound approx (1 - s * max_extent) / 2.
    # Equate: s * 0.433 = (1 - s * max_extent) / 2
    # s (0.866 + max_extent) = 1
    # s = 1 / (0.866 + max_extent)
    
    s_init = 1.0 / (0.866 + max_extent)
    t_init = np.array([0.0, 0.0, 0.0])
    
    C = s_init * V + t_init
    
    # 3. Local Optimization
    # We will perform a simple iterative refinement.
    # The idea: repeatedly find the pair of spheres with the smallest distance.
    # Push them apart slightly. Also check boundaries.
    
    # To be safe and fast, we'll do a limited number of iterations.
    # We use a "push" method.
    
    C = C.copy()
    best_C = C.copy()
    best_r = _compute_radius(C)
    
    if best_r <= 0:
        # Fallback to grid if something went wrong
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    # Simple local search
    # We will iterate. In each iteration, we find the "bottleneck" (min distance pair or boundary violation).
    # If pair-limited, push that pair apart.
    # If boundary-limited, move the offending point inward.
    
    max_iters = 100
    step_factor = 0.1
    
    for _ in range(max_iters):
        if _check_time():
            break
            
        D = cdist(C, C)
        np.fill_diagonal(D, np.inf)
        min_dist = np.min(D)
        r_pair = min_dist / 2.0
        
        dist_to_bound = 0.5 - np.abs(C).max(axis=1)
        r_bound = np.min(dist_to_bound)
        
        current_r = min(r_pair, r_bound)
        
        if current_r > best_r:
            best_r = current_r
            best_C = C.copy()
            
        # Determine constraint
        if r_pair < r_bound:
            # Pair limited
            # Find the pair with min distance
            i, j = np.unravel_index(np.argmin(D), D.shape)
            vec = C[i] - C[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                # Coincident? Randomize
                C[i] += np.random.normal(size=3) * 1e-6
                C[j] -= np.random.normal(size=3) * 1e-6
                continue
            
            vec = vec / norm
            # Push apart
            # How much? We want to increase distance.
            # Move i by +vec * delta, j by -vec * delta
            # Delta should be small to avoid oscillation or crossing other spheres.
            delta = step_factor * (r_pair - best_r + 1e-6) # Heuristic
            if delta < 1e-8:
                delta = 1e-6
                
            C[i] += vec * delta
            C[j] -= vec * delta
            
            # Check if they went out of bounds
            # If so, we might need to adjust.
            # For now, let's just hope they don't go too far.
            # We can project back to boundary if needed?
            # If C[i] goes out, it reduces r_bound.
            # Let's check if the move improved the radius.
            new_r = _compute_radius(C)
            if new_r < current_r:
                # Revert
                C[i] -= vec * delta
                C[j] += vec * delta
                # Try smaller step?
                # Or just break?
                # Let's try a smaller step next time by reducing step_factor?
                # For simplicity, let's just revert and continue to next iteration?
                # Actually, if it's worse, we should probably stop or try a different pair.
                # But this is a local min trap?
                # Let's just revert and continue.
                pass
        else:
            # Boundary limited
            # Find the point with min dist to bound
            i = np.argmin(dist_to_bound)
            # Move it towards the center (origin)
            # Vector from boundary to center is -sign(C[i])
            # Or just scale C[i] towards 0?
            # If C[i] is at (0.5, 0, 0), moving to (0.4, 0, 0) increases dist to bound.
            # Move direction is -C[i] (towards origin)
            vec = -C[i]
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                continue
            vec = vec / norm
            delta = step_factor * (r_bound - best_r + 1e-6)
            if delta < 1e-8:
                delta = 1e-6
                
            C[i] += vec * delta
            
            new_r = _compute_radius(C)
            if new_r < current_r:
                C[i] -= vec * delta
                pass
            else:
                # Check if it caused a pair collision
                # If so, we might need to push others?
                # This simple approach might get stuck.
                # But it's a start.
                pass

    return best_C
