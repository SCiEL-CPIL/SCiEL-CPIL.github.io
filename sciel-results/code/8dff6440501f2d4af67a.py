# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        if L < 1e-9:
            n[i] = np.array([1.0, 0.0])
            continue
        n[i] = np.array([e[1], -e[0]]) / L
    return n

def _min_separation(A, B):
    """
    Compute the minimum separation between two convex polygons A and B.
    Separation > 0 means they are disjoint (gap exists).
    Separation <= 0 means they intersect or touch.
    """
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best = -np.inf
    for n in nA:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
    for n in nB:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
    return best

def _outer_side_for_points(points, phi_deg):
    """
    Calculate the minimum side length of a regular hexagon with rotation phi_deg
    that encloses the given points.
    """
    phi_rad = np.deg2rad(phi_deg)
    normals = np.column_stack((np.cos(phi_rad + np.deg2rad(60.0) * np.arange(6)), 
                               np.sin(phi_rad + np.deg2rad(60.0) * np.arange(6))))
    
    # Projections
    proj = points @ normals.T # (N, 6)
    widths = proj.max(axis=0) - proj.min(axis=0) # (6,)
    max_width = widths.max()
    return max_width / np.sqrt(3.0)

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 590.0
    
    # Generate candidate positions on an axial grid
    # We want to cover a region large enough for 11 hexes
    # Max radius from center for 11 hexes is roughly 2-3 units
    # Let's use radius 4 to be safe
    R = 4
    candidates = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            if abs(q + r) <= R:
                # Axial to Cartesian
                x = 1.5 * q
                y = np.sqrt(3) * (r + q / 2.0)
                candidates.append((x, y))
    
    candidates = np.array(candidates)
    N_cand = len(candidates)
    
    # Precompute pairwise distances between candidates
    # dist[i, j] = distance between candidate i and candidate j
    # If dist < 2, they overlap at scale=1.
    # If dist >= 2, they are separated at scale=1.
    # For scale s, they overlap if dist < 2*s.
    # Separation gap = dist - 2*s.
    
    # We will search for a subset of 11 candidates that forms a connected cluster
    # and then optimize the scale 's' to be as large as possible such that min_gap > 0.
    # Actually, we want to minimize the outer side length.
    # Outer side length S(s) is proportional to s * (width of cluster in unit coords).
    # Let W be the width of the cluster (in units of side length 1).
    # Then S = s * W / sqrt(3) (approx, depends on orientation).
    # We want to minimize s * W.
    # Constraint: for all pairs in cluster, dist(i, j) > 2*s + eps.
    # So s < (dist(i, j) - eps) / 2 for all pairs.
    # Max possible s for a given cluster = min_{i,j in cluster} (dist(i, j) - eps) / 2.
    # So we want to find a cluster of 11 points that minimizes:
    #   [min_{i,j} (dist(i, j) - eps) / 2] * [Width(cluster)]
    
    # This is a combinatorial optimization problem.
    # We can use Simulated Annealing on the set of indices.
    
    # Initial state: pick 11 random connected points? 
    # Or just pick the 11 closest to center?
    # Let's start with a hexagonal number pattern.
    # 1 center, 6 around, 4 from next ring?
    # Total 11.
    
    # Let's define a function to get the width of a set of points
    def _width(points, phi_deg):
        return _outer_side_for_points(points, phi_deg) * np.sqrt(3)
    
    # We will optimize the set of 11 indices and the outer angle.
    # State: list of 11 indices into candidates.
    
    # Initial state:
    # Center (0,0) is index of (0,0)
    # Find index of (0,0)
    idx_map = {(c[0], c[1]): i for i, c in enumerate(candidates)}
    
    # Construct a known good cluster:
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 points
    # Add (1,1), (-1,2), (0,2)? No, let's just take the 11 closest to origin in axial distance.
    # Axial distance: max(|q|, |r|, |q+r|)
    
    def axial_dist(q, r):
        return max(abs(q), abs(r), abs(q+r))
        
    # We need to map cartesian back to axial for distance?
    # Or just use cartesian distance from origin for initial sort.
    dists_to_origin = np.linalg.norm(candidates, axis=1)
    sorted_indices = np.argsort(dists_to_origin)
    
    # The 11 closest points to origin in cartesian are likely to be the hexagonal cluster.
    init_state = sorted_indices[:11].tolist()
    
    # Ensure they are distinct
    if len(set(init_state)) < 11:
         # Fallback: random
         init_state = random.sample(range(N_cand), 11)
         
    best_score = np.inf
    best_state = init_state[:]
    best_s = 0
    best_phi = 0
    best_points = None
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    steps = 100000
    
    current_state = init_state[:]
    current_score = np.inf
    
    # Precompute distances between all candidates for speed
    # D[i, j] = ||c_i - c_j||
    # N_cand is ~ (2*4+1)^2 ~ 81 points. 81x81 matrix is small.
    D = np.zeros((N_cand, N_cand))
    for i in range(N_cand):
        for j in range(N_cand):
            D[i, j] = np.linalg.norm(candidates[i] - candidates[j])
            
    def _evaluate(state, phi_deg):
        """
        state: list of 11 indices
        phi_deg: outer hexagon angle
        Returns: (score, s, points, valid)
        score = outer_side_length
        """
        pts = candidates[state]
        
        # Calculate max possible scale s
        # min_dist = min(D[i, j] for i, j in state, i != j)
        min_d = np.inf
        for i in range(11):
            for j in range(i+1, 11):
                d = D[state[i], state[j]]
                if d < min_d:
                    min_d = d
        
        # We need separation > 1e-6.
        # dist - 2*s > 1e-6 => 2*s < dist - 1e-6 => s < (dist - 1e-6)/2
        max_s = (min_d - 1e-6) / 2.0
        if max_s <= 0:
            return np.inf, 0.0, None, False
            
        # We can choose any s <= max_s.
        # The outer side length is proportional to s.
        # S(s) = s * Width(pts) / sqrt(3) ? 
        # Actually, _outer_side_for_points returns S for side=1.
        # If we scale points by s, S becomes s * S_unit.
        S_unit = _outer_side_for_points(pts, phi_deg)
        S = s * S_unit
        
        # To minimize S, we should use the largest possible s?
        # Wait. S = s * S_unit.
        # If we increase s, the physical size of the hexagons increases.
        # The outer hexagon must enclose them.
        # So yes, S is proportional to s.
        # But we are constrained by min_d.
        # So we MUST use s = max_s?
        # No, we can use ANY s <= max_s.
        # But S = s * S_unit.
        # S_unit depends on the shape of the cluster, not s.
        # So to minimize S, we should choose the SMALLEST s?
        # No, that would make the hexagons tiny.
        # The task says "unit-side regular hexagons".
        # So s MUST be 1.0?
        
        # RE-READ TASK:
        # "inner of shape (11, 3), rows (x, y, angle_deg) of unit-side regular hexagons"
        # "Valid only if: ... every pair of inner hexagons is separated ... by more than 1e-6"
        # "side, its side length (float)"
        
        # Ah! The inner hexagons are FIXED to side length 1.
        # So s is ALWAYS 1.0 for the inner hexagons.
        # The positions (x, y) are what we optimize.
        # The outer hexagon side length is the objective to minimize.
        
        # So my "scale s" idea is wrong.
        # The inner hexagons are rigid.
        # We are placing 11 rigid unit hexagons.
        # We need to find positions (x_i, y_i) such that they don't overlap.
        # And minimize the enclosing hexagon side.
        
        # So the search is over the POSITIONS.
        # My candidate grid approach:
        # If I pick positions from a grid, the distance between centers is fixed.
        # If dist < 2, they overlap.
        # So I can ONLY pick pairs with dist >= 2.
        # This is a constraint satisfaction problem.
        
        # Let's restart the logic.
        pass

    # Correct Logic:
    # 1. Generate candidate centers on a fine grid? 
    #    No, the optimal positions might not be on a lattice.
    #    However, for hexagonal packing, the optimal positions are often on a lattice or close to it.
    #    Let's use a hybrid: 
    #    Start with a lattice configuration.
    #    Use local search (gradient descent) to relax positions.
    #    Use simulated annealing to change topology (which lattice sites are occupied).
    
    # Let's stick to the lattice site selection first.
    # If we select 11 sites from the axial grid such that all pairwise distances >= 2 + eps,
    # then they form a valid packing (with some gaps).
    # We can then shrink the outer hexagon.
    # But we can also MOVE the centers to close the gaps and shrink the outer hexagon further.
    
    # Strategy:
    # 1. Find a valid set of 11 lattice sites (dist >= 2).
    # 2. Use local optimization to move these 11 points to minimize outer side length, 
    #    subject to dist(i,j) >= 2.
    
    # Let's implement the local optimization step properly.
    
    def _local_optimize(centers, max_iter=500):
        """
        centers: (11, 2) array
        Returns: optimized centers
        """
        # Use scipy minimize with constraints?
        # Or simple gradient descent with penalty.
        
        # Penalty for overlap:
        # For each pair i, j:
        # d = ||c_i - c_j||
        # if d < 2: penalty += (2 - d)^2
        
        # Objective: outer_side_length(centers) + lambda * penalty
        
        # We need to compute gradients.
        # Let's use finite differences for simplicity, or just L-BFGS-B on the penalized objective.
        
        def _obj(x):
            c = x.reshape(11, 2)
            # Outer side
            # Try a few angles for outer?
            # Or optimize outer angle too?
            # Let's fix outer angle to 0 for now in this local step, or optimize it.
            # Optimizing outer angle is 1 extra dim.
            
            # Let's just compute min side for optimal angle?
            # That's expensive.
            # Let's assume optimal angle is close to 0 or 30.
            # Actually, let's just use angle 0.
            s_outer = _outer_side_for_points(c, 0.0)
            
            # Overlap penalty
            pen = 0.0
            for i in range(11):
                for j in range(i+1, 11):
                    d = np.linalg.norm(c[i] - c[j])
                    if d < 2.0:
                        pen += (2.0 - d)**2
            
            return s_outer + 10.0 * pen
            
        # Use L-BFGS-B
        from scipy.optimize import minimize
        res = minimize(_obj, centers.flatten(), method='L-BFGS-B', options={'maxiter': max_iter})
        return res.x.reshape(11, 2)

    # Now, the SA search over lattice sites.
    
    # Get initial valid state
    # We need 11 sites with pairwise dist >= 2.
    # Let's try to construct one manually or randomly.
    
    def _get_random_valid_state():
        # Randomly pick 11 sites
        # Check if valid
        # If not, retry
        # This might be slow if valid states are rare.
        # But for R=4, there are many valid states.
        
        for _ in range(1000):
            state = random.sample(range(N_cand), 11)
            valid = True
            for i in range(11):
                for j in range(i+1, 11):
                    if D[state[i], state[j]] < 2.0 - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return state
        return None

    # Get initial state
    init_state = _get_random_valid_state()
    if init_state is None:
        # Fallback to closest 11, even if invalid? 
        # No, must be valid.
        # Increase R?
        # For now, return a known valid solution if we can't find one.
        pass

    current_state = init_state
    current_centers = candidates[current_state].copy()
    
    # Optimize initial state
    current_centers = _local_optimize(current_centers)
    
    # Calculate current score
    def _final_score(centers):
        # Find optimal outer angle
        best_s = np.inf
        best_phi = 0
        for phi in np.linspace(0, 60, 12):
            s = _outer_side_for_points(centers, phi)
            if s < best_s:
                best_s = s
                best_phi = phi
        return best_s, best_phi

    s_cur, phi_cur = _final_score(current_centers)
    best_score = s_cur
    best_state = current_state[:]
    best_centers = current_centers.copy()
    best_phi = phi_cur
    
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    
    step = 0
    while time.time() < deadline - 5.0 and T > T_min:
        step += 1
        
        # Move: pick a random index in state, replace with a random other candidate
        idx = random.randint(0, 10)
        old_site = current_state[idx]
        new_site = random.randint(0, N_cand - 1)
        
        if new_site == old_site:
            continue
            
        # Check if new_site is already in state
        if new_site in current_state:
            continue
            
        new_state = current_state[:]
        new_state[idx] = new_site
        
        # Check validity of new_state (lattice distances)
        valid = True
        for i in range(11):
            for j in range(i+1, 11):
                if D[new_state[i], new_state[j]] < 2.0 - 1e-6:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            continue
            
        # If valid, optimize the new configuration
        new_centers = candidates[new_state].copy()
        # Quick local optimization (fewer iterations for speed)
        new_centers = _local_optimize(new_centers, max_iter=50)
        
        s_new, phi_new = _final_score(new_centers)
        
        delta = s_new - s_cur
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            current_state = new_state
            current_centers = new_centers
            s_cur = s_new
            phi_cur = phi_new
            
            if s_new < best_score:
                best_score = s_new
                best_state = new_state[:]
                best_centers = new_centers.copy()
                best_phi = phi_new
                
        T *= alpha
        
    # Final refinement on the best state found
    if best_state is not None:
        final_centers = candidates[best_state].copy()
        final_centers = _local_optimize(final_centers, max_iter=1000)
        
        # Re-evaluate best score and angle
        s_final, phi_final = _final_score(final_centers)
        
        # Construct output
        inner = np.column_stack((final_centers[:, 0], final_centers[:, 1], np.full(11, 0.0)))
        outer = np.array([0.0, 0.0, phi_final]) # Center should be centroid?
        
        # The outer hexagon center should be the centroid of the inner hexagons?
        # No, it should be the center that minimizes the side length.
        # My _outer_side_for_points assumes center is at origin?
        # No, it calculates width relative to origin?
        # Let's check _outer_side_for_points.
        # It computes proj = points @ normals.T.
        # widths = max - min.
        # This is the width of the set of points.
        # The side length of the enclosing hexagon is max_width / sqrt(3).
        # This is independent of the center of the hexagon, as long as the hexagon is centered appropriately.
        # The center of the minimal enclosing regular hexagon is the center of the minimal bounding box in those directions?
        # Actually, for a regular hexagon, the center is the average of the max and min projections in each normal direction?
        # If we shift the points, the width doesn't change.
        # So the side length is correct.
        # But we need to output the center (x, y) of the outer hexagon.
        # The problem says: "outer of shape (3,), (x, y, angle_deg) of the outer regular hexagon"
        # And "Coordinates are first shifted by the outer center".
        # Wait, "Coordinates are first shifted by the outer center" implies the input to the validator is shifted?
        # No, "shifted inner hexagons must keep edge lengths...".
        # This implies we return coordinates relative to the outer center?
        # Or we return absolute coordinates, and the validator shifts them?
        # "inner of shape (11, 3), rows (x, y, angle_deg) ... outer ... (x, y, angle_deg)"
        # Usually, we return absolute coordinates.
        # Let's assume we return absolute coordinates.
        # The center of the outer hexagon should be the centroid of the inner hexagons?
        # Or the center that minimizes the max distance?
        # For a regular hexagon, the optimal center is the intersection of the midlines of the bounding box?
        # Let's just use the centroid of the inner centers as the outer center.
        
        cx = np.mean(final_centers[:, 0])
        cy = np.mean(final_centers[:, 1])
        
        outer = np.array([cx, cy, phi_final])
        
        # Shift inner coordinates relative to outer center?
        # "Coordinates are first shifted by the outer center"
        # This phrasing is tricky.
        # "shifted inner hexagons must keep edge lengths within 1e-8 of 1"
        # This implies the validator takes our (x,y), subtracts outer (x,y), and checks.
        # So we should return absolute coordinates.
        
        return inner, outer, s_final

    # Fallback
    inner = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.17, 0],
        [1.25, 2.17, 0],
        [-1.25, -2.17, 0],
        [1.25, -2.17, 0],
        [-3.75, 2.17, 0],
        [3.75, 2.17, 0],
        [-3.75, -2.17, 0],
        [3.75, -2.17, 0],
    ], dtype=float)
    outer = np.array([0, 0, 0], dtype=float)
    return inner, outer, 8.0

# EVOLVE-BLOCK-END
