import numpy as np
import time
import math

N = 11

def get_hex_vertices(x, y, theta):
    """
    Returns vertices of a unit regular hexagon centered at (x,y) with rotation theta (radians).
    """
    # Base vertices for a unit hexagon (side 1) centered at (0,0)
    # Flat-top orientation: vertices at 0, 60, 120, 180, 240, 300 degrees
    # Side length 1 => Distance from center to vertex is 1.
    angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    verts = np.vstack([np.cos(angles + theta), np.sin(angles + theta)]).T
    verts += np.array([x, y])
    return verts

def get_hex_normals(theta):
    """
    Returns unit normals to the sides of a unit regular hexagon rotated by theta.
    The normals point outwards.
    For a flat-top hexagon, sides are perpendicular to angles 30, 90, 150, 210, 270, 330 degrees.
    """
    base_normals = np.array([
        [np.cos(np.pi/6), np.sin(np.pi/6)],
        [np.cos(np.pi/2), np.sin(np.pi/2)],
        [np.cos(5*np.pi/6), np.sin(5*np.pi/6)],
        [np.cos(7*np.pi/6), np.sin(7*np.pi/6)],
        [np.cos(3*np.pi/2), np.sin(3*np.pi/2)],
        [np.cos(11*np.pi/6), np.sin(11*np.pi/6)]
    ])
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return base_normals @ R.T

def min_separation(verts1, normals1, verts2, normals2):
    """
    Returns the minimum separation distance between two convex polygons.
    Positive means separated, negative means overlapping (depth).
    Uses Separating Axis Theorem.
    """
    min_gap = float('inf')
    
    # Check all normals from both polygons
    # We can combine them, but checking 12 axes is fine.
    all_normals = np.vstack([normals1, normals2])
    
    for n in all_normals:
        # Project verts1
        proj1 = verts1 @ n
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        
        # Project verts2
        proj2 = verts2 @ n
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        # Calculate gap
        # If intervals [min1, max1] and [min2, max2] are disjoint, there is a separating axis
        gap = max(min1, min2) - max(max1, max2)
        
        if gap > 0:
            return float('inf') # Separated
        
        if gap < min_gap:
            min_gap = gap
            
    return min_gap

def calculate_metrics(params):
    """
    params: array of shape (N, 3) -> [x, y, theta]
    Returns: (side_length, overlap_penalty, is_valid)
    """
    xs = params[:, 0]
    ys = params[:, 1]
    thetas = params[:, 2]
    
    # 1. Calculate required outer hexagon side length
    # Outer hexagon is fixed at (0,0) with rotation 0.
    # Normals for outer hexagon (flat-top):
    outer_normals = np.array([
        [np.cos(np.pi/6), np.sin(np.pi/6)],
        [np.cos(np.pi/2), np.sin(np.pi/2)],
        [np.cos(5*np.pi/6), np.sin(5*np.pi/6)]
    ])
    # Note: Only 3 unique normals needed for max projection due to symmetry?
    # No, we need to check all 6 faces to ensure containment.
    # Actually, for a regular hexagon centered at origin, the condition is:
    # max over all vertices of: max( |v . n1|, |v . n2|, |v . n3| ) <= Apothem
    # Apothem A = S * sqrt(3) / 2.
    # So S = 2 * A / sqrt(3).
    
    max_proj = 0.0
    for i in range(N):
        verts = get_hex_vertices(xs[i], ys[i], thetas[i])
        # Project onto the 3 unique normal directions (using absolute value handles the opposite side)
        # n1, n2, n3 are 120 deg apart.
        projs = np.abs(verts @ outer_normals.T)
        current_max = np.max(projs)
        if current_max > max_proj:
            max_proj = current_max
            
    side = 2.0 * max_proj / np.sqrt(3)
    
    # 2. Calculate overlap penalty
    penalty = 0.0
    is_valid = True
    
    # Precompute vertices and normals for all hexagons to avoid recalculation in pairs?
    # N is small (11), so O(N^2) pairs is 55 pairs.
    # Each pair check involves 12 projections of 6 points. ~72 operations.
    # Total ~4000 operations per evaluation. Very fast.
    
    verts_list = []
    normals_list = []
    for i in range(N):
        v = get_hex_vertices(xs[i], ys[i], thetas[i])
        n = get_hex_normals(thetas[i])
        verts_list.append(v)
        normals_list.append(n)
        
    for i in range(N):
        for j in range(i+1, N):
            sep = min_separation(verts_list[i], normals_list[i], verts_list[j], normals_list[j])
            if sep < 0:
                # Overlap
                is_valid = False
                # Penalty proportional to overlap depth squared
                penalty += (-sep) ** 2
                
    return side, penalty, is_valid

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    best_side = float('inf')
    best_params = None
    best_valid = False
    
    # Initial Configuration: Hexagonal grid
    d = np.sqrt(3) # Distance between centers of adjacent unit hexagons in a grid
    # 7-hex cluster (center + 6 neighbors)
    init_centers = [
        (0, 0),
        (d, 0), (d/2, 1.5), (-d/2, 1.5),
        (-d, 0), (-d/2, -1.5), (d/2, -1.5)
    ]
    # Add 4 more to make 11
    # Extending the hex grid:
    # (2d, 0), (1.5d, 3), (-1.5d, 3), (0, 4.5) is one way.
    # Let's try a more compact arrangement if possible, or just standard grid.
    additional_centers = [
        (2*d, 0),
        (1.5*d, 3.0),
        (-1.5*d, 3.0),
        (0, 4.5)
    ]
    
    all_centers = init_centers + additional_centers
    init_params = np.zeros((N, 3))
    for i in range(N):
        init_params[i, 0] = all_centers[i][0]
        init_params[i, 1] = all_centers[i][1]
        init_params[i, 2] = 0.0
        
    # Simulated Annealing
    np.random.seed(42)
    
    # We will run multiple restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial state
        if r == 0:
            params = init_params.copy()
        else:
            # Perturb the best found so far, or random start
            if best_params is not None and np.random.rand() < 0.5:
                params = best_params + np.random.normal(0, 0.5, (N, 3))
                params[:, 2] = np.random.uniform(-np.pi, np.pi, N)
            else:
                # Random scatter within a reasonable area
                params = np.zeros((N, 3))
                for i in range(N):
                    if i == 0:
                        params[i] = [0, 0, 0]
                    else:
                        # Place in a ring or random
                        angle = np.random.uniform(0, 2*np.pi)
                        radius = np.random.uniform(1.0, 4.0)
                        params[i] = [radius * np.cos(angle), radius * np.sin(angle), np.random.uniform(-np.pi, np.pi)]
                        
        # SA Parameters
        T = 1.0
        T_min = 0.001
        alpha = 0.995 # Cooling rate
        # Adjust steps based on time remaining
        # Estimate time per step: ~0.001s? Let's assume 1000 steps/s.
        # 10s per restart -> 10000 steps.
        steps_per_restart = 5000
        
        current_side, current_penalty, current_valid = calculate_metrics(params)
        current_cost = current_side + 100.0 * current_penalty
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            # Perturb a random hexagon
            k = np.random.randint(N)
            old_x, old_y, old_theta = params[k]
            
            # Move
            move_dist = np.random.uniform(0.1, 0.5)
            move_angle = np.random.uniform(0, 2*np.pi)
            new_x = old_x + move_dist * np.cos(move_angle)
            new_y = old_y + move_dist * np.sin(move_angle)
            
            # Rotate
            new_theta = old_theta + np.random.uniform(-0.2, 0.2)
            
            params[k] = [new_x, new_y, new_theta]
            
            new_side, new_penalty, new_valid = calculate_metrics(params)
            new_cost = new_side + 100.0 * new_penalty
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
                current_cost = new_cost
                current_side = new_side
                current_penalty = new_penalty
                current_valid = new_valid
                
                if new_valid and new_side < best_side:
                    best_side = new_side
                    best_params = params.copy()
                    best_valid = True
            else:
                # Revert
                params[k] = [old_x, old_y, old_theta]
                
            T *= alpha
            if T < T_min:
                break
                
    # If no valid packing found, return a large side (fallback)
    if not best_valid:
        # Fallback: return the initial grid side length even if it overlaps? 
        # No, must be valid. 
        # If SA failed, try a simple deterministic grid and calculate side.
        # The initial grid might overlap if not careful, but for 11 hexes in a grid, 
        # the standard hex grid of 11 hexes (1 center, 6 around, 4 more) 
        # fits in a hexagon of side 3? 
        # Let's just return the seed's fallback logic if we fail.
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Construct output
    inner = []
    for i in range(N):
        x, y, ang = best_params[i]
        # Convert radians to degrees for output
        ang_deg = np.rad2deg(ang)
        inner.append([float(x), float(y), float(ang_deg)])
        
    return inner, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}")
