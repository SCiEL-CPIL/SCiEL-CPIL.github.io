import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def hex_verts(cx, cy, s, rot):
        angles = np.deg2rad(np.arange(6) * 60 + rot)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_dist_hexagons(v1, v2):
        """
        Calculate minimum distance between two convex hexagons.
        Returns 0 if they intersect.
        Optimized for speed using vectorized operations where possible,
        but primarily relies on fast edge-vertex distance checks.
        """
        # 1. Check for intersection using Separating Axis Theorem (SAT)
        # For two convex polygons, if there is a separating axis, they don't intersect.
        # The axes to check are the normals of the edges of both polygons.
        
        # Get normals for v1 and v2
        # Edge vectors for v1
        e1 = np.roll(v1, -1, axis=0) - v1
        # Normals (perpendicular to edges)
        n1 = np.column_stack([-e1[:, 1], e1[:, 0]])
        
        e2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.column_stack([-e2[:, 1], e2[:, 0]])
        
        axes = np.vstack([n1, n2])
        
        # Project both polygons onto each axis
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for overlap
            if max1 < min2 or max2 < min1:
                # Separating axis found, no intersection
                break
        else:
            # No separating axis found, polygons intersect
            return 0.0

        # 2. Calculate minimum distance
        # Since they don't intersect, min distance is the minimum distance
        # between any edge of v1 and any vertex of v2, or vice versa.
        
        min_d = float('inf')
        
        # Distance from vertices of v2 to edges of v1
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            v = p2 - p1
            v_len_sq = np.dot(v, v)
            
            if v_len_sq == 0: continue
            
            # Project all v2 vertices onto this edge
            w = v2 - p1
            c1 = np.dot(w, v) / v_len_sq
            # Clamp projection to [0, 1]
            t = np.clip(c1, 0, 1)
            
            # Closest points on edge
            closest_on_edge = p1 + t[:, np.newaxis] * v
            # Distances
            dists = np.linalg.norm(v2 - closest_on_edge, axis=1)
            min_d = min(min_d, np.min(dists))
            
        # Distance from vertices of v1 to edges of v2
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i+1)%6]
            v = p2 - p1
            v_len_sq = np.dot(v, v)
            
            if v_len_sq == 0: continue
            
            w = v1 - p1
            c1 = np.dot(w, v) / v_len_sq
            t = np.clip(c1, 0, 1)
            
            closest_on_edge = p1 + t[:, np.newaxis] * v
            dists = np.linalg.norm(v1 - closest_on_edge, axis=1)
            min_d = min(min_d, np.min(dists))

        return min_d

    def evaluate_solution(params):
        """
        params: [x1, y1, ..., x12, y12, outer_s]
        Rotations are fixed to 0 for all inner hexagons to maintain symmetry and reduce search space.
        Outer hexagon is fixed at rotation 0.
        Returns: (outer_side_length, validity)
        """
        xs = params[0:12]
        ys = params[12:24]
        outer_s = params[24]
        
        # Generate inner hexagons (all rotated 0)
        inner_verts = []
        for i in range(12):
            v = hex_verts(xs[i], ys[i], 1.0, 0.0)
            inner_verts.append(v)
            
        # Check disjointness
        for i in range(12):
            for j in range(i+1, 12):
                d = min_dist_hexagons(inner_verts[i], inner_verts[j])
                if d < 1e-5: 
                    return float('inf'), False

        # Check containment in outer hexagon
        # Outer hexagon center (0,0), side length outer_s, rotation 0.
        # Inradius r = outer_s * sqrt(3) / 2
        r = outer_s * np.sqrt(3) / 2.0
        
        # Normals for outer hexagon edges (rotation 0 means vertices at 0, 60, ...)
        # Edge normals are at 30, 90, 150, 210, 270, 330 degrees
        normals = np.deg2rad(30 + np.arange(6)*60)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        for i in range(12):
            # Check all vertices of inner hexagon i
            dists = np.dot(inner_verts[i], n_vecs.T)
            if np.any(dists > r + 1e-5):
                return float('inf'), False

        return outer_s, True

    # Initialize
    # Start with a tight honeycomb-like configuration
    # Center hexagon at (0,0)
    # 6 around it, 5 more around those?
    # Let's just use a random cluster near origin to start, 
    # but a structured start is better for SA.
    
    # Honeycomb coordinates for 12 hexagons (side 1)
    # Distance between centers in honeycomb is 2 * inradius = sqrt(3)
    d_center = np.sqrt(3)
    
    # Generate 12 points in a hexagonal grid pattern
    # Center (0,0)
    # Ring 1: 6 points
    # Ring 2: 5 points (to make 12 total)
    
    init_params = np.zeros(25)
    
    # Center
    init_params[0] = 0.0
    init_params[12] = 0.0
    
    # Ring 1
    for k in range(6):
        angle = k * 60
        x = d_center * np.cos(np.deg2rad(angle))
        y = d_center * np.sin(np.deg2rad(angle))
        idx = 1 + k
        init_params[idx] = x
        init_params[idx + 12] = y
        
    # Ring 2 (partial)
    # Just pick 5 points from the next ring
    # Or simply expand the radius slightly and randomize
    # Let's place them at distance 2 * d_center
    r2 = 2 * d_center
    # Angles: 0, 60, 120, 180, 240 (skip 300 to keep symmetry? or just pick 5)
    angles_r2 = [0, 60, 120, 180, 240]
    for k, angle in enumerate(angles_r2):
        x = r2 * np.cos(np.deg2rad(angle))
        y = r2 * np.sin(np.deg2rad(angle))
        idx = 7 + k
        init_params[idx] = x
        init_params[idx + 12] = y
        
    # Initial outer side length
    # The bounding hexagon of this cluster
    # Max radius is r2 = 2*sqrt(3) ~ 3.46
    # Inradius of outer hex must be >= 3.46 + 1 (inner radius)
    # r_outer = 4.46
    # outer_s = r_outer * 2 / sqrt(3) ~ 5.15
    init_params[24] = 6.0
    
    best_params = init_params.copy()
    best_val, best_valid = evaluate_solution(best_params)
    if not best_valid:
        # Expand until valid
        while not best_valid:
            best_params[24] *= 1.1
            best_val, best_valid = evaluate_solution(best_params)
            
    current_params = best_params.copy()
    current_val = best_val
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    steps = 0
    max_steps = 100000
    
    # Move strategies
    # 1. Move one hexagon
    # 2. Move all hexagons (translate cluster)
    # 3. Scale cluster (adjust outer_s and positions proportionally)
    
    while steps < max_steps:
        if time.time() > deadline:
            break
            
        new_params = current_params.copy()
        strategy = np.random.rand()
        
        if strategy < 0.7:
            # Move one hexagon
            idx = np.random.randint(12)
            # Small step
            step = 0.1 * (1.0 - steps/max_steps) # Anneal step size
            new_params[idx] += np.random.normal(0, step)
            new_params[idx+12] += np.random.normal(0, step)
        elif strategy < 0.85:
            # Translate entire cluster
            tx = np.random.normal(0, 0.05)
            ty = np.random.normal(0, 0.05)
            new_params[0:12] += tx
            new_params[12:24] += ty
        else:
            # Try to shrink
            # Scale positions towards center and reduce outer_s
            scale = 1.0 + np.random.normal(0, 0.01)
            if scale < 0.9: scale = 0.9
            new_params[0:12] *= scale
            new_params[12:24] *= scale
            new_params[24] *= scale
            
        new_val, new_valid = evaluate_solution(new_params)
        
        # Acceptance
        if new_val < current_val:
            current_params = new_params
            current_val = new_val
            if new_valid and current_val < best_val:
                best_params = new_params.copy()
                best_val = current_val
        else:
            delta = new_val - current_val
            # If new is invalid, new_val is inf, exp(-inf/T) = 0, so rejected
            if np.random.rand() < np.exp(-delta / T):
                current_params = new_params
                current_val = new_val
                
        T *= cooling
        steps += 1
        
    # Extract best solution
    xs = best_params[0:12]
    ys = best_params[12:24]
    outer_s = best_params[24]
    
    inner = []
    for i in range(12):
        inner.append([xs[i], ys[i], 0.0])
        
    return inner, (0.0, 0.0), outer_s, 0.0

if __name__ == "__main__":
    # Test
    inner, center, side, angle = construct()
    print(f"Side length: {side}")
    for i in inner:
        print(i)
