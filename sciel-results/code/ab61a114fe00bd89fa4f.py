import numpy as np
import time
import itertools
from scipy.optimize import minimize_scalar

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _get_axes(hex_verts):
    """Get the 6 edge normals (axes) of a hexagon."""
    axes = []
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
        axes.append(normal)
    return np.array(axes)

def _sat_overlap(v1, v2):
    """
    Check if two convex polygons overlap using Separating Axis Theorem.
    Returns True if they overlap (or touch), False if separated.
    """
    axes = np.vstack([_get_axes(v1), _get_axes(v2)])
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def _min_dist_polygons(v1, v2):
    """
    Calculate minimum distance between two convex polygons.
    Returns 0 if they overlap.
    """
    if _sat_overlap(v1, v2):
        return 0.0
        
    min_dist = np.inf
    n1, n2 = len(v1), len(v2)
    
    def point_segment_dist(p, a, b):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    for p in v1:
        for i in range(n2):
            d = point_segment_dist(p, v2[i], v2[(i+1)%n2])
            if d < min_dist:
                min_dist = d
                
    for p in v2:
        for i in range(n1):
            d = point_segment_dist(p, v1[i], v1[(i+1)%n1])
            if d < min_dist:
                min_dist = d
                
    return min_dist

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check containment
    for iv in inner_verts:
        for v in iv:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                cross = (b[0]-a[0])*(v[1]-a[1]) - (b[1]-a[1])*(v[0]-a[0])
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False, "outside"
                
    # Check disjointness
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _sat_overlap(inner_verts[i], inner_verts[j]):
                return False, "overlap"
                
    return True, "valid"

def _compute_side_length(inner_verts, outer_angle, outer_center):
    """
    Compute the minimum side length of the outer hexagon 
    to contain all inner hexagons, given the outer angle and center.
    """
    max_proj = 0.0
    cx, cy = outer_center
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        # Project all vertices of all inner hexagons onto this normal
        # relative to the outer center
        for v in inner_verts:
            shifted = v - np.array([cx, cy])
            projs = np.dot(shifted, normal)
            max_proj = max(max_proj, np.max(projs))
            
    return max_proj * 2.0 / np.sqrt(3.0)

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Lattice generation
    # Triangular lattice with spacing sqrt(3) ensures unit hexagons (orientation 0) do not overlap
    # if they are on different lattice points.
    # Basis vectors: u = (sqrt(3), 0), v = (sqrt(3)/2, 1.5)
    sqrt3 = np.sqrt(3)
    u = np.array([sqrt3, 0.0])
    v = np.array([sqrt3/2, 1.5])
    
    # Generate lattice points within a reasonable radius
    R = 4 # Radius in lattice units
    lattice_points = []
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            p = i * u + j * v
            # Filter points that are too far from origin to keep the set small
            if np.linalg.norm(p) < R * sqrt3:
                lattice_points.append(p)
                
    lattice_points = np.array(lattice_points)
    
    # We need to choose 12 points out of these such that no two are on adjacent lattice points
    # Adjacent lattice points correspond to distance sqrt(3).
    # In a triangular lattice, the nearest neighbors are at distance sqrt(3).
    # So we need an independent set of size 12 in the graph where edges connect points at distance sqrt(3).
    
    # Precompute adjacency
    n_pts = len(lattice_points)
    adj = [set() for _ in range(n_pts)]
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            d = np.linalg.norm(lattice_points[i] - lattice_points[j])
            if d < sqrt3 * 1.01: # Close enough to be adjacent
                adj[i].add(j)
                adj[j].add(i)
                
    # Backtracking search for independent sets of size 12
    # We will try to find a few different configurations to give the optimizer variety
    # Or just one good one? Let's find a few.
    
    solutions = []
    
    def backtrack(start, current_set):
        if len(current_set) == 12:
            solutions.append(list(current_set))
            return True # Found one, stop? Or find more? 
                        # Finding multiple diverse solutions might help if one is local optimum.
                        # But let's just find a handful.
            
        if len(current_set) + (n_pts - start) < 12:
            return False
            
        for i in range(start, n_pts):
            # Check if i is compatible with all in current_set
            compatible = True
            for j in current_set:
                if i in adj[j]:
                    compatible = False
                    break
            if not compatible:
                continue
                
            current_set.add(i)
            if backtrack(i+1, current_set):
                return True
            current_set.remove(i)
            
        return False
        
    # To find diverse solutions, we can force different starting points or use randomization
    # But a simple DFS might find a "compact" one which is likely good.
    # Let's try to find 5 solutions by pruning previously found solutions' starting nodes?
    # Simpler: Just run DFS once. If it finds one, use it.
    # Actually, the "flower" pattern is 1 center + 6 ring + 5 outer.
    # On a triangular lattice, the center is (0,0).
    # Neighbors are at dist sqrt(3).
    # So we can pick (0,0). Then we must skip its 6 neighbors.
    # The next ring is at distance 2*sqrt(3) or sqrt(7)?
    # Distances in triangular lattice:
    # (1,0) -> sqrt(3)
    # (2,0) -> 2*sqrt(3)
    # (1,1) -> sqrt(3+3+3/4*4?) No.
    # Let's just rely on the backtracker.
    
    # To ensure we get a valid set, we might need to search deeper if the first few fail.
    # But n_pts is small (~50).
    
    # Let's run backtracking
    if not backtrack(0, set()):
        # Fallback if DFS fails (should not happen for N=12 in this range)
        # Just pick 12 random non-adjacent points
        pass

    # We might want multiple solutions. Let's modify backtrack to find up to 3 solutions.
    # For now, let's assume the first one found is good (usually compact).
    # To be safe, let's run a slightly modified version that finds a few.
    
    # Actually, the specific arrangement matters for the bounding box.
    # A "compact" cluster (like a hexagon of hexagons) is generally best for minimizing the bounding hexagon.
    # The backtracker tends to pick low indices first. If we sort lattice points by distance from origin,
    # it will pick the most central compact cluster.
    
    # Sort lattice points by distance from origin to bias towards compact clusters
    dists = np.linalg.norm(lattice_points, axis=1)
    order = np.argsort(dists)
    sorted_pts = lattice_points[order]
    # Rebuild adjacency for sorted points? Or just map indices.
    # Easier: just use the sorted points directly and rebuild adj.
    
    lattice_points = sorted_pts
    n_pts = len(lattice_points)
    adj = [set() for _ in range(n_pts)]
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            d = np.linalg.norm(lattice_points[i] - lattice_points[j])
            if d < sqrt3 * 1.01:
                adj[i].add(j)
                adj[j].add(i)
                
    solutions = []
    def backtrack_multi(start, current_set, count_limit=3):
        if len(current_set) == 12:
            solutions.append(list(current_set))
            if len(solutions) >= count_limit:
                return True
            return False
            
        if len(current_set) + (n_pts - start) < 12:
            return False
            
        for i in range(start, n_pts):
            compatible = True
            for j in current_set:
                if i in adj[j]:
                    compatible = False
                    break
            if not compatible:
                continue
                
            current_set.add(i)
            if backtrack_multi(i+1, current_set, count_limit):
                return True
            current_set.remove(i)
            
        return False
        
    backtrack_multi(0, set(), 5)
    
    if not solutions:
        # Emergency fallback: linear arrangement
        pts = [np.array([i*sqrt3, 0]) for i in range(N)]
        solutions.append(list(range(N))) # Indices don't match if we just appended, so we need to map back
        # Actually, let's just create a valid config manually if backtracking fails
        # But backtracking should work.
        # For safety, let's construct a valid set manually if solutions is empty
        # A simple line: (0,0), (2,0), (4,0)... wait, dist must be > sqrt(3).
        # (0,0) and (2,0) dist 2*sqrt(3) > sqrt(3). OK.
        # So indices 0, 2, 4... in a 1D lattice subset?
        # Let's just use the first 12 points from the lattice that are pairwise non-adjacent.
        pass

    # Optimize outer angle for each solution
    for sol_idx, sol in enumerate(solutions):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Get the centers
        centers = lattice_points[sol]
        # All inner hexagons have angle 0.0
        inner_hexes = [[centers[i][0], centers[i][1], 0.0] for i in range(N)]
        inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
        
        # Define objective for outer angle
        def obj_outer_angle(theta):
            # theta is in radians
            # Compute side length required
            # We can also shift the outer center to minimize the side length?
            # The task allows outer center to be optimized.
            # For a given set of inner hexagons and a fixed orientation of the outer hexagon,
            # the optimal center is the one that minimizes the max projection.
            # This is a 2D problem. But we can approximate or solve it.
            # Actually, for a convex body K and a container C (hexagon) with fixed orientation,
            # the minimal size is achieved when the container is centered at the "center of symmetry" of the bounding box?
            # No, it's the Chebyshev center of the complement?
            # Simpler: Just optimize the outer center (x,y) and angle (theta) together?
            # That's 3D.
            
            # Let's fix the outer center to the centroid of the inner hexagons first.
            # Then optimize angle.
            # Then optimize center?
            
            # Let's try a nested optimization or just grid search for center if time permits.
            # Given the symmetry of the lattice, the centroid is a good guess.
            
            centroid = np.mean(centers, axis=0)
            
            max_proj = 0.0
            for k in range(6):
                angle = theta + k * np.pi / 3
                normal = np.array([np.cos(angle), np.sin(angle)])
                
                for v in inner_verts:
                    shifted = v - centroid
                    projs = np.dot(shifted, normal)
                    max_proj = max(max_proj, np.max(projs))
                    
            return max_proj * 2.0 / np.sqrt(3.0)
            
        # Optimize theta in [0, pi/3]
        try:
            res = minimize_scalar(obj_outer_angle, bounds=(0, np.pi/3), method='bounded', options={'maxiter': 100})
            best_theta = res.x
            best_side_val = res.fun
            
            # Now, try to optimize the center for this best_theta
            # We can do a small grid search or 2D minimization around the centroid
            centroid = np.mean(centers, axis=0)
            
            def obj_center(cx, cy, theta):
                max_proj = 0.0
                for k in range(6):
                    angle = theta + k * np.pi / 3
                    normal = np.array([np.cos(angle), np.sin(angle)])
                    for v in inner_verts:
                        shifted = v - np.array([cx, cy])
                        projs = np.dot(shifted, normal)
                        max_proj = max(max_proj, np.max(projs))
                return max_proj * 2.0 / np.sqrt(3.0)
                
            # Use Nelder-Mead for center
            x0 = [centroid[0], centroid[1]]
            res_c = minimize(lambda x: obj_center(x[0], x[1], best_theta), x0, method='Nelder-Mead', options={'maxiter': 200})
            best_center = res_c.x
            best_side_val = res_c.fun
            
            if best_side_val < best_side:
                best_side = best_side_val
                best_inner = inner_hexes
                best_outer_center = (best_center[0], best_center[1])
                best_outer_angle = np.rad2deg(best_theta)
                
        except Exception as e:
            pass

    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

