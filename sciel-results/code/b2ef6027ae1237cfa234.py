import numpy as np
import time
import itertools

N = 12
DEADLINE = 145.0

# Precompute unit hexagon vertices (side length 1, centered at origin)
# Vertices are at angles 0, 60, ..., 300 degrees
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

def get_hex_vertices(centers):
    """
    centers: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = centers[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts

def sat_min_separation(v1, v2):
    """
    Compute minimum separation between two convex polygons using SAT.
    Returns the minimum distance (positive if separated, negative if overlapping).
    """
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    axes = np.vstack([edges1, edges2])
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / (norms + 1e-15)
    
    proj1 = v1 @ axes.T
    proj2 = v2 @ axes.T
    
    sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                    proj2.min(axis=0) - proj1.max(axis=0))
    
    return np.max(sep)

def compute_required_side(centers, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute the minimum outer side length needed to contain the convex hull 
    of all inner hexagon vertices.
    """
    verts = get_hex_vertices(np.array(centers, dtype=float))
    all_verts = verts.reshape(-1, 2)
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, -c_o]]) # Note: Rotation matrix for -theta
    
    # Center the points relative to outer center
    v_rel = all_verts - np.array(outer_center)
    v_rot = v_rel @ R_o.T
    
    # Normals of the outer hexagon (apothem directions)
    # For a regular hexagon with flat top/bottom or pointy top?
    # Standard hexagon with vertices at 0, 60... has normals at 30, 90...
    normals = np.array([
        [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
    ])
    
    projs = v_rot @ normals.T
    abs_projs = np.abs(projs)
    max_extent = np.max(abs_projs)
    
    # Apothem = side * sqrt(3)/2
    return max_extent / (np.sqrt(3) / 2)

def is_valid_configuration(centers):
    """
    Check if the configuration has no overlaps.
    """
    verts = get_hex_vertices(np.array(centers, dtype=float))
    for i in range(N):
        v1 = verts[i]
        for j in range(i + 1, N):
            v2 = verts[j]
            sep = sat_min_separation(v1, v2)
            if sep < -1e-5:
                return False
    return True

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    # Strategy: Enumerate subsets of a hexagonal lattice.
    # The optimal packing of 12 unit hexagons is likely a subset of a hexagonal lattice
    # or a slight perturbation thereof. We generate a grid of candidate centers
    # on a hexagonal lattice and select 12 that minimize the bounding side.
    
    # Lattice vectors for unit hexagons (distance between centers of adjacent hexes is 2*apothem = sqrt(3))
    # Wait, if side length is 1, distance between centers of adjacent non-overlapping hexes is 2 * (sqrt(3)/2) = sqrt(3).
    # Let's verify:
    # Hexagon A at (0,0). Vertices at (1,0).
    # Hexagon B at (sqrt(3), 0). Vertices at (sqrt(3)-1, 0).
    # Distance between (1,0) and (sqrt(3)-1, 0) is sqrt(3)-2 which is negative. They overlap.
    # Correct distance between centers of touching unit hexagons:
    # The distance from center to vertex is 1.
    # The distance from center to edge (apothem) is sqrt(3)/2.
    # If two hexagons touch edge-to-edge, the distance between centers is 2 * apothem = sqrt(3).
    # If they touch vertex-to-vertex, the distance is 2.
    # The "tightest" packing is edge-to-edge, so lattice spacing is sqrt(3).
    
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5]) # This gives distance sqrt( (sqrt(3)/2)^2 + 1.5^2 ) = sqrt(3/4 + 9/4) = sqrt(3). Correct.
    
    # Generate a grid of candidate positions
    # We need to cover a region large enough to contain 12 hexagons.
    # A 3x4 block is roughly 3*sqrt(3) x 4*1.5 = 5.2 x 6.
    # Let's generate points in a range [-4, 4] x [-4, 4] on the lattice.
    
    candidates = []
    # Range of lattice indices
    max_idx = 5
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            pos = i * a1 + j * a2
            # Add a small offset to break symmetry? No, keep it pure lattice first.
            # We will try all angles 0, 30, 60 for the inner hexagons?
            # Actually, if they are on a lattice, angle 0 is optimal for lattice alignment.
            # But maybe rotating some helps?
            # For now, assume angle 0 for all, as this is the standard "hexagonal close packing".
            candidates.append([pos[0], pos[1], 0.0])
            
    candidates = np.array(candidates)
    
    # To find the best subset of 12, we can't do C(N, 12) directly if N is large.
    # However, the optimal configuration is likely "compact".
    # We can use a heuristic: start with the center of the grid and add neighbors.
    # Or, simpler: Generate all subsets of a smaller, denser grid near the origin.
    
    # Let's refine the grid to be denser? No, the lattice spacing is fixed by non-overlap.
    # The only freedom is *which* lattice points we pick.
    # Since the grid is infinite, we restrict to a window.
    # Let's pick a window of 15x15 points centered at (0,0).
    
    # Filter candidates to a reasonable bounding box
    # Max side length we care about is probably < 6.
    # So centers shouldn't be further than ~3 from origin.
    candidates = candidates[np.linalg.norm(candidates[:, :2], axis=1) < 4.0]
    
    # We have roughly 100 candidates. Choosing 12 is C(100, 12) which is huge.
    # We need a smarter search.
    # Greedy construction:
    # 1. Pick a center.
    # 2. Add the neighbor that minimizes the increase in bounding box side length.
    # Repeat until 12 are placed.
    # Try this for multiple starting points.
    
    # Precompute all pairwise separations? No, just check validity on the fly.
    # Since it's a lattice, any subset of the lattice is valid (no overlaps).
    # So we just need to minimize the bounding side.
    
    # Let's define a function to get the bounding side of a set of lattice points.
    def get_side_for_points(points):
        # points: (k, 2) array of centers
        # Create dummy centers with angle 0
        centers = np.zeros((len(points), 3))
        centers[:, :2] = points
        return compute_required_side(centers)
    
    # Greedy Search
    # We will try starting from several "seed" points.
    # For each seed, we grow the cluster by adding the point that minimizes the new side length.
    
    # To speed up, we precompute the vertices for each candidate center?
    # No, compute_required_side is fast enough for small N.
    
    # Let's identify the "core" candidates: those close to the origin.
    # Sort candidates by distance to origin.
    dists = np.linalg.norm(candidates[:, :2], axis=1)
    sorted_indices = np.argsort(dists)
    core_candidates = candidates[sorted_indices[:50]]
    
    # Try growing from each of the first 10 core candidates
    # And also try growing from a "center of mass" of a random subset?
    # Let's just try growing from the 10 closest points to origin.
    
    start_indices = sorted_indices[:10]
    
    for start_idx in start_indices:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        current_points = [candidates[start_idx]]
        current_side = get_side_for_points(np.array(current_points))
        
        # Available points
        available = [c for i, c in enumerate(candidates) if i != start_idx]
        
        for step in range(N - 1):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            best_new_point = None
            best_new_side = float('inf')
            
            # To save time, we can limit the search to points close to the current cluster
            current_center = np.mean(np.array(current_points), axis=0)
            # Filter available points to those within a certain distance
            # The cluster grows roughly linearly.
            max_dist = 2.0 + step * 0.5
            
            candidate_points = []
            for p in available:
                if np.linalg.norm(p - current_center) < max_dist:
                    candidate_points.append(p)
            
            # If no candidates in range, pick the closest one
            if not candidate_points:
                candidate_points = [min(available, key=lambda p: np.linalg.norm(p - current_center))]
            
            for p in candidate_points:
                new_points = current_points + [p]
                new_side = get_side_for_points(np.array(new_points))
                if new_side < best_new_side:
                    best_new_side = new_side
                    best_new_point = p
            
            if best_new_point is None:
                break
                
            current_points.append(best_new_point)
            current_side = best_new_side
            
            # Remove used point from available
            # This is O(N^2) due to list removal, but N is small (50-100)
            available = [p for p in available if not (p[0] == best_new_point[0] and p[1] == best_new_point[1])]
            
        # Evaluate final configuration
        if len(current_points) == N:
            final_centers = np.array(current_points)
            side = get_side_for_points(final_centers)
            
            # Check validity (should be true for lattice)
            if is_valid_configuration(final_centers):
                if side < best_score:
                    best_score = side
                    best_result = (final_centers.tolist(), (0.0, 0.0), side, 0.0)
    
    # If greedy didn't find anything or we want to try other strategies:
    # Try a few random subsets of the lattice to see if we get lucky.
    if time.time() - start_time < DEADLINE - 20:
        np.random.seed(42)
        for _ in range(1000):
            if time.time() - start_time > DEADLINE - 10:
                break
            # Pick 12 random points from the core candidates
            idxs = np.random.choice(len(core_candidates), size=N, replace=False)
            points = core_candidates[idxs]
            side = get_side_for_points(points)
            if side < best_score:
                # Verify validity
                if is_valid_configuration(points):
                    best_score = side
                    best_result = (points.tolist(), (0.0, 0.0), side, 0.0)
    
    # Fallback if no valid result found
    if best_result is None:
        # Simple circular arrangement
        inner_default = []
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = 3.5 * np.cos(theta)
            y = 3.5 * np.sin(theta)
            inner_default.append([x, y, 0.0])
        
        # Check if this is valid
        if not is_valid_configuration(inner_default):
             # Adjust radius
             for r in [4.0, 4.5, 5.0]:
                 inner_default = []
                 for i in range(N):
                     theta = 2 * np.pi * i / N
                     x = r * np.cos(theta)
                     y = r * np.sin(theta)
                     inner_default.append([x, y, 0.0])
                 if is_valid_configuration(inner_default):
                     break
        
        required_side = compute_required_side(inner_default)
        best_result = (inner_default, (0.0, 0.0), required_side, 0.0)
        best_score = required_side

    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
