import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Unit regular hexagon vertices (flat-top orientation, side length 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    base_vertices = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def rotate_vertices(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return base_vertices @ R.T
    
    def support_function(points, direction):
        """Max dot product of points with direction vector."""
        if len(points) == 0:
            return 0.0
        return np.max(points @ direction)
    
    def required_side_length(inner_centers, inner_angles, outer_center, outer_angle):
        """
        Compute the minimum side length S of outer hexagon (centered at outer_center, 
        rotated by outer_angle) that contains all inner hexagons.
        """
        # Outer hexagon has 6 sides. The support function of a regular hexagon with 
        # side length S centered at origin, flat-top, rotated by outer_angle:
        # The hexagon is the intersection of 6 half-planes.
        # For a regular hexagon with side S, the apothem (distance from center to side) is S * sqrt(3)/2.
        # The outward normals to the 6 sides are at angles outer_angle + k*60 for k=0..5.
        
        # For each of the 6 sides of the outer hexagon, we need:
        #   max over all inner hexagon vertices of (vertex - outer_center) @ normal_k <= S * sqrt(3)/2
        # for the side with normal normal_k pointing outward.
        
        # Actually, let me reconsider. A regular hexagon with side length S has:
        # - 6 sides, each at distance (apothem) = S * sqrt(3) / 2 from center
        # - The outward normals point at angles phi_k = outer_angle + k * 60 degrees
        
        # The constraint is that all inner hexagon vertices must lie inside the outer hexagon.
        # For side k with outward normal n_k, the constraint is:
        #   (v - c_outer) @ n_k <= apothem = S * sqrt(3) / 2
        # for all vertices v of all inner hexagons.
        
        # So S >= max over all k and all v of: (v - c_outer) @ n_k / (sqrt(3)/2)
        #    = max over all k and all v of: 2/sqrt(3) * (v - c_outer) @ n_k
        
        # Also need S > 0 and the inner hexagons must be disjoint from each other.
        
        oc = np.array(outer_center)
        oa = outer_angle * np.pi / 180.0
        
        # Normals of outer hexagon sides
        normal_angles = oa + np.arange(6) * np.pi / 3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        apothem_factor = np.sqrt(3.0) / 2.0
        
        max_val = 0.0
        for idx in range(N):
            center = np.array(inner_centers[idx])
            angle = inner_angles[idx] * np.pi / 180.0
            verts = rotate_vertices(angle) + center
            # For each normal, compute max dot product
            dots = verts @ normals.T  # shape (6, 6)
            for k in range(6):
                val = np.max(dots[:, k])
                if val > max_val:
                    max_val = val
        
        S = max_val / apothem_factor
        if S < 0:
            S = 0.0
        return S
    
    def hexagon_distance(c1, a1, c2, a2):
        """
        Approximate minimum distance between two unit hexagons with centers c1,c2 and angles a1,a2.
        Use the fact that two convex polygons are disjoint iff there exists a separating axis.
        For regular hexagons, we can check the 12 edge normals (6 from each).
        """
        c1 = np.array(c1)
        c2 = np.array(c2)
        a1 = a1 * np.pi / 180.0
        a2 = a2 * np.pi / 180.0
        
        v1 = rotate_vertices(a1) + c1
        v2 = rotate_vertices(a2) + c2
        
        # Get edge normals for each hexagon
        def edge_normals(vertices):
            normals = []
            for i in range(6):
                edge = vertices[(i+1)%6] - vertices[i]
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-12:
                    normals.append(n / norm)
            return normals
        
        n1 = edge_normals(v1)
        n2 = edge_normals(v2)
        
        all_normals = n1 + n2
        
        min_dist = float('inf')
        for n in all_normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for overlap
            if max1 < min2 or max2 < min1:
                # Disjoint, distance along this axis
                d = min2 - max1 if max1 < min2 else min1 - max2
                if d > 0:
                    min_dist = min(min_dist, d)
                    continue
            else:
                # Overlap on this axis, not a separating axis
                pass
        
        # If no separating axis found, they overlap
        if min_dist == float('inf'):
            return -1.0  # Overlapping
        
        return min_dist
    
    def is_valid_config(inner_centers, inner_angles, outer_center, outer_angle):
        """Check if all inner hexagons are disjoint."""
        for i in range(N):
            for j in range(i+1, N):
                d = hexagon_distance(inner_centers[i], inner_angles[i], 
                                     inner_centers[j], inner_angles[j])
                if d < -1e-6:  # Overlapping
                    return False
        return True
    
    def objective(inner_centers, inner_angles, outer_center, outer_angle):
        """Return the required side length, or infinity if invalid."""
        if not is_valid_config(inner_centers, inner_angles, outer_center, outer_angle):
            return float('inf')
        return required_side_length(inner_centers, inner_angles, outer_center, outer_angle)
    
    # Initialization: compact hexagonal arrangement
    # Use axial coordinates for a hexagonal lattice
    # For 12 hexagons, use a 4x3 parallelogram or a more compact shape
    
    def generate_initial_config():
        """Generate a compact initial configuration of 12 unit hexagons."""
        # Use a hexagonal grid pattern
        # Axial coordinates: x = sqrt(3)*(q + r/2), y = 1.5*r for pointy-top
        # For flat-top: x = 1.5*q, y = sqrt(3)*(r + q/2)
        
        # Let's use a simple approach: place in a hexagonal cluster
        # Center + 6 around it + 5 more
        
        sqrt3 = np.sqrt(3.0)
        
        # Flat-top hexagon lattice spacing:
        # Horizontal neighbor: (1.5, 0)
        # Upper-right: (0.75, sqrt3/2)
        # Upper-left: (-0.75, sqrt3/2)
        # etc.
        
        # Generate positions using axial coordinates
        positions = []
        
        # Use a set of axial coordinates that form a compact shape
        # For 12 hexagons, try a 3-4-3-2 arrangement or similar
        
        # Simpler: use a rectangular grid with offset
        # Row 0: 4 hexes
        # Row 1: 4 hexes  
        # Row 2: 4 hexes
        
        # Flat-top hex lattice:
        # (x, y) = (1.5 * col + (row % 2) * 0.75, row * sqrt3/2)
        
        for row in range(3):
            for col in range(4):
                x = 1.5 * col + (row % 2) * 0.75
                y = row * sqrt3 / 2
                positions.append([x, y])
        
        # Center the configuration
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        centers = [[p[0] - cx, p[1] - cy] for p in positions]
        angles = [0.0] * N
        
        return centers, angles
    
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Local search
    def local_search(centers, angles, outer_center, outer_angle, max_iter=5000):
        """Perform local search to minimize S."""
        current_S = objective(centers, angles, outer_center, outer_angle)
        if current_S == float('inf'):
            # Try to fix invalid configuration
            return centers, angles, outer_center, outer_angle, float('inf')
        
        best_local_S = current_S
        best_local_centers = [c[:] for c in centers]
        best_local_angles = angles[:]
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random hexagon to perturb
            idx = random.randint(0, N-1)
            
            # Random perturbation
            dx = random.gauss(0, 0.1)
            dy = random.gauss(0, 0.1)
            da = random.gauss(0, 5.0)  # degrees
            
            new_center = [centers[idx][0] + dx, centers[idx][1] + dy]
            new_angle = angles[idx] + da
            
            new_centers = [c[:] for c in centers]
            new_centers[idx] = new_center
            new_angles = angles[:]
            new_angles[idx] = new_angle
            
            new_S = objective(new_centers, new_angles, outer_center, outer_angle)
            
            if new_S < best_local_S:
                best_local_S = new_S
                best_local_centers = new_centers
                best_local_angles = new_angles
                centers = new_centers
                angles = new_angles
                current_S = new_S
                # Reduce step size occasionally
                if random.random() < 0.1:
                    pass  # Step size is fixed for simplicity
        
        return best_local_centers, best_local_angles, outer_center, outer_angle, best_local_S
    
    # Run multiple restarts
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate initial configuration
        centers, angles = generate_initial_config()
        
        # Add some randomness
        for i in range(N):
            centers[i][0] += random.gauss(0, 0.5)
            centers[i][1] += random.gauss(0, 0.5)
            angles[i] = random.uniform(-15, 15)
        
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        # Local search
        centers, angles, outer_center, outer_angle, S = local_search(
            centers, angles, outer_center, outer_angle, max_iter=2000
        )
        
        if S < best_S:
            best_S = S
            best_inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
            best_center = outer_center
            best_angle = outer_angle
    
    # If no valid solution found, return a trivial valid one
    if best_inner is None:
        # Place hexagons in a line far apart
        best_inner = [[i * 3.0, 0.0, 0.0] for i in range(N)]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_S = required_side_length(best_inner, [b[2] for b in best_inner], best_center, best_angle)
    
    # Final answer
    inner = best_inner
    center = best_center
    side = best_S
    angle = best_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
