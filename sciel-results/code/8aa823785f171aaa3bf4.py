"""Rewritten program: combinatorial search over structured hexagon packings."""
import numpy as np
import time

N = 11

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def hex_vertices(cx, cy, angle, s=1.0):
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def min_container_side(points):
        """
        Given a set of 2D points, find the minimum side length of a regular hexagon
        that contains all points. Try all rotations and optimal centers.
        Returns (S, center_x, center_y, angle_degrees).
        """
        pts = np.array(points)
        best_S = float('inf')
        best_c = (0.0, 0.0)
        best_a = 0.0
        
        # The container hexagon's normals are at angles phi + pi/6 + k*pi/3 for k=0..5
        # Equivalent: 3 pairs of opposite normals at angles phi + pi/6, phi + pi/6 + pi/3, phi + pi/6 + 2*pi/3
        # For each phi, the 3 normal directions (mod pi) are:
        #   d0 = phi + pi/6
        #   d1 = phi + pi/6 + pi/3
        #   d2 = phi + pi/6 + 2*pi/3
        # For each direction d, the range of projections is max(dot(p,d)) - min(dot(p,d))
        # The optimal center is the midpoint in each direction, but for a hexagon the center
        # that minimizes max range is: for each pair of opposite normals, the center projects
        # to the midpoint of the projection range. These 3 midpoints should coincide at the
        # optimal center (for a hexagon, the center is determined by the 3 midpoints).
        
        # Actually, for a given rotation, the minimum S is:
        # S = max over the 3 normal pairs of (range_i) / sqrt(3)
        # where range_i is the width of the point set in direction i.
        # The center is the intersection of the 3 midlines, but since the hexagon is centrally
        # symmetric, the center that works is: for each normal direction n_i, the center's
        # projection should be (max_i + min_i)/2. These 3 constraints are consistent iff
        # the point set is "hexagonally symmetric" enough. In general, we can compute:
        # c = average of the 3 midpoints? No, that's not right.
        # 
        # Better approach: For a given rotation phi, define the 3 unit normal vectors n0, n1, n2.
        # The hexagon centered at c with side S contains all points iff for all p and all k:
        #   |dot(p - c, n_k)| <= S * sqrt(3) / 2
        # This is equivalent to:
        #   dot(c, n_k) in [max_p dot(p,n_k) - S*sqrt(3)/2, min_p dot(p,n_k) + S*sqrt(3)/2]
        # For feasibility, we need: max_p dot(p,n_k) - S*sqrt(3)/2 <= min_p dot(p,n_k) + S*sqrt(3)/2
        # i.e., range_k = max_p dot(p,n_k) - min_p dot(p,n_k) <= S * sqrt(3)
        # So S >= range_k / sqrt(3) for all k.
        # And c must satisfy the 3 interval constraints. The intervals are:
        #   [max_k - S*sqrt(3)/2, min_k + S*sqrt(3)/2] for each k
        # These intervals are non-empty iff S >= range_k/sqrt(3).
        # To find c, we need a point whose projections onto n0, n1, n2 lie in these intervals.
        # Since n0, n1, n2 span R^2 (they're not all parallel), we can solve for c.
        # Let A = [n0^T; n1^T; n2^T] (3x2 matrix). We need A @ c in the box.
        # The minimal S is max(range_k / sqrt(3)), and at that S, the intervals are non-empty.
        # We can find c by taking the midpoint of each interval and solving the least-squares
        # or checking feasibility. Actually, for the minimal S, the intervals just touch, so
        # there's a unique c (or a line segment). Let's just compute c as the centroid of
        # the 3 midpoints... no. Let me just use: c such that dot(c, n_k) = (max_k + min_k)/2
        # for k=0,1,2. This is a 3x2 system; solve in least squares.
        
        for phi in np.linspace(0, np.pi/3, 37):  # 37 samples of rotation
            n0 = np.array([np.cos(phi + np.pi/6), np.sin(phi + np.pi/6)])
            n1 = np.array([np.cos(phi + np.pi/6 + np.pi/3), np.sin(phi + np.pi/6 + np.pi/3)])
            n2 = np.array([np.cos(phi + np.pi/6 + 2*np.pi/3), np.sin(phi + np.pi/6 + 2*np.pi/3)])
            
            A = np.stack([n0, n1, n2])  # 3x2
            projs = pts @ A.T  # 66x3
            max_p = projs.max(axis=0)
            min_p = projs.min(axis=0)
            ranges = max_p - min_p
            S_candidate = np.max(ranges) / np.sqrt(3)
            
            if S_candidate < best_S:
                best_S = S_candidate
                # Find center: solve A @ c = mid where mid = (max_p + min_p)/2
                mid = (max_p + min_p) / 2.0
                # Least squares solution
                c, _, _, _ = np.linalg.lstsq(A, mid, rcond=None)
                best_c = (c[0], c[1])
                best_a = phi
        
        return best_S, best_c, np.degrees(best_a)
    
    def generate_configurations():
        """Generate candidate configurations of 11 unit hexagons."""
        configs = []
        
        # Triangular lattice basis
        a1 = np.array([np.sqrt(3), 0])
        a2 = np.array([np.sqrt(3)/2, 1.5])
        
        # Standard hexagonal cluster: 1 center + 6 ring + 4 more
        # Ring 0: (0,0)
        # Ring 1: 6 neighbors at distance sqrt(3)
        # Ring 2: 12 neighbors at distance 2*sqrt(3) - we pick 4
        
        # Configuration 1: Center + 6 ring + 4 from ring 2 at "corner" positions
        # Ring 2 positions are at 2*a1, 2*a2, 2*a2-a1, -a1, -a1+a2, -a1+2*a2, etc.
        # Let me enumerate ring 2 positions:
        ring2_positions = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                pos = i * a1 + j * a2
                dist = np.linalg.norm(pos)
                if abs(dist - 2*np.sqrt(3)) < 0.1 or abs(dist - np.sqrt(3)*np.sqrt(7)) < 0.1:
                    # Check if it's actually ring 2 (not ring 0 or 1)
                    if dist > 1.5 * np.sqrt(3) and dist < 2.5 * np.sqrt(3):
                        ring2_positions.append(pos)
        
        # Deduplicate
        unique_ring2 = []
        for p in ring2_positions:
            if not any(np.allclose(p, q, atol=0.1) for q in unique_ring2):
                unique_ring2.append(p)
        
        ring1_positions = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                pos = i * a1 + j * a2
                if np.linalg.norm(pos) > 0.1:
                    ring1_positions.append(pos)
        unique_ring1 = []
        for p in ring1_positions:
            if not any(np.allclose(p, q, atol=0.1) for q in unique_ring1):
                unique_ring1.append(p)
        
        # Config: center + all 6 ring1 + 4 from ring2
        base = [np.array([0.0, 0.0])] + unique_ring1
        if len(unique_ring2) >= 4:
            # Try different subsets of 4 from ring2
            from itertools import combinations
            for subset in combinations(range(len(unique_ring2)), 4):
                config = base + [unique_ring2[i] for i in subset]
                configs.append(config)
        
        # Configuration 2: All on triangular lattice, compact 11-point sets
        # Enumerate all sets of 11 lattice points that form a connected cluster
        # This is expensive; instead, use known compact shapes
        # Shape: a "hexagonal number" arrangement. H(n) = 3n(n-1)+1
        # H(1)=1, H(2)=7, H(3)=19. So 11 is between H(2) and H(3).
        # Take the 7-point hex and add 4 more adjacent points.
        
        # Another approach: brute-force over small grid of lattice positions
        # and pick the 11 that minimize the container size.
        # Grid: all lattice points within radius 3*sqrt(3) of origin
        lattice_points = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                pos = i * a1 + j * a2
                if np.linalg.norm(pos) < 4 * np.sqrt(3):
                    lattice_points.append(pos)
        
        # Deduplicate
        unique_lattice = []
        for p in lattice_points:
            if not any(np.allclose(p, q, atol=0.1) for q in unique_lattice):
                unique_lattice.append(p)
        
        # For each subset of 11 from unique_lattice that includes the origin (or is connected),
        # compute the container size. This is C(|unique_lattice|, 11) which is too large.
        # Instead, use a greedy or local search.
        
        # Greedy: start with the 7-point hex, add 4 points one at a time that minimize container size
        base7 = [np.array([0.0, 0.0])] + unique_ring1
        current = list(base7)
        
        for _ in range(4):
            best_add = None
            best_S = float('inf')
            for p in unique_lattice:
                if any(np.allclose(p, q, atol=0.1) for q in current):
                    continue
                candidate = current + [p]
                S, _, _ = min_container_side(candidate)
                if S < best_S:
                    best_S = S
                    best_add = p
            if best_add is not None:
                current.append(best_add)
            else:
                break
        
        configs.append(current)
        
        # Also try: start with different 7-point bases (rotated hexagons)
        for rot_angle in [0, np.pi/6, np.pi/3]:
            R = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                          [np.sin(rot_angle), np.cos(rot_angle)]])
            base7_rot = [R @ np.array([0.0, 0.0])] + [R @ p for p in unique_ring1]
            current_rot = list(base7_rot)
            for _ in range(4):
                best_add = None
                best_S = float('inf')
                for p in unique_lattice:
                    if any(np.allclose(p, q, atol=0.1) for q in current_rot):
                        continue
                    candidate = current_rot + [p]
                    S, _, _ = min_container_side(candidate)
                    if S < best_S:
                        best_S = S
                        best_add = p
                if best_add is not None:
                    current_rot.append(best_add)
                else:
                    break
            configs.append(current_rot)
        
        return configs
    
    # Generate and evaluate configurations
    configs = generate_configurations()
    
    best_S = float('inf')
    best_config = None
    best_container = None
    
    for config in configs:
        if len(config) != N:
            continue
        S, (cx, cy), angle_deg = min_container_side(config)
        if S < best_S:
            best_S = S
            best_config = config
            best_container = (cx, cy, angle_deg)
        
        if time.time() > deadline - 20:
            break
    
    # If no valid config found, use a simple fallback
    if best_config is None:
        a1 = np.array([np.sqrt(3), 0])
        a2 = np.array([np.sqrt(3)/2, 1.5])
        best_config = [np.array([0.0, 0.0])]
        for k in range(6):
            theta = k * np.pi / 3
            best_config.append(np.array([np.sqrt(3)*np.cos(theta), np.sqrt(3)*np.sin(theta)]))
        for k in range(4):
            theta = k * np.pi / 2 + np.pi/6
            best_config.append(2*np.sqrt(3)*np.array([np.cos(theta), np.sin(theta)]))
        best_S, (cx, cy), angle_deg = min_container_side(best_config)
        best_container = (cx, cy, angle_deg)
    
    # Build the inner hexagons (all axis-aligned, angle 0)
    inner = []
    for p in best_config:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    cx, cy, angle_deg = best_container
    return inner, (float(cx), float(cy)), float(best_S), float(angle_deg)
