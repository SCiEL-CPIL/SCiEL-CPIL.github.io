"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.spatial import cKDTree

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Hexagon geometry helpers
    def hexagon_vertices(cx, cy, angle_deg, side=1.0):
        """Return 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + math.radians(60 * i)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices
    
    def point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
        """Check if point is inside a regular hexagon."""
        dx = px - cx
        dy = py - cy
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        max_dot = 0
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            dot = lx * math.cos(theta) + ly * math.sin(theta)
            if dot > max_dot:
                max_dot = dot
        return max_dot <= side * math.sqrt(3) / 2 + 1e-9
    
    def hexagon_overlap_dist(a, b):
        """
        Compute minimum distance between centers of two unit hexagons 
        with given angles such that they just touch (no overlap).
        For same orientation, this is 2.0 (vertex touching).
        For general orientations, we use a conservative bound.
        """
        # For same orientation: min distance = 2.0
        # For different orientations, the minimum can be slightly less
        # but for safety, use 2.0 as a conservative lower bound
        return 2.0
    
    def compute_min_outer_hexagon(centers, angle_deg=0.0):
        """
        Given centers of unit hexagons (all same orientation),
        compute the minimum side length of a regular hexagon that contains all of them.
        
        The outer hexagon must contain all vertices of all inner hexagons.
        Each inner hexagon's vertices are at distance 1 from its center.
        So we need the outer hexagon to contain the union of circles of radius 1 
        centered at each inner center.
        
        For a regular hexagon with side length S (circumradius S), centered at (cx, cy)
        with orientation angle_deg, a point (px, py) is inside if:
        For each of 6 face normals (at angles 30+60k + angle_deg), the projection 
        of (px-cx, py-cy) onto that normal <= S * sqrt(3)/2.
        
        We want to minimize S. The optimal center is the centroid.
        """
        n = len(centers)
        cx = sum(c[0] for c in centers) / n
        cy = sum(c[1] for c in centers) / n
        
        # For a regular hexagon with circumradius R (= side length S),
        # the inradius (apothem) is R * sqrt(3)/2.
        # A point (px, py) is inside the hexagon centered at (cx, cy) with 
        # orientation o if for all 6 face normals n_k:
        #   (px-cx, py-cy) · n_k <= R * sqrt(3)/2
        # where n_k are unit vectors at angles (30 + 60k + o) degrees.
        #
        # The minimum R is: max over all points and all normals of 
        #   (point - center) · n_k / (sqrt(3)/2)
        # but we also need to consider that the hexagons have extent.
        #
        # Actually, the constraint is: for each inner hexagon vertex v,
        # v must be inside the outer hexagon.
        # The outer hexagon with circumradius R, center (cx,cy), orientation o
        # contains point p iff for all k: (p - c) · n_k(o) <= R * sqrt(3)/2
        # where n_k(o) = (cos(30+60k+o), sin(30+60k+o))
        
        # Collect all vertices of all inner hexagons
        all_vertices = []
        for x, y in centers:
            verts = hexagon_vertices(x, y, angle_deg)
            all_vertices.extend(verts)
        
        # For a given orientation o of the outer hexagon, find minimum R
        def min_R_for_orientation(o_deg):
            o_rad = math.radians(o_deg)
            R = 0
            for vx, vy in all_vertices:
                dx = vx - cx
                dy = vy - cy
                for k in range(6):
                    theta = math.radians(30 + 60 * k) + o_rad
                    dot = dx * math.cos(theta) + dy * math.sin(theta)
                    # dot <= R * sqrt(3)/2  =>  R >= dot / (sqrt(3)/2)
                    if dot > 0:
                        R = max(R, dot / (math.sqrt(3) / 2))
            return R
        
        # Try several orientations to find the minimum
        best_R = float('inf')
        for o_deg in range(0, 60, 1):  # Try every degree from 0 to 59
            R = min_R_for_orientation(o_deg)
            if R < best_R:
                best_R = R
        
        # Also try some finer orientations around the best
        if best_R < float('inf'):
            for o_deg in np.linspace(0, 60, 600):
                R = min_R_for_orientation(o_deg)
                if R < best_R:
                    best_R = R
        
        return best_R, (cx, cy)
    
    def check_valid(inner, ocx, ocy, oside, oangle_deg):
        """Check if all inner hexagons are inside outer and don't overlap."""
        # Check containment: all vertices of inner hexagons must be in outer
        for x, y, a in inner:
            verts = hexagon_vertices(x, y, a)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, ocx, ocy, oangle_deg, oside):
                    return False
        
        # Check non-overlap: center distance >= 2.0 for same orientation
        n = len(inner)
        for i in range(n):
            for j in range(i + 1, n):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 2.0 - 1e-6:
                    return False
        return True
    
    # Generate hexagonal lattice positions
    # Triangular lattice with spacing 2.0 (minimum distance for same-orientation unit hexagons)
    sqrt3 = math.sqrt(3)
    positions = []
    max_r = 5  # Search radius in lattice units
    for r in range(-max_r, max_r + 1):
        for c in range(-max_r, max_r + 1):
            x = 2.0 * c + (r % 2) * 1.0
            y = r * sqrt3
            positions.append((x, y))
    
    # Sort by distance from origin
    positions.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    
    best_result = None
    best_score = float('inf')
    
    # Strategy 1: Use compact hexagonal clusters
    # Centered hexagonal numbers: 1, 7, 19, 37, ...
    # For N=11: use the 7-hex cluster (radius 1) + 4 more hexagons
    
    # Generate all subsets of size N from the lattice positions
    # But that's too many. Instead, use greedy/structured approaches.
    
    import random
    random.seed(42)
    
    # Approach A: Start with compact clusters and add remaining hexagons
    # The 7-hex cluster: center + 6 neighbors at distance 2
    # Positions for 7-hex cluster (same orientation, spacing 2):
    # Center: (0,0)
    # Neighbors: (2,0), (1, sqrt3), (-1, sqrt3), (-2,0), (-1,-sqrt3), (1,-sqrt3)
    
    cluster_7 = [(0.0, 0.0),
                 (2.0, 0.0), (1.0, sqrt3), (-1.0, sqrt3),
                 (-2.0, 0.0), (-1.0, -sqrt3), (1.0, -sqrt3)]
    
    # For N=11, we need 4 more. Try different arrangements.
    
    # Method: Try many candidate arrangements
    arrangements = []
    
    # Arrangement 1: 7-hex cluster + 4 in a row extending from one side
    # Extend from the right: add (3, sqrt3), (3, -sqrt3), (4, 0), (5, 0)? 
    # No, need to check distances.
    
    # Let's use a more systematic approach: 
    # Try all subsets of the lattice of size N that are "compact"
    
    # Generate candidate positions in a larger area
    lattice_positions = []
    max_coord = 8
    for r in range(-max_coord, max_coord + 1):
        for c in range(-max_coord, max_coord + 1):
            x = 2.0 * c + (r % 2) * 1.0
            y = r * sqrt3
            if abs(x) <= 12 and abs(y) <= 12:
                lattice_positions.append((x, y))
    
    lattice_positions.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    
    # Greedy approach: try to build compact clusters
    # For each starting position, greedily add nearest available positions
    
    def greedy_cluster(start_idx, n_needed, lattice_pos):
        """Greedily build a cluster of n_needed positions starting from start_idx."""
        placed = [lattice_pos[start_idx]]
        remaining = [i for i in range(len(lattice_pos)) if i != start_idx]
        
        while len(placed) < n_needed and remaining:
            # Find the nearest position to the cluster that doesn't overlap
            best_i = None
            best_dist = float('inf')
            for i in remaining:
                x, y = lattice_pos[i]
                # Check distance to all placed
                min_d = min(math.sqrt((x - px)**2 + (y - py)**2) for px, py in placed)
                if min_d >= 2.0 - 1e-6:
                    if min_d < best_dist:
                        best_dist = min_d
                        best_i = i
            
            if best_i is None:
                break
            placed.append(lattice_pos[best_i])
            remaining.remove(best_i)
        
        return placed
    
    # Try greedy from several starting positions
    tried_starts = set()
    for start_idx in range(min(50, len(lattice_positions))):
        if time.time() > deadline:
            break
        
        placed = greedy_cluster(start_idx, N, lattice_positions)
        if len(placed) < N:
            continue
        
        # Center the cluster
        cx = sum(p[0] for p in placed) / N
        cy = sum(p[1] for p in placed) / N
        centered = [(x - cx, y - cy) for x, y in placed]
        
        # Compute minimum outer hexagon
        oside, (ocx, ocy) = compute_min_outer_hexagon(centered)
        
        # Verify
        inner = [[x, y, 0.0] for x, y in centered]
        if check_valid(inner, ocx, ocy, oside, 0.0):
            if oside < best_score:
                best_score = oside
                best_result = (inner, (ocx, ocy), oside, 0.0)
    
    # Also try the 7+4 structured approach
    # 7-hex cluster centered at origin
    # Add 4 more hexagons in various configurations
    
    # Configuration: add hexagons at the "corners" of the 7-cluster
    # The 7-cluster has 6 outer hexagons. We can add hexagons adjacent to pairs of them.
    
    # Let me try: extend the 7-cluster by adding hexagons at positions that are 
    # adjacent to 2 existing hexagons (filling gaps)
    
    # For the 7-cluster, the "gap" positions (where a new hex could fit touching 2 existing)
    # are at distance sqrt(3) from center, at angles 30, 90, 150, 210, 270, 330 degrees
    # Wait, let me think more carefully.
    
    # The 6 outer hexagons of the 7-cluster are at:
    # (2,0), (1,sqrt3), (-1,sqrt3), (-2,0), (-1,-sqrt3), (1,-sqrt3)
    # 
    # A new hexagon at (3, sqrt3) would be adjacent to (2,0) and (1,sqrt3):
    # dist to (2,0): sqrt(1+3) = 2 ✓
    # dist to (1,sqrt3): sqrt(4+0) = 2 ✓
    # 
    # So we can add hexagons at the "edge" positions.
    
    edge_positions = [
        (3.0, sqrt3), (3.0, -sqrt3),
        (-3.0, sqrt3), (-3.0, -sqrt3),
        (1.0, 2*sqrt3), (-1.0, 2*sqrt3),
        (1.0, -2*sqrt3), (-1.0, -2*sqrt3),
    ]
    
    # Try adding 4 of these 8 positions to the 7-cluster
    from itertools import combinations
    for combo in combinations(range(8), 4):
        if time.time() > deadline:
            break
        
        placed = cluster_7 + [edge_positions[i] for i in combo]
        
        # Check all pairwise distances
        valid = True
        for i in range(len(placed)):
            for j in range(i+1, len(placed)):
                dx = placed[i][0] - placed[j][0]
                dy = placed[i][1] - placed[j][1]
                if math.sqrt(dx*dx + dy*dy) < 2.0 - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Center the cluster
        cx = sum(p[0] for p in placed) / N
        cy = sum(p[1] for p in placed) / N
        centered = [(x - cx, y - cy) for x, y in placed]
        
        # Compute minimum outer hexagon
        oside, (ocx, ocy) = compute_min_outer_hexagon(centered)
        
        # Verify
        inner = [[x, y, 0.0] for x, y in centered]
        if check_valid(inner, ocx, ocy, oside, 0.0):
            if oside < best_score:
                best_score = oside
                best_result = (inner, (ocx, ocy), oside, 0.0)
    
    # Also try rotating some inner hexagons to potentially reduce the bounding hexagon
    # But for now, keep all at 0 degrees for simplicity
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END

