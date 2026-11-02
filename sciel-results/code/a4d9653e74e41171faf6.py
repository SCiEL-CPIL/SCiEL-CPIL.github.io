"""Packing 12 unit hexagons in a minimal outer hexagon.
Strategy: rigid honeycomb cluster with global rotation optimization."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 12
APOTHEM = np.sqrt(3) / 2
S3 = np.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    HEX_ANGLES = np.deg2rad(np.arange(0, 360, 60))
    HEX_NORMALS = np.array([
        [np.cos(np.deg2rad(30 + k * 60)), np.sin(np.deg2rad(30 + k * 60))]
        for k in range(6)
    ])

    def hex_verts(cx, cy, side, angle_deg):
        a = np.radians(angle_deg)
        angles = a + HEX_ANGLES
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def all_points_in_hex(pts, cx, cy, side, angle_deg):
        """pts: (m,2) array. Returns True if all inside the hexagon."""
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        a = np.radians(angle_deg)
        cos_a = np.cos(-a)
        sin_a = np.sin(-a)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        for nx, ny in HEX_NORMALS:
            if np.any(lx * nx + ly * ny > side * APOTHEM + 1e-9):
                return False
        return True

    def sat_disjoint(v1, v2, tol=1e-9):
        """SAT: returns True if polygons v1, v2 (6,2) are disjoint."""
        edges = np.vstack([np.roll(v1, -1, 0) - v1, np.roll(v2, -1, 0) - v2])
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
        for axis in normals:
            p1 = v1 @ axis
            p2 = v2 @ axis
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
        return False

    def pairwise_disjoint(verts_list):
        for i in range(len(verts_list)):
            for j in range(i + 1, len(verts_list)):
                if not sat_disjoint(verts_list[i], verts_list[j]):
                    return False
        return True

    def compute_outer_side(all_verts, oa_deg):
        """Compute minimum outer side length for given vertex set and outer angle."""
        oa = np.radians(oa_deg)
        normals = np.array([[np.cos(oa + np.deg2rad(30 + k * 60)),
                             np.sin(oa + np.deg2rad(30 + k * 60))] for k in range(6)])
        max_side = 0.0
        for nx, ny in normals:
            d = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            m = np.max(d)
            if m > 0:
                s = m / APOTHEM
                if s > max_side:
                    max_side = s
        return max_side

    def compute_outer_side_centered(verts_centered, oa_deg):
        """verts_centered: vertices relative to cluster centroid."""
        return compute_outer_side(verts_centered, oa_deg)

    # --- Generate honeycomb cluster configurations ---
    # A honeycomb cluster: positions on hexagonal lattice with spacing 2 (side=1 hexagons)
    # Hex lattice: positions at (2*i + j, S3*j) for integers i,j, with appropriate spacing
    # For unit hexagons (side=1), adjacent centers are 2 apart.
    # Hex grid: center at (0,0), neighbors at distance 2 in 6 directions.
    
    def make_honeycomb_positions():
        """Generate all positions in a hexagonal grid region."""
        # Hex grid with axial coordinates (q, r)
        # Position: x = 2*q + S3*r, y = 3*r  ... no, let me think more carefully.
        # For regular hexagons of side 1:
        # - Flat-top: vertices at angles 0,60,120,180,240,300
        # - Pointy-top: vertices at angles 30,90,150,210,270,330
        # Adjacent hex centers are 2*apothem = sqrt(3) apart? No.
        # For flat-top hex with side s: width = 2s, height = s*sqrt(3)
        # Horizontal spacing: 1.5*s, vertical spacing: s*sqrt(3)
        # Actually for our hex_verts with angles 0,60,120,...:
        # vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), ...
        # So width (flat-to-flat) = s*sqrt(3), height (point-to-point) = 2s
        # Adjacent centers: horizontally = s*sqrt(3), vertically = 2s
        # For s=1: horiz = sqrt(3), vert = 2
        # 
        # Actually let me just use the standard hex grid:
        # For pointy-top hexagons (vertices at 30,90,...):
        #   x = sqrt(3) * (col + row/2), y = 1.5 * row  (for side=1)
        # For flat-top hexagons (vertices at 0,60,...):
        #   x = 1.5 * col, y = sqrt(3) * (row + col/2)
        #
        # Our hex_verts uses angles 0,60,120,... which is flat-top.
        # So: x = 1.5 * col, y = sqrt(3) * (row + col/2)
        # For side=1: adjacent distance = 2 (diameter)
        # Wait, let me verify: two flat-top hexes side by side horizontally:
        # Center distance = 1.5 * 1 = 1.5? No, that's for the grid spacing.
        # Actually the distance between centers of adjacent flat-top hexes:
        # Horizontal neighbors: 1.5 * side = 1.5? That can't be right for side=1.
        # Let me just compute: hex at (0,0) has rightmost vertex at (1,0).
        # Hex at (d,0) has leftmost vertex at (d-1,0).
        # For them to just touch: d-1 = 1, so d = 2.
        # So horizontal spacing = 2 for flat-top.
        # Vertical: top vertex at (0.5, sqrt(3)/2), bottom at (-0.5, -sqrt(3)/2)
        # Height = sqrt(3). Two stacked vertically: spacing = sqrt(3)? 
        # Top hex bottom vertex at y = -sqrt(3)/2, bottom hex top vertex at y = d + sqrt(3)/2
        # Touch: -sqrt(3)/2 = d + sqrt(3)/2 => d = -sqrt(3)? That's wrong direction.
        # For vertical stacking of flat-top hexes, they share an edge.
        # The edge from (0.5, sqrt(3)/2) to (-0.5, sqrt(3)/2) has midpoint (0, sqrt(3)/2).
        # The adjacent hex above shares this edge, so its center is at (0, sqrt(3)).
        # So vertical spacing = sqrt(3).
        # 
        # General hex grid for flat-top, side=1:
        # col spacing: 2 (horizontal)
        # row spacing: sqrt(3) (vertical)  
        # offset: odd columns shifted by sqrt(3)/2
        # Actually: x = 2*col, y = sqrt(3)*row + (sqrt(3)/2 if col is odd else 0)
        # Hmm, let me just enumerate positions and check distances.
        
        positions = []
        # Use axial coordinates: positions at (2*q, S3*r) won't work directly.
        # Let me just place them on a grid and verify.
        # For flat-top hex side 1:
        # Horizontal neighbor: (2, 0)
        # Upper-right neighbor: (1, S3)
        # Upper-left neighbor: (-1, S3)
        # Vertical neighbor: (0, 2*S3) -- wait no
        # Let me just compute: the 6 neighbors of (0,0) for flat-top hex side 1:
        # The hex has vertices at (1,0), (0.5, S3/2), (-0.5, S3/2), (-1,0), (-0.5,-S3/2), (0.5,-S3/2)
        # Edges: (1,0)-(0.5,S3/2), (0.5,S3/2)-(-0.5,S3/2), etc.
        # Edge (0.5,S3/2)-(-0.5,S3/2) is horizontal, midpoint (0, S3/2), length 1.
        # Adjacent hex sharing this edge has center at (0, S3).
        # Edge (1,0)-(0.5,S3/2): midpoint (0.75, S3/4), direction (-0.5, S3/2), length 1.
        # Adjacent hex: center at (0.75+0.75, S3/4+S3/4) = (1.5, S3/2)? 
        # Let me verify: hex at (1.5, S3/2) has leftmost vertex at (0.5, S3/2) and lower-left at (1, S3/2 - S3/2) = (1,0). Yes!
        # So neighbors of (0,0): (0, S3), (1.5, S3/2), (1.5, -S3/2), (0, -S3), (-1.5, -S3/2), (-1.5, S3/2)
        # Wait, let me recheck (0, S3): hex at (0,S3) has bottom vertex at (0, S3-S3/2) = (0, S3/2)? No.
        # Hex at (0, S3) vertices: (1, S3), (0.5, S3+S3/2), (-0.5, S3+S3/2), (-1, S3), (-0.5, S3-S3/2), (0.5, S3-S3/2)
        # = (1,S3), (0.5, 1.5*S3), (-0.5, 1.5*S3), (-1,S3), (-0.5, S3/2), (0.5, S3/2)
        # So bottom edge is from (-0.5, S3/2) to (0.5, S3/2), which matches top edge of (0,0). 
        # So (0, S3) is a neighbor. Distance = S3 = sqrt(3) ≈ 1.732. But side=1 hexes should be 2 apart?
        # Oh wait, I think I'm confusing myself. The "side length" of the hexagon is 1, and the distance between centers of adjacent hexes in a honeycomb is 2 (the diameter).
        # But sqrt(3) < 2, so they'd overlap! Let me recheck.
        # Hex at (0,0) with side 1: vertices at distance 1 from center.
        # Hex at (0, sqrt(3)) with side 1: bottom vertex at (0, sqrt(3)-1) = (0, 0.732).
        # Top vertex of (0,0) hex: the topmost point. For flat-top with vertices at angles 0,60,...:
        # (cos(60°), sin(60°)) = (0.5, 0.866) = (0.5, S3/2).
        # So top of (0,0) is at y = S3/2 ≈ 0.866.
        # Bottom of (0, S3) is at y = S3 - 1 ≈ 0.732.
        # Since 0.732 < 0.866, they OVERLAP! So (0, S3) is NOT a valid neighbor.
        # 
        # I think the issue is that for flat-top hexagons, the vertical neighbor distance should be 2*S3/2 + something...
        # Actually, I realize my hex_verts function creates hexagons with vertices at angles 0,60,120,180,240,300 degrees.
        # This is a "pointy-left-right" or "flat-top" hexagon.
        # Width (horizontal extent) = 2 (from x=-1 to x=1)
        # Height (vertical extent) = sqrt(3) (from y=-sqrt(3)/2 to y=sqrt(3)/2)
        # 
        # For two such hexes to be adjacent sharing a horizontal edge:
        # The top edge of lower hex is at y = sqrt(3)/2, from x=-0.5 to x=0.5.
        # The bottom edge of upper hex should be at the same location.
        # Upper hex center at (0, y_c): its bottom edge is at y_c - sqrt(3)/2.
        # So y_c - sqrt(3)/2 = sqrt(3)/2 => y_c = sqrt(3).
        # But then the upper hex's bottom vertex... wait, the bottom edge IS from (-0.5, y_c - sqrt(3)/2) to (0.5, y_c - sqrt(3)/2).
        # So they share the edge. Distance between centers = sqrt(3) ≈ 1.732.
        # But do they overlap? The lower hex's topmost points are the edge endpoints at (±0.5, sqrt(3)/2).
        # The upper hex's bottommost points are the same edge endpoints.
        # So they just touch along that edge. No overlap. 
        # 
        # So the correct neighbor distances for flat-top hex side 1:
        # - Vertical (sharing horizontal edge): sqrt(3)
        # - Diagonal (sharing slanted edge): 2
        # Let me verify diagonal: hex at (0,0) and hex at (1.5, sqrt(3)/2).
        # Distance = sqrt(1.5^2 + (sqrt(3)/2)^2) = sqrt(2.25 + 0.75) = sqrt(3) ≈ 1.732.
        # Hmm, that's also sqrt(3). Let me check if they share an edge.
        # Hex at (0,0): upper-right edge from (1,0) to (0.5, sqrt(3)/2).
        # Hex at (1.5, sqrt(3)/2): its lower-left edge... vertices:
        # (1.5+1, sqrt(3)/2) = (2.5, sqrt(3)/2)
        # (1.5+0.5, sqrt(3)/2+sqrt(3)/2) = (2, sqrt(3))
        # (1.5-0.5, sqrt(3)/2+sqrt(3)/2) = (1, sqrt(3))
        # (1.5-1, sqrt(3)/2) = (0.5, sqrt(3)/2)
        # (1.5-0.5, sqrt(3)/2-sqrt(3)/2) = (1, 0)
        # (1.5+0.5, sqrt(3)/2-sqrt(3)/2) = (2, 0)
        # So edges: (2.5,s3/2)-(2,s3), (2,s3)-(1,s3), (1,s3)-(0.5,s3/2), (0.5,s3/2)-(1,0), (1,0)-(2,0), (2,0)-(2.5,s3/2)
        # The edge (0.5,s3/2)-(1,0) is shared with hex at (0,0)'s edge (1,0)-(0.5,s3/2). Yes!
        # So all 6 neighbors are at distance sqrt(3).
        
        # So for flat-top hex side 1, the honeycomb lattice has spacing sqrt(3) between adjacent centers.
        # The 6 neighbor directions from (0,0):
        # (0, S3), (S3/2*sqrt(3)/S3*1.5, ...) -- let me just list them:
        # (0, S3), (1.5, S3/2), (1.5, -S3/2), (0, -S3), (-1.5, -S3/2), (-1.5, S3/2)
        
        neighbor_dirs = np.array([
            [0, S3], [1.5, S3/2], [1.5, -S3/2],
            [0, -S3], [-1.5, -S3/2], [-1.5, S3/2]
        ])
        
        # Generate positions using BFS from origin
        pos_set = {(0, 0)}
        frontier = [(0, 0)]
        for _ in range(5):  # 5 layers deep
            new_frontier = []
            for (q, r) in frontier:
                x = q * 1.5
                y = r * S3 + (S3/2 if q % 2 != 0 else 0)
                # Actually this coordinate system is getting confusing. Let me just use the neighbor offsets directly.
                pass
            
            # Let me use a different approach: just enumerate a grid of positions
            break
        
        # Simpler: just generate positions on the hex lattice directly
        # Using the fact that positions are at:
        # (1.5*i, S3*j + S3/2*(i%2)) for integers i,j -- this isn't quite right either.
        # Let me just use the neighbor vectors and do BFS properly.
        
        pos_set = set()
        pos_set.add((0.0, 0.0))
        queue = [(0.0, 0.0)]
        for _ in range(6):
            new_queue = []
            for (px, py) in queue:
                for dx, dy in neighbor_dirs:
                    nx, ny = px + dx, py + dy
                    key = (round(nx, 6), round(ny, 6))
                    if key not in pos_set:
                        pos_set.add(key)
                        new_queue.append((nx, ny))
            queue = new_queue
        
        return [(x, y) for x, y in pos_set]

    # Precompute all valid honeycomb positions
    all_positions = make_honeycomb_positions()
    print(f"Generated {len(all_positions)} honeycomb positions", flush=True)

    # Find the best subset of 12 positions that forms a compact cluster
    # Strategy: try all combinations is too expensive. Instead, use greedy + local search.
    
    def cluster_cost(positions_subset):
        """Cost: bounding hexagon side length for these positions."""
        pts = np.array(positions_subset)
        # Center at centroid
        centroid = pts.mean(axis=0)
        pts_c = pts - centroid
        # Compute outer side for various angles
        min_side = float('inf')
        for oa in range(0, 60, 5):
            side = compute_outer_side(pts_c, float(oa))
            if side < min_side:
                min_side = side
        return min_side

    # Greedy: start with center, add closest positions
    def greedy_cluster(n_positions=12):
        center = (0.0, 0.0)
        selected = [center]
        remaining = [p for p in all_positions if p != center]
        
        for _ in range(n_positions - 1):
            best_pos = None
            best_cost = float('inf')
            for p in remaining:
                trial = selected + [p]
                cost = cluster_cost(trial)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = p
            if best_pos is not None:
                selected.append(best_pos)
                remaining.remove(best_pos)
            else:
                break
        return selected

    # Also try some predefined compact shapes
    def predefined_clusters():
        """Several known compact arrangements of 12 hexagons."""
        clusters = []
        
        # 3x4 rectangle in hex grid
        # Positions: rows of 3 or 4
        # Let me define them using the neighbor structure
        # Center at (0,0), then build out
        
        # Hexagonal number arrangement: 1 + 6 + 5 (partial second ring) = 12
        # First ring: 6 neighbors
        ring1 = [(0, S3), (1.5, S3/2), (1.5, -S3/2), (0, -S3), (-1.5, -S3/2), (-1.5, S3/2)]
        
        # Second ring positions (distance 2 from center, or neighbors of ring1 not in ring1)
        ring2 = []
        for (px, py) in ring1:
            for dx, dy in neighbor_dirs:
                nx, ny = px + dx, py + dy
                key = (round(nx, 6), round(ny, 6))
                if key not in [(0.0, 0.0)] and key not in [(round(a,6), round(b,6)) for a,b in ring1]:
                    if key not in ring2:
                        ring2.append(key)
        
        # 1 center + 6 ring1 + 5 from ring2
        # Try all 5-subsets of ring2
        best_combo = None
        best_cost = float('inf')
        for combo in itertools.combinations(range(len(ring2)), 5):
            subset = [(0.0, 0.0)] + ring1 + [ring2[i] for i in combo]
            cost = cluster_cost(subset)
            if cost < best_cost:
                best_cost = cost
                best_combo = combo
        
        if best_combo is not None:
            clusters.append([(0.0, 0.0)] + ring1 + [ring2[i] for i in best_combo])
        
        # 4x3 arrangement: two rows of 4, two rows of... let me just do 3 rows of 4
        # In hex grid, a "rectangle" of 3x4:
        # Row 0: (0,0), (1.5, S3/2), (3, S3), (4.5, 1.5*S3)
        # Row 1: (0, S3), (1.5, 1.5*S3), (3, 2*S3), (4.5, 2.5*S3)
        # Row 2: (0, 2*S3), (1.5, 2.5*S3), (3, 3*S3), (4.5, 3.5*S3)
        # That's 12 positions but very elongated.
        
        # Better: 2 rows of 6? Or a more compact shape.
        # Let me try: center + 6 + 5 from ring2 is probably the most compact.
        
        # Also try: 7 in a hexagonal shape (1+6) + 5 more
        # Or: a 4-4-4 arrangement
        # Row of 4: (0,0), (1.5, S3/2), (3, S3), (4.5, 1.5*S3)
        # Next row offset: (0.75, S3*0.75)... this is getting complicated.
        
        # Let me just use the greedy approach as well
        return clusters

    # Generate candidate clusters
    candidates = []
    
    # Predefined compact clusters
    pred = predefined_clusters()
    candidates.extend(pred)
    
    # Greedy cluster
    greedy = greedy_cluster(12)
    candidates.append(greedy)
    
    # Also try: 2 centers + 10 around
    def two_center_cluster():
        c1, c2 = (0.0, 0.0), (S3, 0.0)  # Adjacent
        # Actually let me use (0,0) and (1.5, S3/2) which are neighbors
        c1, c2 = (0.0, 0.0), (1.5, S3/2)
        # Find 10 more positions that minimize bounding hex
        # Use greedy from these two
        selected = [c1, c2]
        remaining = [p for p in all_positions if p not in [c1, c2]]
        for _ in range(10):
            best_pos = None
            best_cost = float('inf')
            for p in remaining[:50]:  # Limit search
                trial = selected + [p]
                cost = cluster_cost(trial)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = p
            if best_pos is not None:
                selected.append(best_pos)
                remaining.remove(best_pos)
            else:
                break
        return selected
    
    candidates.append(two_center_cluster())
    
    # Evaluate all candidates
    best_cluster = None
    best_cluster_cost = float('inf')
    for cluster in candidates:
        if len(cluster) == 12:
            cost = cluster_cost(cluster)
            if cost < best_cluster_cost:
                best_cluster_cost = cost
                best_cluster = cluster
    
    if best_cluster is None:
        # Fallback: use greedy
        best_cluster = greedy_cluster(12)
        best_cluster_cost = cluster_cost(best_cluster)
    
    print(f"Best cluster cost: {best_cluster_cost:.4f}", flush=True)

    # Now optimize: cluster rotation + outer hexagon orientation
    # The cluster is rigid; we only rotate it and choose outer hex orientation.
    # Outer hex center = cluster centroid (by symmetry, this is optimal).
    
    def objective_2d(params):
        """params: [cluster_rotation_deg, outer_rotation_deg]"""
        if time.time() > deadline - 5:
            return 1e10
        
        cr = params[0]  # cluster rotation
        orr = params[1]  # outer hex rotation
        
        # Rotate cluster positions
        theta = np.radians(cr)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        pts = np.array(best_cluster)
        # Center at origin
        centroid = pts.mean(axis=0)
        pts_c = pts - centroid
        # Rotate
        pts_rot = np.column_stack([
            pts_c[:, 0] * cos_t - pts_c[:, 1] * sin_t,
            pts_c[:, 0] * sin_t + pts_c[:, 1] * cos_t
        ])
        
        # Compute outer side
        side = compute_outer_side(pts_rot, float(orr))
        return side

    # Multi-start Nelder-Mead
    best_result = None
    best_val = float('inf')
    
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 20:
            break
        if i == 0:
            p0 = np.array([0.0, 0.0])
        elif i == 1:
            p0 = np.array([30.0, 0.0])
        elif i == 2:
            p0 = np.array([0.0, 30.0])
        elif i == 3:
            p0 = np.array([15.0, 15.0])
        elif i == 4:
            p0 = np.array([45.0, 45.0])
        else:
            p0 = np.array([np.random.uniform(-90, 90), np.random.uniform(-90, 90)])
        
        try:
            res = minimize(objective_2d, p0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Also try differential evolution for global search
    if time.time() < deadline - 40:
        try:
            de_res = differential_evolution(objective_2d, [(-90, 90), (-90, 90)],
                                            maxiter=50, popsize=20, tol=0.01,
                                            seed=42, polish=False, update_interval=10)
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_result = de_res.x.copy()
        except Exception:
            pass

    if best_result is None:
        best_result = np.array([0.0, 0.0])
        best_val = objective_2d(best_result)

    # Final polish
    if time.time() < deadline - 10:
        try:
            res2 = minimize(objective_2d, best_result, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x.copy()
        except Exception:
            pass

    # Construct the final answer
    cr = best_result[0]  # cluster rotation
    orr = best_result[1]  # outer hex rotation
    
    theta = np.radians(cr)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    pts = np.array(best_cluster)
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    pts_rot = np.column_stack([
        pts_c[:, 0] * cos_t - pts_c[:, 1] * sin_t,
        pts_c[:, 0] * sin_t + pts_c[:, 1] * cos_t
    ])
    
    # Inner hexagons: all at angle 0 (same orientation as cluster)
    inner = [(float(pts_rot[i, 0]), float(pts_rot[i, 1]), 0.0) for i in range(12)]
    
    # Outer hexagon: centered at origin, side = best_val, rotation = orr
    oc = (0.0, 0.0)
    os = float(best_val)
    oa = float(orr)
    
    # Validate: check all inner hexes are inside outer and disjoint
    verts_list = [hex_verts(x, y, 1.0, a) for x, y, a in inner]
    
    # Check inside outer
    all_v = np.vstack(verts_list)
    if not all_points_in_hex(all_v, oc[0], oc[1], os, oa):
        # Expand outer side
        while not all_points_in_hex(all_v, oc[0], oc[1], os, oa) and os < 20:
            os += 0.01
    
    # Check disjoint
    if not pairwise_disjoint(verts_list):
        # Fallback: spread out positions
        print("WARNING: cluster not disjoint, expanding", flush=True)
        # Scale positions
        scale = 1.0
        while not pairwise_disjoint([hex_verts(x*scale, y*scale, 1.0, a) for x, y, a in inner]) and scale < 5:
            scale += 0.05
        inner = [(x*scale, y*scale, a) for x, y, a in inner]
        os = compute_outer_side(np.vstack([hex_verts(x, y, 1.0, a) for x, y, a in inner]), oa)
        while not all_points_in_hex(np.vstack([hex_verts(x, y, 1.0, a) for x, y, a in inner]),
                                     oc[0], oc[1], os, oa) and os < 20:
            os += 0.01

    return inner, oc, os, oa
# EVOLVE-BLOCK-END
