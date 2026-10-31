import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute unit hex vertices (flat-top: vertices at 0, 60, 120, 180, 240, 300)
    # Using "pointy-top" relative to x-axis for standard math orientation
    # Vertices at angles 0, 60, ...
    base_verts = [(math.cos(math.radians(60*i)), math.sin(math.radians(60*i))) for i in range(6)]

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        return [(cx + S*(vx*ca - vy*sa), cy + S*(vx*sa + vy*ca)) for vx, vy in base_verts]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def segments_intersect(p1, p2, p3, p4):
        d1 = cross(p3, p4, p1); d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3); d4 = cross(p1, p2, p4)
        if ((d1>0 and d2<0) or (d1<0 and d2>0)) and ((d3>0 and d4<0) or (d3<0 and d4>0)):
            return True
        return False

    def point_in_polygon(p, poly):
        x, y = p
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]; xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_disjoint(va, vb):
        # Fast check: bounding boxes
        min_xa = min(v[0] for v in va); max_xa = max(v[0] for v in va)
        min_ya = min(v[1] for v in va); max_ya = max(v[1] for v in va)
        min_xb = min(v[0] for v in vb); max_xb = max(v[0] for v in vb)
        min_yb = min(v[1] for v in vb); max_yb = max(v[1] for v in vb)
        if max_xa < min_xb or max_xb < min_xa or max_ya < min_yb or max_yb < min_ya:
            return True
        
        for v in va:
            if point_in_polygon(v, vb): return False
        for v in vb:
            if point_in_polygon(v, va): return False
        for i in range(6):
            if segments_intersect(va[i], va[(i+1)%6], vb[i], vb[(i+1)%6]):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        inner_verts = [transform_verts(ix, iy, 1.0, ia) for ix, iy, ia in inner]
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60*i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0]-ocx)*nx + (v[1]-ocy)*ny
                    if proj > max_proj: max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S: max_S = S_needed
        return max_S

    def all_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = [transform_verts(ix, iy, 1.0, ia) for ix, iy, ia in inner]
        # Check containment
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        # Check disjointness
        for i in range(len(inner_verts)):
            for j in range(i+1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def update_best(inner, ocx, ocy, S, oangle):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        if S < best_S and all_fit(inner, ocx, ocy, S, oangle):
            best_S = S
            best_inner = [list(x) for x in inner]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True
            return True
        return False

    # --- STRATEGY: Exhaustive Search over 11-Polyhexes ---
    # 1. Generate all unique connected clusters of 11 unit hexagons on a triangular lattice.
    # 2. For each unique shape, compute the minimal enclosing regular hexagon size.
    # 3. Select the shape with the smallest enclosing hexagon.
    # 4. Perform local refinement on the best shape to account for potential non-lattice optimizations 
    #    (though for convex containers and convex polygons, lattice packing is usually optimal or near-optimal).

    # Lattice generation
    # Axial coordinates (q, r). 
    # Pixel coordinates for pointy-top hexagons (side length 1):
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # Distance between centers of touching hexagons is sqrt(3).
    # Wait, standard pointy-top hex grid:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # Here size=1.
    
    sqrt3 = math.sqrt(3)
    
    # BFS to generate all polyhexes of size N
    def axial_to_pixel(q, r):
        return (sqrt3 * (q + r/2.0), 1.5 * r)
        
    def pixel_to_axial(x, y):
        # Inverse of above
        r = y / 1.5
        q = x / sqrt3 - r/2.0
        return (round(q), round(r))

    # Neighbors in axial coordinates
    neighbors = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

    # We will generate all sets of 11 connected cells.
    # To avoid duplicates, we use a canonical form (sorted tuple of axial coords, normalized by translation).
    # Since N=11 is small, the number of polyhexes is 29,126. This is very manageable.
    
    all_polyhexes = set()
    
    # Start with a single cell at (0,0)
    current_set = {(0,0)}
    
    # Recursive growth
    def grow(current, size):
        if size == N:
            # Canonicalize: translate so min-q is 0, then min-r is 0? 
            # Better: translate so the "lexicographically smallest" cell is at (0,0)
            # Or just sort and subtract min
            coords = sorted(current)
            min_q = min(c[0] for c in coords)
            min_r = min(c[1] for c in coords)
            # This simple translation doesn't handle rotation/reflection symmetry fully,
            # but since we just need to find the *best* shape, checking duplicates is less critical
            # than ensuring we find the optimal one. 
            # However, checking 29k shapes is fast. We can just store the sorted tuple.
            # To reduce duplicates significantly, we can normalize translation.
            norm_coords = tuple((q - min_q, r - min_r) for q, r in coords)
            all_polyhexes.add(norm_coords)
            return
            
        # Add a new cell adjacent to any existing cell
        added = set()
        for cell in current:
            for n in neighbors:
                new_cell = (cell[0] + n[0], cell[1] + n[1])
                if new_cell not in current:
                    new_set = current | {new_cell}
                    # To avoid generating the same set multiple times in different orders,
                    # we can enforce an ordering? No, BFS naturally handles connectivity.
                    # But we might add (1,0) then (0,1) vs (0,1) then (1,0).
                    # The canonicalization at the end handles the final set identity.
                    # To prune search space: only add cells that are "new" in a specific sense?
                    # Actually, for N=11, the branching factor is manageable.
                    grow(new_set, size + 1)

    # Optimization: The recursive grow above might be slow in Python due to set copying and deep recursion.
    # Let's use an iterative approach or a more efficient pruning.
    # Given N=11, the number of nodes in the search tree is roughly sum_{k=1}^{11} (number of k-polyhexes).
    # 1-poly: 1
    # 2-poly: 1
    # 3-poly: 2
    # 4-poly: 4
    # 5-poly: 8
    # 6-poly: 15
    # 7-poly: 28
    # 8-poly: 52
    # 9-poly: 94
    # 10-poly: 168
    # 11-poly: 291
    # Wait, the number of *free* polyhexes is small, but the number of *fixed* (lattice-specific) polyhexes is larger.
    # The recursive growth generates fixed polyhexes.
    # Total nodes is manageable.
    
    # Let's implement a safer, iterative BFS with a queue.
    # State: frozenset of cells.
    # To avoid re-generating the same set, we check if the canonical form is already in a visited set?
    # No, we need to explore all. But we can prune if the canonical form of the *prefix* is already seen?
    # Actually, for small N, just running the recursion is fine if we add a time check.
    
    # Let's restart the generation with a time guard.
    all_polyhexes = set()
    
    def grow_iterative(start_set):
        queue = [start_set]
        while queue:
            current = queue.pop()
            if len(current) > N:
                continue
            if len(current) == N:
                coords = sorted(current)
                min_q = min(c[0] for c in coords)
                min_r = min(c[1] for c in coords)
                norm_coords = tuple((q - min_q, r - min_r) for q, r in coords)
                all_polyhexes.add(norm_coords)
                continue
                
            if time.time() > deadline - 20: # Leave 20s for evaluation and refinement
                return

            # Generate neighbors
            for cell in current:
                for n in neighbors:
                    new_cell = (cell[0] + n[0], cell[1] + n[1])
                    if new_cell not in current:
                        new_set = current | {new_cell}
                        # Pruning: Only add if the new cell is "lexicographically greater" than all existing?
                        # This ensures we only generate each set once (in a canonical order of addition).
                        # This is a standard trick for polyomino/polyhex generation.
                        # Let's sort the cells. If new_cell > max(current), we add it.
                        # Wait, this restricts the shape to be "growing" in one direction?
                        # No, it just orders the addition. 
                        # Let's check: If we have {(0,0), (1,0)}, max is (1,0).
                        # Can we add (0,1)? (0,1) < (1,0) in lex order? 
                        # Lex order: (0,1) < (1,0) is True. So we would NOT add (0,1) if we require new > max.
                        # This would miss shapes that extend "backwards".
                        # Correct pruning: 
                        # We maintain the set in a canonical order. 
                        # Actually, the standard method is:
                        # 1. Keep a list of cells.
                        # 2. When adding a new cell, ensure it is not "equivalent" to a previous addition step.
                        # Simpler: Just use the set `all_polyhexes` at the end to deduplicate.
                        # For N=11, the number of intermediate states is small enough that deduplication at the end is sufficient,
                        # provided we don't blow up the search tree.
                        # The search tree size for fixed polyhexes of size 11 is bounded by ~10^5 nodes.
                        # Python can handle this in < 1 second.
                        
                        queue.append(new_set)

    grow_iterative(frozenset([(0,0)]))
    
    # Evaluate all generated polyhexes
    best_shape_coords = None
    min_S_shape = float('inf')
    
    for shape in all_polyhexes:
        if time.time() > deadline - 10:
            break
            
        # Convert axial to pixel
        pixel_coords = [axial_to_pixel(q, r) for q, r in shape]
        
        # Center the shape
        cx = sum(p[0] for p in pixel_coords) / N
        cy = sum(p[1] for p in pixel_coords) / N
        centered_coords = [(p[0]-cx, p[1]-cy) for p in pixel_coords]
        
        inner = [[x, y, 0.0] for x, y in centered_coords]
        
        # Compute minimal enclosing hexagon
        S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
        
        if S < min_S_shape:
            min_S_shape = S
            best_shape_coords = centered_coords

    if best_shape_coords is not None:
        best_inner = [[x, y, 0.0] for x, y in best_shape_coords]
        best_S = min_S_shape
        best_center = (0.0, 0.0)
        best_angle = 0.0
        # Verify fit
        if all_fit(best_inner, 0.0, 0.0, best_S, 0.0):
            best_valid = True
        else:
            # If numerical issues caused a fail, try slightly larger S
            best_S *= 1.001
            if all_fit(best_inner, 0.0, 0.0, best_S, 0.0):
                best_valid = True

    # Local refinement on the best shape found
    if best_valid and best_inner is not None:
        def local_refine(inner, seed):
            nonlocal best_S, best_inner, best_center, best_angle, best_valid
            rng = random.Random(seed)
            cur = [list(x) for x in inner]
            cx = sum(p[0] for p in cur) / N
            cy = sum(p[1] for p in cur) / N
            cur = [[p[0]-cx, p[1]-cy, p[2]] for p in cur]
            cur_S = compute_outer_hex(cur, 0.0, 0.0, 0.0)
            
            for it in range(5000):
                if time.time() > deadline - 5:
                    break
                old = [list(x) for x in cur]
                idx = rng.randrange(N)
                # Small perturbation
                cur[idx][0] += rng.gauss(0, 0.005)
                cur[idx][1] += rng.gauss(0, 0.005)
                
                new_S = compute_outer_hex(cur, 0.0, 0.0, 0.0)
                if new_S < cur_S and all_fit(cur, 0.0, 0.0, new_S, 0.0):
                    cur_S = new_S
                    if new_S < best_S:
                        best_S = new_S
                        best_inner = [list(x) for x in cur]
                        best_center = (0.0, 0.0)
                        best_angle = 0.0
                        best_valid = True
                else:
                    cur = old

        local_refine(best_inner, 42)

    # Fallback if exhaustive search failed or found nothing
    if not best_valid:
        # Use the previous "flower" heuristic as fallback
        dist = math.sqrt(3)
        candidate_1 = []
        # 1 center
        candidate_1.append((0,0))
        # 6 neighbors
        for i in range(6):
            a = math.radians(60*i)
            candidate_1.append((dist*math.cos(a), dist*math.sin(a)))
        # 4 next-neighbors (corners of the hexagon of neighbors)
        # These are at distance 2*dist from center? No, distance sqrt(3)*sqrt(3) = 3?
        # Axial (1,1), (1,-1), (-1,1), (-1,-1) are not neighbors of (0,0).
        # Axial neighbors of (0,0): (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # Next neighbors: (2,0), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,0), (-1,1), (0,2), (1,2), (2,1)
        # Let's just pick 4 that are far apart.
        # (2,0), (0,2), (-2,0), (0,-2) ?
        # Axial (2,0) -> pixel (2*sqrt3, 0)
        # Axial (0,2) -> pixel (sqrt3, 3)
        # Let's just use the 11 closest lattice points to origin.
        lattice_points = []
        for q in range(-4, 5):
            for r in range(-4, 5):
                x, y = axial_to_pixel(q, r)
                lattice_points.append((x, y, q, r))
        
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        candidate_1 = [lattice_points[i][:2] for i in range(11)]
        
        cx = sum(p[0] for p in candidate_1) / N
        cy = sum(p[1] for p in candidate_1) / N
        inner = [[p[0]-cx, p[1]-cy, 0.0] for p in candidate_1]
        S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
        if all_fit(inner, 0.0, 0.0, S, 0.0):
            best_S = S
            best_inner = inner
            best_center = (0.0, 0.0)
            best_angle = 0.0
            best_valid = True

    if best_valid and best_inner is not None:
        return best_inner, best_center, best_S, best_angle

    # Absolute fallback
    sp = math.sqrt(3)
    inner = [[(k - N//2)*sp, 0.0, 0.0] for k in range(N)]
    S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
    for _ in range(200):
        if all_fit(inner, 0.0, 0.0, S, 0.0):
            return inner, (0.0, 0.0), S, 0.0
        inner = [[p[0]*1.05, p[1]*1.05, p[2]] for p in inner]
        S = compute_outer_hex(inner, 0.0, 0.0, S, 0.0)
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
