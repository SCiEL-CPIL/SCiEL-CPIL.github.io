"""Improved program: Discrete topological search with continuous container optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (lists of (x,y))."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def generate_polyhexes(n_cells):
    """
    Generate all free polyhexes of size n_cells using BFS on axial coordinates.
    Returns a list of sets of (q, r) tuples.
    """
    def neighbors(q, r):
        return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]

    def canonicalize(cells):
        """
        Return the canonical form of a set of cells to identify free polyhexes.
        We rotate and translate to find the "smallest" representation.
        """
        # Translation to origin
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        # Actually, better to normalize relative to the first cell
        # Let's use a simpler canonicalization: rotate 0, 60, 120... and translate to min lexicographic
        best = None
        for rot in range(6):
            # Apply rotation
            rotated = set()
            for q, r in cells:
                # Rotation formulas for axial coordinates
                # 60 deg: (q, r) -> (-r, q+r)
                # We can just apply the transform repeatedly or use a lookup
                # Let's do it explicitly for clarity, though slow for large N, N=11 is small
                pass
            
            # Efficient rotation in axial:
            # 0: (q, r)
            # 1: (-r, q+r)
            # 2: (-q-r, q)
            # 3: (-q, -r)
            # 4: (r, -q-r)
            # 5: (q+r, -q)
            
            if rot == 0:
                rot_cells = set(cells)
            elif rot == 1:
                rot_cells = set({(-r, q+r) for q, r in cells})
            elif rot == 2:
                rot_cells = set({(-q-r, q) for q, r in cells})
            elif rot == 3:
                rot_cells = set({(-q, -r) for q, r in cells})
            elif rot == 4:
                rot_cells = set({(r, -q-r) for q, r in cells})
            else: # rot == 5
                rot_cells = set({(q+r, -q) for q, r in cells})
            
            # Translate to min lexicographic
            min_q = min(c[0] for c in rot_cells)
            min_r = min(c[1] for c in rot_cells)
            # Actually, we should find the translation that makes the sorted list of tuples minimal
            # Sort tuples
            sorted_rot = sorted(rot_cells)
            # Find translation to make first element (0,0)? No, that doesn't work for all.
            # Standard canonicalization: Translate so that the minimum q is 0, and for that q, min r is 0.
            # Then compare lexicographically.
            
            # Let's just generate all translations that map one cell to (0,0) and pick the min sorted tuple list
            candidates = []
            for anchor in rot_cells:
                trans = set({(q - anchor[0], r - anchor[1]) for q, r in rot_cells})
                candidates.append(sorted(trans))
            
            min_candidate = min(candidates)
            
            if best is None or min_candidate < best:
                best = min_candidate
                
        return tuple(best)

    # BFS
    all_polyhexes = set()
    queue = [frozenset([(0, 0)])]
    all_polyhexes.add(canonicalize([(0, 0)]))
    
    while queue:
        current = queue.pop(0)
        if len(current) == n_cells:
            continue # We want exactly n_cells
            
        # Try adding a neighbor
        added = False
        for cell in current:
            for nb in neighbors(cell[0], cell[1]):
                if nb not in current:
                    new_set = frozenset(current | {nb})
                    canon = canonicalize(new_set)
                    if canon not in all_polyhexes:
                        all_polyhexes.add(canon)
                        queue.append(new_set)
                        added = True
        
    # Convert back to lists of tuples
    # Note: canonicalize returns a tuple of tuples. We need to reconstruct a representative set.
    # For our purposes, we just need the relative positions.
    # The canonical form is the "shape". We can use the canonical form directly to generate coordinates.
    # However, canonicalize returns a tuple of (q,r). This IS a set of coordinates (normalized).
    # So we can just return the list of canonical forms.
    
    return [list(shape) for shape in all_polyhexes]

def optimize_container(inner, deadline):
    """
    For a fixed set of inner hexagons (positions and angles), optimize the outer hexagon.
    Returns (center, angle, side_length).
    """
    # Initial guess: centroid
    cx = sum(p[0] for p in inner) / len(inner)
    cy = sum(p[1] for p in inner) / len(inner)
    
    def obj(params):
        ocx, ocy, oa = params
        return compute_required_side_fast(inner, (ocx, ocy), oa)
    
    best_val = float('inf')
    best_p = None
    
    # Multi-start on angle
    for ia in range(0, 60, 5):
        if time.time() > deadline - 2:
            break
        p0 = np.array([cx, cy, float(ia)])
        try:
            res = minimize(obj, p0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 400})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x
        except:
            pass
            
    if best_p is None:
        return (cx, cy), 0.0, obj([cx, cy, 0.0])
        
    return (float(best_p[0]), float(best_p[1])), float(best_p[2]), float(best_val)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0 # 10s safety margin

    # 1. Generate topologies
    # 11-cell polyhexes. There are 170 free polyhexes of order 11.
    # Generating them might take a bit, but it's one-time cost.
    try:
        topologies = generate_polyhexes(N)
    except Exception as e:
        print(f"Topology generation failed: {e}")
        # Fallback to a simple linear chain
        topologies = [list(range(N))] # This is invalid for hex grid, need proper fallback
        # Let's use a known good shape: 3-4-4 or similar
        # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (0,2), (-2,1), (-2,-1)
        topologies = [[(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (0, 2), (-2, 1), (-2, -1)]]

    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0

    # 2. Iterate over topologies
    # We cannot test all 12^11 orientations.
    # Strategy:
    # A. Test all aligned (angle=0).
    # B. Test single flips (one hexagon rotated by 30, 60, etc.).
    # C. Test random subsets of flips.
    
    # Pre-compute axial to Cartesian
    def axial_to_cart(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return x, y

    # Orientation patterns to try
    # 0: All 0
    # 1: Random individual rotations
    # 2: Random global rotation of the whole cluster (handled by container optimization, but inner angles relative to each other matter)
    
    # We will try a few "modes" of orientation assignment for each topology.
    # Mode 0: All angles 0.
    # Mode 1: Each hexagon angle is 0 or 30 (random).
    # Mode 2: Each hexagon angle is 0, 30, 60 (random).
    
    num_topologies = len(topologies)
    if time.time() > deadline - 10:
        # Not enough time to even start properly, return a basic config
        pass

    for ti, topo in enumerate(topologies):
        if time.time() > deadline - 5:
            break
            
        # Convert topology to Cartesian coordinates
        base_coords = [axial_to_cart(q, r) for q, r in topo]
        
        # Center the topology at origin for symmetry
        cx = sum(c[0] for c in base_coords) / len(base_coords)
        cy = sum(c[1] for c in base_coords) / len(base_coords)
        centered_coords = [(x - cx, y - cy) for x, y in base_coords]
        
        # Try different orientation assignments
        # 1. All aligned
        angles_0 = [0.0] * N
        inner_0 = [[x, y, a] for (x, y), a in zip(centered_coords, angles_0)]
        
        # Check disjointness (should be true for valid polyhexes with unit spacing)
        if check_disjoint(inner_0, eps=1e-5):
            oc, oa, side = optimize_container(inner_0, deadline)
            if side < best_side:
                best_side = side
                best_inner = inner_0
                best_outer_center = oc
                best_outer_angle = oa
        
        # 2. Randomized individual orientations
        # We can't do full random search for all topologies if there are many.
        # Let's do a few random trials per topology.
        np.random.seed(ti) # Deterministic random per topology
        
        # Trial A: Each hexagon has angle 0 or 30
        angles_a = [0.0 if np.random.random() < 0.5 else 30.0 for _ in range(N)]
        inner_a = [[x, y, a] for (x, y), a in zip(centered_coords, angles_a)]
        if check_disjoint(inner_a, eps=1e-5):
            oc, oa, side = optimize_container(inner_a, deadline)
            if side < best_side:
                best_side = side
                best_inner = inner_a
                best_outer_center = oc
                best_outer_angle = oa
                
        # Trial B: Each hexagon has angle 0, 30, or 60
        angles_b = [float(np.random.choice([0, 30, 60])) for _ in range(N)]
        inner_b = [[x, y, a] for (x, y), a in zip(centered_coords, angles_b)]
        if check_disjoint(inner_b, eps=1e-5):
            oc, oa, side = optimize_container(inner_b, deadline)
            if side < best_side:
                best_side = side
                best_inner = inner_b
                best_outer_center = oc
                best_outer_angle = oa

    # 3. Final Polish: Local search on the best found configuration
    if best_inner is not None and time.time() < deadline - 10:
        # Convert best_inner to a parameter vector for local search
        # We will only search positions slightly, keeping angles fixed?
        # Or search angles slightly?
        # Let's do a simple local search on positions and angles to relax the packing.
        
        def obj_local(params):
            il = []
            for i in range(N):
                il.append([params[i*3], params[i*3+1], params[i*3+2]])
            ocx, ocy, oa = params[N*3], params[N*3+1], params[N*3+2]
            side = compute_required_side_fast(il, (ocx, ocy), oa)
            # Penalty for overlap
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, _ = il[i]
                    xj, yj, _ = il[j]
                    d = math.hypot(xi-xj, yi-yj)
                    if d < SQRT3 - 0.01:
                        pen += (SQRT3 - 0.01 - d)**2
            return side + 100.0 * pen

        p0 = np.zeros(N*3 + 3)
        for i, (x, y, a) in enumerate(best_inner):
            p0[i*3] = x
            p0[i*3+1] = y
            p0[i*3+2] = a
        p0[N*3] = best_outer_center[0]
        p0[N*3+1] = best_outer_center[1]
        p0[N*3+2] = best_outer_angle
        
        try:
            res = minimize(obj_local, p0, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 2000})
            if res.fun < best_side:
                # Verify validity
                final_inner = []
                for i in range(N):
                    final_inner.append([float(res.x[i*3]), float(res.x[i*3+1]), float(res.x[i*3+2])])
                final_oc = (float(res.x[N*3]), float(res.x[N*3+1]))
                final_oa = float(res.x[N*3+2])
                
                if check_disjoint(final_inner, eps=1e-5):
                    final_side = compute_required_side_fast(final_inner, final_oc, final_oa)
                    if final_side < best_side:
                        best_inner = final_inner
                        best_outer_center = final_oc
                        best_outer_angle = final_oa
                        best_side = final_side
        except Exception:
            pass

    # Fallback if nothing found
    if best_inner is None:
        # Use the first topology with all aligned
        topo = topologies[0]
        base_coords = [axial_to_cart(q, r) for q, r in topo]
        cx = sum(c[0] for c in base_coords) / len(base_coords)
        cy = sum(c[1] for c in base_coords) / len(base_coords)
        centered_coords = [(x - cx, y - cy) for x, y in base_coords]
        best_inner = [[x, y, 0.0] for x, y in centered_coords]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
