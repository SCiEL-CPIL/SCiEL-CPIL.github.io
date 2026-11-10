import time
import numpy as np
from itertools import combinations

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 590.0

    # --- Geometry Helpers ---

    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return np.array([x, y])

    def hex_vertices(cx, cy, s, a_deg):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

    def edge_normals(V):
        n = np.empty((6, 2))
        for i in range(6):
            e = V[(i + 1) % 6] - V[i]
            L = np.hypot(e[0], e[1])
            # Normal pointing outward (assuming CCW vertices)
            # For a standard hexagon, edge i goes from v_i to v_{i+1}.
            # Outward normal is (dy, -dx) / L if vertices are CCW? 
            # Let's check: v0=(1,0), v1=(0.5, 0.866). e=(-0.5, 0.866). 
            # Normal (0.866, 0.5) points out. (dy, -dx) = (0.866, 0.5). Correct.
            n[i] = np.array([e[1], -e[0]]) / L
        return n

    def min_separation(A, B):
        """
        Returns the minimum gap between two convex polygons A and B.
        Gap > 0 means separated. Gap <= 0 means touching or intersecting.
        We check all edge normals of A and B.
        """
        nA = edge_normals(A)
        nB = edge_normals(B)
        best = -np.inf
        for n in nA:
            loA, hiA = (A @ n).min(), (A @ n).max()
            loB, hiB = (B @ n).min(), (B @ n).max()
            sep = max(loB - hiA, loA - hiB)
            if sep > best:
                best = sep
        for n in nB:
            loA, hiA = (A @ n).min(), (A @ n).max()
            loB, hiB = (B @ n).min(), (B @ n).max()
            sep = max(loB - hiA, loA - hiB)
            if sep > best:
                best = sep
        return best

    def min_enclosing_hexagon_side(points):
        """
        Finds the minimum side length of a regular hexagon enclosing the set of points.
        Returns (side_length, center_x, center_y, angle_deg).
        
        For a fixed angle phi, the minimal side length is max(widths) / sqrt(3).
        The center is the midpoint of the projections.
        We minimize over phi in [0, 60].
        """
        def side_for_phi(phi_deg):
            phi_rad = np.deg2rad(phi_deg)
            # Normals of the hexagon edges
            normals = np.column_stack((
                np.cos(phi_rad + np.deg2rad(60.0) * np.arange(6)), 
                np.sin(phi_rad + np.deg2rad(60.0) * np.arange(6))
            ))
            proj = points @ normals.T # (N, 6)
            widths = proj.max(axis=0) - proj.min(axis=0)
            return widths.max() / np.sqrt(3.0)
        
        # Coarse grid search
        phis = np.linspace(0, 60, 121)
        sides = np.array([side_for_phi(p) for p in phis])
        min_idx = np.argmin(sides)
        phi_min = phis[min_idx]
        
        # Refine with golden section search
        def golden_section_minimize(f, a, b, tol=1e-6):
            gr = (np.sqrt(5) - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = f(c)
            fd = f(d)
            while abs(b - a) > tol:
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = f(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = f(d)
            return (a + b) / 2

        # Determine a small window around phi_min
        step = 60.0 / 120.0
        a = max(0, phi_min - step)
        b = min(60, phi_min + step)
        
        if b - a > 1e-6:
            phi_opt = golden_section_minimize(side_for_phi, a, b)
        else:
            phi_opt = phi_min
            
        s_opt = side_for_phi(phi_opt)
        
        # Calculate center
        phi_rad = np.deg2rad(phi_opt)
        normals = np.column_stack((
            np.cos(phi_rad + np.deg2rad(60.0) * np.arange(6)), 
            np.sin(phi_rad + np.deg2rad(60.0) * np.arange(6))
        ))
        proj = points @ normals.T
        # Center projection is midpoint of min/max for each normal
        # This system is overdetermined (6 eqs, 2 unknowns). Use least squares.
        center_proj = (proj.max(axis=0) + proj.min(axis=0)) / 2.0
        # normals is (6, 2). We want [cx, cy] such that normals @ [cx, cy] ~ center_proj
        # Actually, proj = points @ normals.T. 
        # p . n_k = x*n_kx + y*n_ky.
        # So we solve normals @ [cx, cy]^T = center_proj.
        cx, cy = np.linalg.lstsq(normals, center_proj, rcond=None)[0]
        
        return s_opt, cx, cy, phi_opt

    def is_valid_packing(axial_coords):
        """
        Checks if the set of axial coordinates forms a valid packing of unit hexagons.
        On a hex lattice, unit hexagons are non-overlapping if and only if they are not adjacent.
        Adjacency in axial coords: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1).
        """
        coord_set = set(axial_coords)
        for q, r in axial_coords:
            for dq, dr in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
                if (q+dq, r+dr) in coord_set:
                    return False
        return True

    # --- Combinatorial Search ---
    
    # We generate all connected sets of 11 hexagons on the hex lattice.
    # To ensure we cover "compact" shapes, we restrict the bounding box.
    # Max distance from centroid is roughly sqrt(11)/2 * something.
    # A 11-hexagon cluster can fit in a 4x4 area.
    
    # We use a recursive growth algorithm.
    # To avoid duplicates, we use a canonical form:
    # Translate the set so that the lexicographically smallest (q, r) is (0, 0).
    # However, this doesn't handle rotations/reflections.
    # For packing, rotations/reflections of the same shape yield the same enclosing hexagon size.
    # We can just generate all "fixed" polyhexes (translations distinct) and canonicalize by sorting coordinates.
    # But we must ensure we don't miss shapes that don't contain (0,0) in their canonical form?
    # No, every shape has a unique canonical form (min lex point at 0,0).
    # So we generate all connected sets containing (0,0), canonicalize them, and deduplicate.
    
    best_side = np.inf
    best_inner = None
    best_outer = None
    
    seen_canonicals = set()
    
    # Directions for growth
    dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    def grow(current_set, remaining):
        nonlocal best_side, best_inner, best_outer, seen_canonicals
        
        if time.time() > deadline - 10.0:
            return False # Signal timeout
            
        if remaining == 0:
            # Canonicalize: sort and tuple
            # We don't strictly need to translate to 0,0 for deduplication if we just want unique geometries,
            # but translating ensures we don't count translations as different.
            # Since we start from (0,0) and grow, all shapes contain (0,0) in their "growth history",
            # but the final set might not have (0,0) as the min-lex point.
            # Example: grow (0,0)->(1,0). Set {(0,0), (1,0)}. Min is (0,0).
            # Example: grow (0,0)->(-1,0). Set {(0,0), (-1,0)}. Min is (-1,0).
            # Canonical form: translate so min is (0,0).
            min_q = min(q for q, r in current_set)
            min_r = min(r for q, r in current_set)
            # Note: min_q and min_r might come from different points.
            # Proper canonicalization for free polyhexes is complex (requires checking all rotations).
            # For this problem, we can just use the set of relative coordinates from the centroid or min-lex point.
            # Let's use min-lex point (q,r) as origin.
            origin_q, origin_r = min(current_set, key=lambda x: (x[0], x[1]))
            rel_coords = tuple(sorted((q - origin_q, r - origin_r) for q, r in current_set))
            
            if rel_coords in seen_canonicals:
                return True
            
            seen_canonicals.add(rel_coords)
            
            # Check validity (non-adjacency)
            if not is_valid_packing(list(current_set)):
                return True
            
            # Convert to cartesian
            cart = np.array([axial_to_cartesian(q, r) for q, r in current_set])
            
            # Find minimal enclosing hexagon
            s, cx, cy, phi = min_enclosing_hexagon_side(cart)
            
            if s < best_side:
                best_side = s
                # Store result
                # Shift inner hexagons so outer center is at (0,0)
                inner_rel = cart - np.array([cx, cy])
                best_inner = np.column_stack((inner_rel[:, 0], inner_rel[:, 1], np.full(11, 0.0)))
                best_outer = np.array([0.0, 0.0, phi])
            
            return True
        
        # Find neighbors
        neighbors = set()
        for q, r in current_set:
            for dq, dr in dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in current_set:
                    # Pruning: keep it compact
                    # Bounding box check
                    # If we grow too far, we can prune.
                    # Max span in q or r is about 4.
                    qs = [qq for qq, rr in current_set | {(nq, nr)}]
                    rs = [rr for qq, rr in current_set | {(nq, nr)}]
                    if max(qs) - min(qs) > 5 or max(rs) - min(rs) > 5:
                        continue
                    neighbors.add((nq, nr))
        
        # To reduce symmetry duplicates, we could restrict growth directions,
        # but with canonicalization, we can just try all.
        # To speed up, we can order neighbors.
        for nb in neighbors:
            new_set = current_set | {nb}
            ok = grow(new_set, remaining - 1)
            if not ok:
                return False
        
        return True

    # Start search
    # We start with a single hexagon at (0,0) and need to add 10 more.
    # Note: This generates all connected sets of size 11 that can be reached by adding one hex at a time.
    # This covers all "polyhexes".
    
    initial_set = {(0, 0)}
    grow(initial_set, 10)
    
    if best_inner is not None:
        return best_inner, best_outer, best_side

    # Fallback: Return a valid, though not optimal, configuration
    # 11 hexagons in a row? No, that's long.
    # Let's return the "star" shape or similar known good shape.
    # Center (0,0), and 10 neighbors? No, neighbors are adjacent, so invalid.
    # We need non-adjacent.
    # A valid packing:
    # (0,0), (2,0), (4,0)... no.
    # Let's use a specific valid set:
    # 3x3 block minus corners?
    # (0,0), (2,0), (4,0)
    # (1,1), (3,1)
    # (0,2), (2,2), (4,2)
    # (1,3), (3,3)
    # (0,4), (2,4), (4,4) -> 13 points.
    
    # Simple valid set:
    # (0,0), (2,0), (4,0)
    # (1,1), (3,1)
    # (0,2), (2,2), (4,2)
    # (1,3), (3,3)
    # (0,4) -> 11 points.
    
    coords = [
        (0,0), (2,0), (4,0),
        (1,1), (3,1),
        (0,2), (2,2), (4,2),
        (1,3), (3,3),
        (0,4)
    ]
    
    # Check validity
    if not is_valid_packing(coords):
        # Fallback to a definitely valid one: spread out
        coords = [(i*2, 0) for i in range(11)]
        
    cart = np.array([axial_to_cartesian(q, r) for q, r in coords])
    s, cx, cy, phi = min_enclosing_hexagon_side(cart)
    
    inner_rel = cart - np.array([cx, cy])
    inner = np.column_stack((inner_rel[:, 0], inner_rel[:, 1], np.full(11, 0.0)))
    outer = np.array([0.0, 0.0, phi])
    
    return inner, outer, s

# EVOLVE-BLOCK-END
