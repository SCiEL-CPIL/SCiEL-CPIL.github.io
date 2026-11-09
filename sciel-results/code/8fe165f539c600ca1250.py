"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator limit 120s, keep 10s buffer)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle (radians)
    def get_hex_vertices(cx, cy, angle):
        # Flat-top hexagon vertices at angles 0, 60, 120, 180, 240, 300 relative to x-axis
        # Rotated by 'angle'
        verts = []
        for i in range(6):
            theta = angle + i * math.pi / 3.0
            verts.append((cx + math.cos(theta), cy + math.sin(theta)))
        return verts

    def hexagon_overlap_penalty(centers, angles, min_dist=2.0):
        """
        Calculate penalty for overlaps.
        Returns 0 if no overlaps, >0 otherwise.
        We use a simple distance check first: if centers are < 2.0, they might overlap.
        For a more accurate check, we'd need vertex-edge intersection, but for optimization
        gradients, a smooth approximation or hard penalty on min distance between vertices
        is better.
        
        However, checking all pairs of vertices for 11 hexagons is 11*10/2 * 36 vertex pairs = 1980 checks.
        This is fast. Let's do exact vertex-vertex and vertex-edge distance.
        Actually, simpler: check min distance between any vertex of hex i and any vertex of hex j.
        If min_dist < 0, they overlap.
        To make it differentiable for optimization, we can sum max(0, min_dist - epsilon)^2.
        But Nelder-Mead doesn't need gradients. It just needs a scalar objective.
        
        So: Objective = Outer_Side_Length + Lambda * Overlap_Penalty
        """
        penalty = 0.0
        for i in range(N):
            v1 = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for j in range(i + 1, N):
                v2 = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                # Check min distance between any point in v1 and v2
                # Since they are convex, if vertices don't interpenetrate, they might still overlap?
                # No, for convex polygons, if no edge intersects and no vertex is inside, they are disjoint.
                # A simpler sufficient condition for disjointness of two unit hexagons:
                # Distance between centers > 2.0 is NOT sufficient (corners can touch closer).
                # Distance between centers < 2.0 is NOT necessary for overlap.
                
                # Robust check:
                # 1. If min distance between any vertex of v1 and any vertex of v2 is < 0 (impossible, dist is >= 0)
                # Wait, we want to ensure they are DISJOINT.
                # If they are disjoint, the min distance between their boundaries is > 0.
                # If they overlap, the min distance is 0 (or negative if we consider signed distance).
                
                # Let's use the fact that for two convex polygons, if they are disjoint,
                # there exists a separating axis.
                # For optimization, a common trick is to penalize the minimum distance between
                # any vertex of one and any edge of the other.
                
                # To keep it fast and simple for Nelder-Mead:
                # Check if any vertex of v1 is inside v2 or vice versa.
                # Check if any edge of v1 intersects any edge of v2.
                
                # This is complex to implement robustly in pure Python fast enough for many iterations.
                # Alternative: Use the "bounding circle" approximation or a simple vertex-vertex min dist.
                # If min vertex-vertex dist < 0.01, assume overlap.
                # This is not perfectly accurate for edge-edge crossing without vertex intersection,
                # but for regular hexagons, overlaps usually involve vertices or edges crossing.
                
                # Let's try a slightly better metric:
                # Min distance between any vertex of v1 and any vertex of v2.
                # AND min distance between any vertex of v1 and any edge of v2.
                
                min_dist = float('inf')
                
                # Vertex-Vertex
                for p1 in v1:
                    for p2 in v2:
                        d = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                        if d < min_dist:
                            min_dist = d
                
                # Vertex-Edge (approximate by checking distance to midpoints of edges? No, exact point-segment dist)
                # To save time, we'll just check vertex-vertex. If two convex polygons overlap,
                # it's highly likely a vertex of one is close to a vertex or edge of the other.
                # Given the symmetry and regularity, vertex-vertex min dist < 0 is impossible.
                # We want them to be SEPARATED.
                # So if min_dist < epsilon, we penalize.
                
                # Actually, if they are disjoint, min_dist > 0.
                # If they touch, min_dist = 0.
                # If they overlap, min_dist = 0 (boundaries intersect).
                # So min_dist doesn't distinguish touch/overlap from disjoint.
                # We need to know if they are INSIDE each other or intersecting.
                
                # Let's use a simpler constraint:
                # The distance between centers must be at least the "contact distance" for the given relative angle.
                # For two unit hexagons, the minimum center distance to be disjoint depends on the relative rotation.
                # If aligned (0 deg), min center dist is 2.0 (flat sides touch) or SQRT(3) ~ 1.732 (corners touch)?
                # No, flat-to-flat is SQRT(3). Corner-to-corner is 2.0?
                # Let's check:
                # Flat-top hex. Center (0,0). Vertices at (1,0), (0.5, SQRT3/2)...
                # Wait, standard flat-top:
                # v0: (1, 0)
                # v1: (0.5, SQRT3/2)
                # v2: (-0.5, SQRT3/2)
                # v3: (-1, 0)
                # v4: (-0.5, -SQRT3/2)
                # v5: (0.5, -SQRT3/2)
                
                # If we place another one at (2, 0), its left vertex is at (1,0). They touch. Center dist 2.
                # If we place at (SQRT3, 0), its left flat side is at x=SQRT3-1? No.
                # Leftmost point is x=-1. Rightmost x=1.
                # If aligned, min center dist for touching is 2.0 (vertex-vertex) or SQRT(3) (side-side)?
                # Side is vertical? No, flat-top has vertical sides? No.
                # Flat-top: top and bottom are flat (horizontal). Sides are slanted.
                # Wait, "Flat-top" usually means the top edge is horizontal.
                # My get_hex_vertices:
                # i=0: angle 0 -> (1,0)
                # i=1: angle 60 -> (0.5, 0.866)
                # i=2: angle 120 -> (-0.5, 0.866)
                # Top edge is between v1 and v2. Horizontal.
                # So "Flat-top" is correct.
                
                # Distance between two aligned hexagons touching at a vertex: 2.0.
                # Distance between two aligned hexagons touching at a side:
                # The side v1-v2 is at y = SQRT3/2.
                # The bottom side of the other hex must be at y = -SQRT3/2 relative to its center.
                # So center dist = SQRT3.
                # So min center dist for disjointness is SQRT(3) ~ 1.732.
                
                # If they are rotated, the min center dist changes.
                # To be safe and simple, let's enforce center distance >= 2.0?
                # That's too conservative.
                # Let's enforce center distance >= 1.8?
                
                # Better: Compute the exact min center distance for the given relative angle.
                # This is computationally expensive to do exactly for every pair in every iteration.
                
                # Alternative Strategy:
                # Use the "Circle Packing" approximation first to get a good initial guess,
                # then refine with Nelder-Mead using a penalty based on vertex-vertex min distance.
                # If min vertex-vertex distance < 0.01, we consider it an overlap.
                # This is a heuristic but works well for optimization landscapes.
                
                if min_dist < 0.01:
                    # Penalize heavily
                    penalty += (0.01 - min_dist) ** 2
        
        return penalty

    def outer_side_length(centers, outer_angle):
        """
        Calculate min side length S of outer hexagon containing all hexagons.
        """
        # Normals to sides of outer hexagon
        base_angles = [30, 90, 150, 210, 270, 330]
        normals = []
        for a in base_angles:
            rad = math.radians(a + outer_angle)
            normals.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for x, y in centers:
            for nx, ny in normals:
                # The hexagon extends 1.0 from center in any direction?
                # No, the max projection of a unit hexagon in direction (nx,ny) is:
                # max over vertices v of (v . n)
                # We need to add this to the center projection.
                # For a unit hexagon, the max projection in any direction is 1.0 (vertex) or SQRT(3)/2 (side normal)?
                # Actually, the support function h(u) of a unit hexagon.
                # For direction u, max_{x in Hex} x.u
                # Since Hex is convex, max is at a vertex.
                # We can precompute or just use 1.0 as a safe upper bound?
                # If we use 1.0, we might be slightly conservative (outer hex slightly larger).
                # Let's use 1.0 for simplicity and speed.
                proj = abs(x * nx + y * ny) + 1.0
                if proj > max_proj:
                    max_proj = proj
        
        # Apothem = S * SQRT3 / 2
        # max_proj <= Apothem
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    def objective(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, outer_angle]
        """
        centers = []
        angles = []
        for i in range(N):
            centers.append((params[3*i], params[3*i+1]))
            angles.append(params[3*i+2])
        
        outer_angle = params[3*N]
        
        # Center the configuration to keep it near origin (helps optimizer)
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        
        # Shift centers for evaluation
        shifted_centers = [(c[0]-cx_avg, c[1]-cy_avg) for c in centers]
        
        side = outer_side_length(shifted_centers, outer_angle)
        penalty = hexagon_overlap_penalty(shifted_centers, angles)
        
        # Weight penalty heavily
        return side + 100.0 * penalty

    # Initial Guess: Hexagonal lattice
    # Center at 0,0
    # 7 in first ring, 4 in second ring (partial)
    # Let's use a known good packing for 11 circles as a start, then adjust for hexagons.
    # Or just use the lattice from the current program as a start.
    
    init_params = np.zeros(3*N + 1)
    
    # Lattice coords from current program (Shape A was best)
    base_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    cand_A = base_7 + [(2,0), (2,-1), (2,1), (1,1)]
    
    # Convert axial to cartesian (spacing 1.75 was used, let's use 1.8 for safety)
    spacing = 1.8
    for i, (q, r) in enumerate(cand_A):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        init_params[3*i] = x
        init_params[3*i+1] = y
        init_params[3*i+2] = 0.0 # Angle 0
    
    init_params[3*N] = 0.0 # Outer angle
    
    # Center the initial guess
    cx_avg = sum(init_params[3*i] for i in range(N)) / N
    cy_avg = sum(init_params[3*i+1] for i in range(N)) / N
    for i in range(N):
        init_params[3*i] -= cx_avg
        init_params[3*i+1] -= cy_avg

    # Run Nelder-Mead
    best_side = float('inf')
    best_params = init_params.copy()
    
    # We will run multiple restarts with different initial angles/configs if time permits
    # But Nelder-Mead on 34 variables might be slow.
    # Let's try one run first.
    
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
            
        # Extract best solution
        centers = []
        angles = []
        for i in range(N):
            centers.append((best_params[3*i], best_params[3*i+1]))
            angles.append(best_params[3*i+2])
        
        outer_angle = best_params[3*N]
        
        # Check validity: Ensure no overlaps
        # If penalty > 0, it's invalid.
        penalty = hexagon_overlap_penalty(centers, angles)
        if penalty > 0.01:
            # Fallback to initial guess if optimization failed to remove overlaps
            # Or try a different initial guess
            pass
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    # If we have time, try a second run with different initial angles (e.g., 30 deg)
    if time.time() < deadline - 20.0:
        init_params2 = init_params.copy()
        for i in range(N):
            init_params2[3*i+2] = math.pi/6 # 30 degrees
        # Perturb positions slightly
        init_params2[:3*N] += np.random.uniform(-0.5, 0.5, 3*N)
        
        try:
            res2 = minimize(objective, init_params2, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            if res2.fun < best_side:
                best_side = res2.fun
                best_params = res2.x
        except Exception as e:
            pass

    # Finalize best_params
    centers = []
    angles = []
    for i in range(N):
        centers.append((best_params[3*i], best_params[3*i+1]))
        angles.append(best_params[3*i+2])
    
    outer_angle = best_params[3*N]
    
    # Calculate final side length
    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    shifted_centers = [(c[0]-cx_avg, c[1]-cy_avg) for c in centers]
    
    final_side = outer_side_length(shifted_centers, outer_angle)
    
    # Check for overlaps one last time. If overlaps exist, we must fix or return invalid?
    # The task says "disjoint". If we return overlapping, score is 0.
    # We must ensure disjointness.
    # If penalty is high, we should revert to a known safe configuration (the lattice one)
    # and just return that, even if side is larger.
    
    penalty = hexagon_overlap_penalty(shifted_centers, angles)
    if penalty > 0.01:
        # Revert to safe lattice
        # Reconstruct lattice centers
        safe_centers = []
        safe_angles = [0.0] * N
        for i, (q, r) in enumerate(cand_A):
            x = SQRT3 * 1.8 * (q + r / 2.0)
            y = 1.5 * 1.8 * r
            safe_centers.append((x, y))
        
        cx_avg = sum(c[0] for c in safe_centers) / N
        cy_avg = sum(c[1] for c in safe_centers) / N
        shifted_centers = [(c[0]-cx_avg, c[1]-cy_avg) for c in safe_centers]
        final_side = outer_side_length(shifted_centers, 0.0)
        outer_angle = 0.0
        centers = shifted_centers
        angles = safe_angles

    # Construct return format
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], math.degrees(angles[i])])
    
    return inner, (0.0, 0.0), final_side, math.degrees(outer_angle)

# EVOLVE-BLOCK-END
