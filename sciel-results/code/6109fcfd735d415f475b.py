"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (pointy-top orientation at 0 deg? No, standard math angle)."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _outer_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _project_range(verts, axis):
    """Min and max projection of polygon vertices onto a unit axis."""
    dots = verts @ axis
    return dots.min(), dots.max()

def _sat_overlap_depth(v1, v2, tol=1e-9):
    """Returns the minimum penetration depth if polygons overlap, else 0.
    Uses SAT to find the minimum separation distance (negative if overlapping).
    """
    min_overlap = float('inf')
    found_overlap = False
    
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            r1min, r1max = _project_range(v1, normal)
            r2min, r2max = _project_range(v2, normal)
            
            # Check separation
            if r1max < r2min + tol or r2max < r1min + tol:
                # Separated on this axis
                # Calculate distance of separation
                if r1max < r2min + tol:
                    sep = r2min - r1max
                else:
                    sep = r1min - r2max
                # If sep > 0, they are separated. We only care if they are NOT separated on ANY axis.
                # Wait, SAT says they are separated if there is ANY axis where they are separated.
                # So if we find a separation, we return 0 (no overlap).
                return 0.0
            
            # If not separated, calculate overlap on this axis
            overlap = min(r1max, r2max) - max(r1min, r2min)
            if overlap < min_overlap:
                min_overlap = overlap
            found_overlap = True
            
    if found_overlap:
        return min_overlap
    return 0.0

def _point_in_convex(px, py, poly, tol=1e-9):
    """Check if point is inside a convex polygon (given in CCW order)."""
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        ex = poly[j, 0] - poly[i, 0]
        ey = poly[j, 1] - poly[i, 1]
        px_rel = px - poly[i, 0]
        py_rel = py - poly[i, 1]
        cross = ex * py_rel - ey * px_rel
        if cross < -tol:
            return False
    return True

def _distance_to_boundary_point(px, py, outer_verts):
    """Calculate distance from point to boundary of outer hexagon.
    Returns positive distance if outside, 0 if on boundary, negative if inside?
    Actually, for penalty we want 0 if inside, and distance if outside.
    Let's compute min distance to edges. If point is inside, this distance is not the violation.
    We need to check if it's inside first.
    """
    # Check if inside
    if _point_in_convex(px, py, outer_verts, tol=1e-6):
        return 0.0
    
    # If outside, find min distance to any edge segment
    min_dist = float('inf')
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i+1) % n]
        
        v = p2 - p1
        w = np.array([px, py]) - p1
        
        c1 = np.dot(w, v)
        if c1 <= 0:
            dist = np.linalg.norm(w)
        else:
            c2 = np.dot(v, v)
            if c2 <= c1:
                dist = np.linalg.norm(w - v)
            else:
                t = c1 / c2
                dist = np.linalg.norm(w - t * v)
        min_dist = min(min_dist, dist)
    return min_dist

def _penalty_function(params, outer_side):
    """Calculate total penalty for a configuration.
    params: array of shape (N, 3) -> [x, y, angle_deg]
    outer_side: side length of container
    """
    penalty = 0.0
    
    # 1. Boundary violation penalty
    # We check if all vertices of each inner hex are inside the outer hex.
    # If not, we add the distance of the violating vertex.
    outer_verts = _outer_vertices(0.0, 0.0, outer_side, 0.0)
    
    for i in range(N):
        x, y, a = params[i]
        verts = _hex_vertices(x, y, a, 1.0)
        for v in verts:
            d = _distance_to_boundary_point(v[0], v[1], outer_verts)
            if d > 0:
                penalty += d * d # Quadratic penalty for smoothness
    
    # 2. Overlap penalty
    # We check all pairs. If they overlap, we add the penetration depth.
    verts_list = []
    for i in range(N):
        x, y, a = params[i]
        verts_list.append(_hex_vertices(x, y, a, 1.0))
        
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(verts_list[i], verts_list[j], tol=1e-6)
            if depth > 0:
                penalty += depth * depth # Quadratic penalty

    return penalty

def _generate_initial_topologies():
    """Generate a list of initial parameter arrays (N, 3) for different topologies."""
    topologies = []
    
    # 1. Ring Topology: 1 in center, 11 in a ring? Or 6+6?
    # Let's try 1 center, 6 around, 5 more?
    # Or just 12 in a ring.
    # Radius for 12 hexagons side 1 in a ring.
    # Distance between adjacent centers in ring must be >= 2*inradius = sqrt(3) ~ 1.732.
    # Chord length L = 2 * R_ring * sin(pi/12).
    # R_ring = L / (2 * sin(pi/12)) = 1.732 / (2 * 0.2588) = 1.732 / 0.5176 ~ 3.34.
    # Let's place 12 in a ring of radius 3.0 (tighter, will push out)
    ring_params = np.zeros((N, 3))
    for i in range(N):
        angle = 2 * math.pi * i / N
        x = 3.0 * math.cos(angle)
        y = 3.0 * math.sin(angle)
        # Orient hexagon to face out? Or just 0.
        ring_params[i] = [x, y, 0.0]
    topologies.append(ring_params)
    
    # 2. Hexagonal Cluster (Center + 6 + 5?)
    # Standard hex packing: 1 center, 6 neighbors. Total 7.
    # Add 5 more on the outside?
    # Center at (0,0).
    # Neighbors at distance sqrt(3) ~ 1.732.
    # Let's create a base hex cluster of 7.
    params = np.zeros((N, 3))
    params[0] = [0, 0, 0]
    for i in range(1, 7):
        angle = 2 * math.pi * (i-1) / 6
        x = 1.732 * math.cos(angle)
        y = 1.732 * math.sin(angle)
        params[i] = [x, y, 0.0]
    # Add 5 more. Maybe extend the hexagon.
    # Place them further out.
    for i in range(7, 12):
        angle = 2 * math.pi * (i-7) / 5 + 0.2 # offset
        x = 3.0 * math.cos(angle)
        y = 3.0 * math.sin(angle)
        params[i] = [x, y, 0.0]
    topologies.append(params)
    
    # 3. Rectangular Grid 3x4
    # Hexagons in a grid.
    # Width of hex = 2, Height = sqrt(3).
    # Horizontal spacing = 2. Vertical spacing = 1.5 * sqrt(3) ~ 2.598?
    # For pointy top, horizontal is 1.5, vertical is 2?
    # Let's use flat-top (angle 0).
    # Width = 2, Height = sqrt(3).
    # Horizontal neighbors: dx = 2.
    # Vertical neighbors: dy = sqrt(3), dx = 1.
    # Let's just place them in a grid with spacing 2.1 to allow rotation/adjustment.
    params = np.zeros((N, 3))
    idx = 0
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * 2.1
            y = (r - 1.0) * 1.9
            params[idx] = [x, y, 0.0]
            idx += 1
    topologies.append(params)
    
    # 4. Random
    params = np.random.rand(N, 3) * 6.0 - 3.0
    params[:, 2] *= 60.0 # Angle 0-60
    topologies.append(params)
    
    return topologies

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_placements = None
    best_side = float('inf')
    
    # We will try to find a feasible packing for a given side length.
    # We start with a reasonable guess and refine.
    # The theoretical lower bound is sqrt(6) ~ 2.45.
    # A good upper bound is around 4.0 - 5.0.
    
    # Strategy:
    # 1. Generate topologies.
    # 2. For each topology, optimize the positions to minimize penalty for a fixed outer side.
    # 3. If penalty is ~0, we have a feasible packing.
    # 4. We can then try to reduce the outer side? 
    #    Actually, the score is the outer side length.
    #    So we want to find the *minimum* outer side length that allows a feasible packing.
    #    However, optimizing for a fixed side and checking feasibility is easier.
    #    We can binary search on the side length, but optimization is expensive.
    #    Alternative: Optimize the packing for a *variable* outer side? No, outer side is the score.
    #    Let's assume we can find a packing for side S. Then we try S-0.1.
    
    # Let's try a few target side lengths, starting from a likely minimum.
    # If we can pack 12 unit hexagons, the area is 12 * (3*sqrt(3)/2) ~ 31.18.
    # Area of outer hex side S is (3*sqrt(3)/2) * S^2.
    # S^2 >= 31.18 / (3*sqrt(3)/2) = 12.
    # S >= sqrt(12) ~ 3.46.
    # So minimum S is at least 3.46.
    # Let's try S = 3.5, 3.6, 3.8, 4.0.
    
    target_sides = [3.5, 3.6, 3.7, 3.8, 4.0, 4.5]
    
    # We will iterate through topologies and sides.
    # To save time, we will try to find *any* feasible packing for the smallest possible side.
    # We'll sort sides ascending.
    
    topologies = _generate_initial_topologies()
    
    # Shuffle topologies to ensure variety
    np.random.shuffle(topologies)
    
    # We'll run optimization for each (side, topology) pair.
    # But we want the best (lowest side).
    # So we should try the smallest sides first.
    
    for side in target_sides:
        if time.time() > deadline - 10.0:
            break
            
        for topo in topologies:
            if time.time() > deadline - 5.0:
                break
                
            # Optimize this topology for this side
            # Use Nelder-Mead
            # Initial params: topo
            # Objective: _penalty_function(params, side)
            
            try:
                res = minimize(
                    _penalty_function, 
                    topo, 
                    args=(side,), 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
                )
            except Exception as e:
                continue
                
            if res.fun < 1e-3: # Feasible
                # We found a packing for this side.
                # Since we are iterating sides ascending, this is the best so far.
                # We can stop if we find a feasible one for the smallest side?
                # Yes, if we find feasible for 3.5, that's better than 4.0.
                # But maybe 3.4 is possible? We didn't try 3.4.
                # Let's add 3.4 to the list if we succeed at 3.5?
                # For now, let's just record the best.
                
                if side < best_side:
                    best_side = side
                    best_placements = res.x.copy()
                    
                # If we found a feasible packing for the *smallest* tried side, 
                # we might want to try even smaller?
                # But we are limited by time.
                # Let's break out of topology loop if we found feasible for current side,
                # to try the next (larger) side? No, we want the smallest.
                # So if we found feasible for 3.5, we don't need to try other topologies for 3.5?
                # We might find a *better* packing (less penalty) but it's already feasible.
                # The score is just the side length.
                # So any feasible packing for side S gives score S.
                # We want the minimum S.
                # So if we find feasible for 3.5, we should try to see if 3.4 is possible.
                # But we are iterating 3.5, 3.6...
                # So we should check 3.4 first.
                
                # Let's restart the side loop with a smaller range?
                # Or just break and let the outer loop handle it?
                # Actually, if we find feasible for 3.5, we should try 3.4, 3.45 etc.
                # Let's break the topology loop and try to refine the side range.
                
                break
                
        # If we found a feasible packing for this side, we can try to go lower?
        # We'll just continue to the next side in the list, but the list is ascending.
        # So if we found 3.5 feasible, we still try 3.6? That's worse.
        # We should only try sides if we haven't found a feasible one yet, 
        # OR if we want to verify.
        # Actually, if we found feasible for 3.5, we should stop trying larger sides.
        # And we should try smaller sides.
        
        if best_placements is not None and best_side == side:
            # We found a feasible packing for 'side'.
            # Let's try to find a smaller one.
            # We'll do a quick binary search downwards?
            # Or just try a few smaller values.
            new_sides = [side - 0.1, side - 0.05, side - 0.02]
            for ns in new_sides:
                if ns < 3.4: # Don't go too low
                    continue
                if time.time() > deadline - 5.0:
                    break
                # Try to optimize for this new side using the best topology found?
                # Or just random?
                # Let's use the best topology found so far as a start?
                # We don't track which topology worked.
                # Let's just try random or the first topology.
                for topo in topologies[:2]:
                    try:
                        res = minimize(
                            _penalty_function, 
                            topo, 
                            args=(ns,), 
                            method='Nelder-Mead', 
                            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
                        )
                    except Exception:
                        continue
                    if res.fun < 1e-3:
                        if ns < best_side:
                            best_side = ns
                            best_placements = res.x.copy()
                        break
                if best_side < side:
                    break
            
            # If we improved, we should continue trying to go lower?
            # But time is limited.
            break # Stop trying larger sides
            
    # If no feasible packing found, return a large side with a fallback placement
    if best_placements is None:
        # Fallback: place in a line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_placements, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
