import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Generate candidate inner configurations
    # We look for 12 disjoint unit hexagons packed tightly.
    # A standard hexagonal packing of 19 hexagons (radius 2 cluster) has side length 3 for the convex hull in hex coordinates,
    # but we only need 12.
    # Let's generate the 19 centers of a radius-2 hex cluster in axial coordinates (q, r).
    # Axial to Cartesian for unit hexagons (flat-top or pointy-top? 
    # Standard math convention for regular hexagons often uses:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # This corresponds to pointy-top hexagons if we consider y as vertical.
    # Let's stick to this coordinate system.
    
    centers_axial = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                centers_axial.append((q, r))
    
    # We have 19 centers. We need to select 12 that form a compact shape.
    # The most compact subset of 12 from a hex lattice is usually the central 1 + 6 neighbors + 5 of the next ring.
    # Or simply the 12 hexagons that form a hexagon of "radius" 1 (which is 7) plus 5 more?
    # Actually, a hexagon of side length 2 (in hex units) contains 1+6+12 = 19 hexagons.
    # A hexagon of side length 1 (in hex units) contains 7 hexagons.
    # We need 12.
    # Let's take the 7 central ones and add 5 from the outer ring to minimize the bounding box.
    # Due to symmetry, we can just take the first 12 from a sorted list by distance from center, 
    # but we must ensure they are connected and compact.
    # The 7 central ones are at dist 0 and 1. The next 12 are at dist 2.
    # If we take all 7 central + 5 outer, we have 12.
    # To minimize the enclosing hexagon, we should balance the 5 outer ones around the center.
    
    # Let's define the 7 central ones (dist <= 1 in hex metric)
    central = [c for c in centers_axial if max(abs(c[0]), abs(c[1]), abs(c[0]+c[1])) <= 1]
    # There are 7 such centers.
    
    # Outer ring (dist = 2)
    outer = [c for c in centers_axial if max(abs(c[0]), abs(c[1]), abs(c[0]+c[1])) == 2]
    # There are 12 such centers.
    
    # We need to pick 5 from the outer ring.
    # To maintain symmetry and minimize the bounding box, we should pick them evenly.
    # The outer ring has 12 positions. Picking 5 breaks 6-fold symmetry.
    # However, the bounding box of the union of 7 central + 5 outer will be determined by the furthest points.
    # Let's try picking 5 that are as spread out as possible.
    # Indices 0, 2, 4, 6, 8? Or 0, 3, 6, 9, 12?
    # Let's just generate a few candidates for the set of 12 centers.
    
    candidates = []
    
    # Candidate 1: 7 central + 5 specific outer
    # Sort outer by angle
    outer_sorted = sorted(outer, key=lambda p: math.atan2(1.5*p[1], math.sqrt(3)*(p[0] + p[1]/2.0)))
    
    # Try different subsets of 5 from the 12 outer
    # We can't try all C(12,5)=792 subsets in a complex optimization, but we can try a few symmetric ones.
    # Subset A: Indices 0, 2, 4, 6, 8 (every other one starting 0)
    subset_a = [outer_sorted[i] for i in [0, 2, 4, 6, 8]]
    candidates.append(central + subset_a)
    
    # Subset B: Indices 1, 3, 5, 7, 9
    subset_b = [outer_sorted[i] for i in [1, 3, 5, 7, 9]]
    candidates.append(central + subset_b)
    
    # Subset C: Indices 0, 3, 6, 9, 12 (0,3,6,9,0? no, 0,3,6,9, and maybe 11 or 1?)
    # Let's try 0, 3, 6, 9, 11
    subset_c = [outer_sorted[i] for i in [0, 3, 6, 9, 11]]
    candidates.append(central + subset_c)
    
    # Candidate 2: Maybe a different shape?
    # What if we use a 4x3 rectangular grid of hexagons?
    # In a hex grid, a "rectangle" is tricky.
    # Let's stick to the radial clusters.
    
    best_result = None
    best_score = float('inf')
    
    for centers in candidates:
        if time.time() > deadline:
            break
            
        # Convert axial to Cartesian
        pts = []
        for q, r in centers:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        pts = np.array(pts)
        
        # The inner hexagons are unit hexagons.
        # Their vertices extend from the center by distance 1 (circumradius).
        # For a unit hexagon (side 1), the circumradius is 1.
        # The vertices of an inner hexagon at (cx, cy) with angle theta are:
        # (cx + cos(theta + k*pi/3), cy + sin(theta + k*pi/3)) for k=0..5
        
        # We need to find the minimal side length S of a regular hexagon (centered at 0,0, rotated by phi)
        # that contains all vertices of all inner hexagons.
        # Note: The problem allows rotating the inner hexagons. 
        # However, for a dense packing, aligning all inner hexagons with the same orientation (or the lattice orientation) is usually optimal or near-optimal for the bounding box.
        # The lattice orientation above corresponds to pointy-top hexagons (vertices at 90, 150, etc? No, x=sqrt3... y=1.5... 
        # Let's check vertex angles for the lattice hexagons.
        # If we use the standard axial->cartesian, the hexagons are "pointy-top" relative to the y-axis?
        # Center (0,0). Neighbors at (sqrt3, 0), (0, 1.5), (-sqrt3, 0)...
        # Wait, distance to (sqrt3, 0) is sqrt3. Distance to (0, 1.5) is 1.5.
        # For unit hexagons to touch, the distance between centers must be sqrt(3) (for flat-top) or 1.5 (for pointy-top)?
        # Side length 1.
        # Flat-top: width = 2, height = sqrt(3). Distance to horizontal neighbor = 2? No, 2 * (3/4 * 2) = 1.5?
        # Let's restart the geometry carefully.
        
        # Regular Hexagon side s=1.
        # Circumradius R=1. Inradius r=s*sqrt(3)/2 = sqrt(3)/2.
        
        # Orientation 1: Flat-top (flat edge on top/bottom).
        # Vertices at angles 0, 60, 120, 180, 240, 300 degrees.
        # Width (x-extent) = 2. Height (y-extent) = sqrt(3).
        # Distance between centers of touching flat-top hexagons:
        # Horizontal: 2 * (3/4 * 2) ? No.
        # x-coord of vertices: 1, 0.5, -0.5, -1, -0.5, 0.5.
        # If we place two flat-top hexagons side-by-side horizontally, they touch at x=0.5 and x=-0.5? No.
        # The rightmost vertex is at x=1. The leftmost of the next one must be at x=1.
        # So center distance = 2.
        # Vertical distance? They nest.
        # This is getting complicated. Let's use the property that the bounding hexagon of a set of points P is determined by the maximum support function in 6 directions.
        
        # Let's assume the inner hexagons are all oriented the same way as the "lattice" hexagons.
        # And let's assume the outer hexagon is also aligned with the lattice or we optimize its rotation.
        
        # Actually, the simplest robust approach:
        # 1. Define the set of all vertices of the 12 inner hexagons.
        # 2. For a given outer hexagon orientation phi, the required side length S is determined by the maximum distance from the center to any vertex projected onto the 6 normals of the outer hexagon.
        #    Specifically, for a regular hexagon centered at origin with side S, the inradius (apothem) is S * sqrt(3) / 2.
        #    The condition for a point (x,y) to be inside is:
        #    |x * cos(phi) + y * sin(phi)| <= S * sqrt(3) / 2
        #    |x * cos(phi + pi/3) + y * sin(phi + pi/3)| <= S * sqrt(3) / 2
        #    ... for all 6 directions.
        #    So S = (2 / sqrt(3)) * max( |proj_k| ) for k=0..5.
        
        # We need to generate the vertices of the 12 inner hexagons.
        # Let's fix the orientation of the inner hexagons. 
        # The lattice generated by (sqrt(3)*(q+r/2), 1.5*r) corresponds to hexagons with vertices at:
        # If we assume the lattice hexagons are "pointy-top" (vertex at top), the vertices are at angles 90, 150, 210, 270, 330, 30.
        # Let's verify touching distance for pointy-top.
        # Vertex at 90 deg: (0, 1).
        # Neighbor at (0, 1.5) in axial?
        # If center is (0,0), top vertex is (0,1).
        # If neighbor center is (0, 1.5), its bottom vertex is (0, 1.5 - 1) = (0, 0.5).
        # They don't touch. Distance between centers 1.5.
        # For pointy-top, horizontal neighbor is at (sqrt(3), 0).
        # Right vertex of center 0: (sqrt(3)/2, 0.5).
        # Left vertex of center (sqrt(3), 0): (sqrt(3) - sqrt(3)/2, -0.5) = (sqrt(3)/2, -0.5).
        # They don't touch.
        
        # Okay, standard hex grid coordinates:
        # For pointy-top:
        # x = sqrt(3) * (q + r/2)
        # y = 3/2 * r
        # This grid has neighbors at distance sqrt(3) (horizontal) and 1.5 (vertical-ish).
        # Wait, distance between (0,0) and (sqrt(3), 0) is sqrt(3).
        # Distance between (0,0) and (0, 1.5) is 1.5.
        # For unit hexagons to touch, the distance must be 2 * inradius = sqrt(3) (for flat-top horizontal touch?)
        # Inradius of unit hex is sqrt(3)/2.
        # If flat-top, width is 2, height sqrt(3).
        # Horizontal neighbors touch if distance is 2? No, the "flat" side is vertical?
        # Flat-top: Flat edge on top. Vertices at 0, 60...
        # x-extent: -1 to 1.
        # If we place two flat-top hexagons side-by-side, the right edge of left is a vertical segment? No, it's a vertex at 0 deg (1,0) and 60 deg (0.5, sqrt(3)/2).
        # The rightmost point is (1,0).
        # The leftmost point of the next is (-1,0) + offset.
        # So offset is 2.
        # Vertical offset?
        
        # Let's just calculate the bounding box of the vertices numerically for the specific lattice I generated.
        # The lattice I generated: x = sqrt(3)*(q+r/2), y=1.5*r.
        # Let's assume the inner hexagons are oriented such that they tile this lattice perfectly.
        # This implies the inner hexagons are "pointy-top" relative to the y-axis?
        # Let's check the vertices of a hexagon that fits in this lattice.
        # If the lattice is defined by the centers of a tiling of pointy-top hexagons:
        # The vertices of a pointy-top hexagon centered at (0,0) are:
        # (0, 1), (-sqrt(3)/2, 0.5), (-sqrt(3)/2, -0.5), (0, -1), (sqrt(3)/2, -0.5), (sqrt(3)/2, 0.5).
        # Let's check if these tiles form the lattice.
        # Neighbor at (sqrt(3), 0).
        # Its leftmost vertices: (sqrt(3) - sqrt(3)/2, 0.5) = (sqrt(3)/2, 0.5) and (sqrt(3) - sqrt(3)/2, -0.5) = (sqrt(3)/2, -0.5).
        # These match the rightmost vertices of the center hexagon.
        # So YES, the lattice generated by x=sqrt(3)*(q+r/2), y=1.5*r corresponds to a tiling of **pointy-top** unit hexagons.
        
        # So, for each center (cx, cy), the vertices are:
        # (cx, cy+1)
        # (cx - sqrt(3)/2, cy + 0.5)
        # (cx - sqrt(3)/2, cy - 0.5)
        # (cx, cy - 1)
        # (cx + sqrt(3)/2, cy - 0.5)
        # (cx + sqrt(3)/2, cy + 0.5)
        
        # We can rotate these inner hexagons. Let's assume we keep them aligned with the lattice (angle 0) for simplicity first, 
        # or we can optimize the global rotation of the entire cluster.
        # The problem says "each inner hexagon may translate and rotate".
        # Rotating individual hexagons in a tight packing is hard to optimize. 
        # However, rotating the *entire cluster* of 12 hexagons (keeping their relative orientations fixed to each other and the lattice) 
        # is equivalent to rotating the set of vertices.
        # Since the outer hexagon is also regular, the optimal alignment is likely when the symmetry axes align.
        
        # Let's collect all vertices of the 12 inner hexagons.
        all_vertices = []
        for px, py in pts:
            # Pointy-top vertices
            all_vertices.append((px, py + 1.0))
            all_vertices.append((px - math.sqrt(3)/2, py + 0.5))
            all_vertices.append((px - math.sqrt(3)/2, py - 0.5))
            all_vertices.append((px, py - 1.0))
            all_vertices.append((px + math.sqrt(3)/2, py - 0.5))
            all_vertices.append((px + math.sqrt(3)/2, py + 0.5))
            
        V = np.array(all_vertices)
        
        # Now, find the minimal side length S of a regular hexagon (centered at 0,0) containing V.
        # We can rotate the outer hexagon by phi.
        # S(phi) = (2/sqrt(3)) * max( | V_k . n_i(phi) | ) for all vertices k and normals i.
        # Normals are at angles phi, phi+60, ... phi+300.
        
        # To minimize S, we should align the "flattest" direction of the cluster with the "flattest" direction of the hexagon?
        # Or align the "pointiest" with "pointiest"?
        # Let's sample phi from 0 to 60 degrees (due to symmetry of hexagon, 60 deg is enough, but cluster might have lower symmetry).
        # Actually, the cluster of 12 might have 6-fold symmetry if we chose the outer 5 carefully?
        # My subsets A, B, C broke 6-fold symmetry.
        # Let's check if a 6-fold symmetric subset of 12 exists in the 19-hex cluster.
        # 19 = 1 + 6 + 12.
        # If we take 1 + 6 + 5, we don't have 6-fold symmetry.
        # If we take 1 + 6 + 6 (from the outer ring), that's 13 hexagons. Too many.
        # If we take 1 + 6 + 5, we lose symmetry.
        # Is there another packing of 12?
        # A 2x3 rectangle of hexagons?
        # In a hex grid, a "rectangle" of 2x3 (2 rows, 3 cols) has 6 hexagons.
        # 4x3 = 12 hexagons.
        # Let's try a 4x3 rectangular block.
        
        # Generate 4x3 block in axial coords.
        # This is tricky in axial.
        # Let's stick to the radial cluster but try to find the best phi for each candidate.
        
        min_s_for_candidate = float('inf')
        best_phi = 0
        
        # Sample phi
        phis = np.linspace(0, math.pi/3, 361)
        for phi in phis:
            # Normals
            angles = phi + np.arange(6) * math.pi / 3
            cos_n = np.cos(angles)
            sin_n = np.sin(angles)
            
            # Project all vertices onto these normals
            # V is (N, 2). 
            # Projections: V @ [cos_n, sin_n]
            # We need max absolute projection for each normal, then max over normals.
            
            # Efficiently:
            # proj = V[:, 0] * cos_n[None, :] + V[:, 1] * sin_n[None, :]
            # This is (N, 6)
            # max_abs_proj = np.max(np.abs(proj), axis=0) # (6,)
            # current_s = (2 / math.sqrt(3)) * np.max(max_abs_proj)
            
            # To save time, we can just check the convex hull vertices, but N=12*6=72 is small.
            proj = V[:, 0] * cos_n[None, :] + V[:, 1] * sin_n[None, :]
            max_abs_proj = np.max(np.abs(proj), axis=0)
            current_s = (2.0 / math.sqrt(3.0)) * np.max(max_abs_proj)
            
            if current_s < min_s_for_candidate:
                min_s_for_candidate = current_s
                best_phi = phi
                
        if min_s_for_candidate < best_score:
            best_score = min_s_for_candidate
            best_result = (centers, best_phi)
            
    if best_result is None:
        # Fallback: Return the seed program's result or a valid default
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    centers, phi = best_result
    
    # Construct the inner list
    # The angle in the output is the rotation of the inner hexagon.
    # We assumed the inner hexagons are pointy-top (angle 90 deg? or 0 deg?).
    # My vertex calculation used:
    # (0, 1) -> angle 90 deg.
    # Standard "0 degrees" in many libraries might mean flat-top or pointy-top.
    # The task asks for "angle_degrees".
    # If I output 0.0, does the evaluator assume flat-top or pointy-top?
    # Usually, 0 degrees means the first vertex is at (1,0) (flat-top? no, vertex at 0).
    # If vertex at (1,0), it's flat-top? No, vertex at 0 deg means a point on the right.
    # Flat-top has flat edge on top. Vertex at 0, 60, 120...
    # Pointy-top has vertex at 90, 150...
    # If I calculated vertices for pointy-top (vertex at 90), I should report an angle that rotates the standard hexagon to match.
    # If the standard hexagon at angle 0 has a vertex at (1,0) (Flat-Top), then to get Pointy-Top (vertex at 90), I need to rotate by 90 degrees?
    # Or if standard is Pointy-Top at 0, then 0 is fine.
    
    # Let's assume the evaluator's "angle 0" corresponds to a hexagon with a vertex at (1,0) (Flat-Top).
    # My calculation used Pointy-Top vertices.
    # To align my calculated vertices with the evaluator's rendering, I need to specify the angle.
    # If I say angle=90, and the evaluator rotates a Flat-Top hexagon by 90, it becomes Pointy-Top.
    # So I should return 90.0 if my geometry was Pointy-Top.
    
    # Wait, the outer hexagon rotation is also part of the output.
    # I found `phi` which is the rotation of the OUTER hexagon's normals.
    # The output format is: inner rows (x, y, angle_degrees), outer center, outer side, outer angle.
    
    # Let's refine the output:
    # Inner hexagons: All have the same orientation.
    # If I used Pointy-Top geometry, and assume evaluator uses Flat-Top for angle=0, then inner angle = 90.
    # Outer hexagon: I found phi. 
    # The normals of the outer hexagon are at phi, phi+60...
    # A regular hexagon with "angle" theta usually has a vertex at theta? Or a flat edge?
    # If "angle" defines the orientation of the first vertex, then for the outer hexagon, 
    # if I want normals at phi, the vertices are at phi + 30, phi + 90... (for flat-top?)
    # This is ambiguous. 
    # However, the score is based on the side length. The evaluator likely computes the side length from the geometry.
    # If I provide the side length S and the angle, the evaluator checks if the inner hexagons fit.
    # If the evaluator's definition of angle is consistent, I just need to be consistent with my calculation.
    
    # Let's assume the evaluator uses the standard:
    # Hexagon at angle 0 has vertices at 0, 60, 120... (Flat-Top).
    # My inner hexagons are Pointy-Top. So I set inner angle to 90.
    # My outer hexagon normals are at phi.
    # For a Flat-Top hexagon, normals are at 30, 90, 150...
    # If I want normals at phi, I need to rotate the Flat-Top hexagon by (phi - 30).
    # So outer angle = phi - 30 degrees.
    
    inner_output = []
    for q, r in centers:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner_output.append([x, y, 90.0]) # Pointy-Top
        
    outer_center = [0.0, 0.0]
    outer_side = best_score
    outer_angle = math.degrees(phi) - 30.0
    
    return inner_output, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
