"""Packing 12 unit hexagons in a regular hexagon using a rotated lattice strategy."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # 1. Generate candidate lattice positions for the 12 unit hexagons.
    # We use a hexagonal lattice. The optimal arrangement for 12 items is often
    # a central hexagon surrounded by a ring, or a 4x3 rectangular block, etc.
    # However, the "honeycomb" cluster of 19 (radius 2) is the densest packing.
    # We select the 12 "closest" or "most central" positions from a larger grid
    # to ensure high density.
    
    # Axial coordinates for a hexagonal grid
    # We generate a large enough set of points and pick the best 12.
    # A good candidate set is the 19 points of the radius-2 hexagon cluster.
    # Let's generate axial coordinates (q, r) such that max(|q|, |r|, |q+r|) <= 2
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                candidates.append((q, r))
    
    # Sort candidates by distance from origin to pick the central 12
    # Distance in axial coords: 3/2 * |q + r| + 3/4 * (|q-r| + |q+r|) ... 
    # Simpler: convert to cartesian and sort by radius
    def axial_to_cartesian(q, r):
        # Flat-top hexagon, side length 1
        # x = 3/2 * q
        # y = sqrt(3) * (r + q/2)
        x = 1.5 * q
        y = math.sqrt(3) * (r + q / 2.0)
        return x, y

    cartesian_candidates = []
    for q, r in candidates:
        x, y = axial_to_cartesian(q, r)
        cartesian_candidates.append((x, y, q, r))
    
    # Sort by distance from center
    cartesian_candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Select the 12 most central points
    # Note: There are 19 points in the radius-2 cluster.
    # The 12 most central will form a dense cluster.
    selected_points = cartesian_candidates[:N]
    
    # Extract x, y
    centers = np.array([(p[0], p[1]) for p in selected_points])
    
    # 2. Determine the optimal rotation angle for the cluster.
    # The outer hexagon has 6-fold symmetry. The cluster of 12 might have lower symmetry.
    # We need to rotate the cluster relative to the outer hexagon to minimize the bounding box.
    # The "width" of the cluster in a direction theta is the max projection of (center + vertex) 
    # minus min projection.
    # However, we can simplify by noting that the outer hexagon side length L relates to the 
    # distance from center to the farthest point in the cluster, adjusted for the hexagon geometry.
    
    # For a regular hexagon of side L, the distance from center to vertex is L.
    # The distance from center to edge midpoint is L * sqrt(3)/2.
    # The containment condition is: for all points P in the union of inner hexagons,
    # the distance from P to the center must be <= L if P is near a vertex, 
    # or the projection onto the normal of the edge must be <= L * sqrt(3)/2.
    
    # Actually, the condition is:
    # max_{P in cluster} ( max_{k=0..5} |dot(P, u_k)| / (sqrt(3)/2) ) <= L
    # where u_k are the unit normals to the edges of the outer hexagon.
    # Wait, the support function of a regular hexagon of side L in direction u is:
    # h(u) = L * sqrt(3)/2 * max(1, |cos(theta)|) ... no.
    # Let's use the property that a point (x,y) is inside a regular hexagon of side L centered at origin
    # if max( |x|/L, |y|/(L*sqrt(3)/2), |(x/2 + y*sqrt(3)/2)|/L ) <= 1 ? No, that's for a specific orientation.
    
    # Let's define the outer hexagon with vertices at angles 0, 60, ..., 300.
    # The edges are at angles 30, 90, ..., 330 (normals).
    # The distance from center to edge is apothem A = L * sqrt(3) / 2.
    # A point (x,y) is inside if its projection on any of the 6 normal vectors is <= A.
    # The normals are at angles 30 + 60*k.
    # n_k = (cos(30+60k), sin(30+60k)).
    # Condition: dot(P, n_k) <= A for all k.
    # This is equivalent to: max_k |dot(P, n_k)| <= A.
    # So L = (2/sqrt(3)) * max_k |dot(P, n_k)|.
    
    # We need to find the rotation angle phi of the inner cluster such that
    # L(phi) = (2/sqrt(3)) * max_{P in Cluster, k=0..5} |dot(R_phi P, n_k)| is minimized.
    
    # The set of points P includes all vertices of all 12 unit hexagons.
    # Vertices of a unit hexagon centered at (cx, cy) with rotation alpha:
    # v_j = (cx, cy) + (cos(alpha + 60j), sin(alpha + 60j))
    
    # Since the inner hexagons can rotate, we should align them to maximize packing efficiency.
    # Usually, aligning them with the lattice (alpha=0) is good, but maybe rotating them
    # relative to the outer hexagon helps.
    # However, the problem allows each inner hexagon to rotate.
    # To minimize the bounding box, we should orient each inner hexagon to fit best.
    # But for a lattice packing, it's often optimal to keep them aligned with the lattice.
    # Let's assume all inner hexagons have the same rotation alpha_inner.
    # And the whole cluster is rotated by phi.
    # Total rotation of inner hexagon j is phi + alpha_inner.
    
    # Let's simplify: Assume inner hexagons are aligned with the cluster lattice (alpha_inner = 0).
    # We rotate the whole configuration by phi.
    # The vertices of the j-th hexagon are at R_phi (C_j + V_j), where V_j are unit hex vertices.
    
    # We want to minimize f(phi) = max_{j, v, k} |dot(R_phi (C_j + v), n_k)|
    
    # Let's sample phi from 0 to 60 degrees.
    # The function is periodic and symmetric.
    
    best_L = float('inf')
    best_phi = 0.0
    
    # Precompute unit hex vertices
    unit_vertices = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)
    ])
    
    # Precompute normals for outer hexagon
    normals = np.array([
        [math.cos(math.radians(30 + 60*k)), math.sin(math.radians(30 + 60*k))] for k in range(6)
    ])
    
    # Sample rotations
    # The symmetry of the 12-point cluster might be 6-fold or 3-fold or 2-fold.
    # Let's sample finely.
    num_samples = 360
    for i in range(num_samples):
        if time.time() > deadline:
            break
        phi = math.radians(i * 60.0 / num_samples)
        R_phi = np.array([
            [math.cos(phi), -math.sin(phi)],
            [math.sin(phi), math.cos(phi)]
        ])
        
        max_proj = 0.0
        for c in centers:
            # Rotate center
            rc = R_phi @ c
            # Rotate vertices
            rv = R_phi @ unit_vertices.T
            # Add center to vertices
            all_pts = (rc[:, np.newaxis] + rv).T
            # Project onto normals
            projs = np.abs(all_pts @ normals.T)
            current_max = np.max(projs)
            if current_max > max_proj:
                max_proj = current_max
            if max_proj > best_L * math.sqrt(3) / 2: # Early termination if already worse
                break
        
        L = (2.0 / math.sqrt(3.0)) * max_proj
        if L < best_L:
            best_L = L
            best_phi = phi
    
    # Refine the best phi
    if best_L < float('inf'):
        # Local search around best_phi
        step = 0.1
        for _ in range(10):
            if time.time() > deadline:
                break
            improved = False
            for delta in [-step, step]:
                test_phi = best_phi + math.radians(delta)
                R_phi = np.array([
                    [math.cos(test_phi), -math.sin(test_phi)],
                    [math.sin(test_phi), math.cos(test_phi)]
                ])
                max_proj = 0.0
                for c in centers:
                    rc = R_phi @ c
                    rv = R_phi @ unit_vertices.T
                    all_pts = (rc[:, np.newaxis] + rv).T
                    projs = np.abs(all_pts @ normals.T)
                    current_max = np.max(projs)
                    if current_max > max_proj:
                        max_proj = current_max
                test_L = (2.0 / math.sqrt(3.0)) * max_proj
                if test_L < best_L:
                    best_L = test_L
                    best_phi = test_phi
                    improved = True
            if not improved:
                step /= 2.0
                if step < 0.01:
                    break
    
    # Construct the final output
    # Inner hexagons: (x, y, angle_degrees)
    # The angle in the output is the rotation of the inner hexagon.
    # In our model, the inner hexagons are rotated by best_phi relative to the global frame.
    # The vertices were calculated as R_phi @ (C + V). This means the hexagon itself is rotated by best_phi.
    
    inner = []
    for c in centers:
        rc = np.array([math.cos(best_phi), -math.sin(best_phi), math.sin(best_phi), math.cos(best_phi)])
        # Wait, the center coordinates in the output should be the final coordinates in the global frame.
        # The outer hexagon is centered at (0,0) and rotated by 0 (standard orientation).
        # The inner hexagons are placed at R_phi @ C_j.
        x, y = R_phi @ c
        # The angle of the inner hexagon is best_phi (in degrees).
        angle_deg = math.degrees(best_phi)
        inner.append([x, y, angle_deg])
        
    outer_center = (0.0, 0.0)
    outer_side = best_L
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
