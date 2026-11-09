# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, permutations

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s kill

    def hex_vertices(cx, cy, s, a_deg):
        a = np.radians(a_deg)
        angles = a + np.radians(60 * np.arange(6))
        return np.array([[cx + s * np.cos(angles[i]), cy + s * np.sin(angles[i])] for i in range(6)])

    def is_valid(inner, outer, side):
        """Check validity: non-overlap and containment."""
        # Check finite
        if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and np.isfinite(side)):
            return False
        if side < np.sqrt(11):
            return False

        # Shift coordinates so outer center is at origin
        ocx, ocy, oa = outer
        o_verts = hex_vertices(ocx, ocy, side, oa)
        shifted_inner = inner.copy()
        shifted_inner[:, 0] -= ocx
        shifted_inner[:, 1] -= ocy

        # Check edge lengths of shifted inner hexagons are ~1
        for i in range(11):
            x, y, ang = shifted_inner[i]
            verts = hex_vertices(x, y, 1.0, ang)
            for j in range(6):
                dx = verts[(j + 1) % 6, 0] - verts[j, 0]
                dy = verts[(j + 1) % 6, 1] - verts[j, 1]
                if abs(np.sqrt(dx * dx + dy * dy) - 1.0) > 1e-8:
                    return False

        # Check non-overlap: for every pair, on at least one of 12 edge normals, projected intervals are apart by > 1e-6
        # Edge normals of a hexagon at angle a: directions a+0, a+60, ..., a+300, and their opposites.
        # Actually, the 6 edges have normals at a+0, a+60, ..., a+300 (outward normals).
        # Two hexagons are separated if for some normal direction n (from the 12 directions = 6 normals + 6 antiparallels),
        # the max projection of hex1 in direction n + min projection of hex2 in direction n are apart... 
        # Wait, the condition is: "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6"
        # Let's interpret: for a given normal direction u (unit vector), the interval of projections of hex A is [min_A, max_A] and for hex B is [min_B, max_B].
        # They are "apart" if max_A < min_B - 1e-6 or max_B < min_A - 1e-6.
        # The 12 edge normals: for hexagon with angle a, the 6 outward normals are at angles a+0, a+60, ..., a+300.
        # But the condition says "their 12 edge normals". Does it mean the union of both hexagons' normals? Or just the 6 normals of one?
        # "on at least one of their 12 edge normals" - likely means the 12 directions formed by the 6 normals of hex1 and 6 normals of hex2? Or 6 normals total?
        # Actually, for two hexagons, there are 6+6=12 edge normals if we consider both. But typically, separation is checked against all relevant directions.
        # Let's assume we check all 12 directions: 6 normals of hex1 and 6 normals of hex2.
        
        for i in range(11):
            for j in range(i + 1, 11):
                x1, y1, a1 = shifted_inner[i]
                x2, y2, a2 = shifted_inner[j]
                v1 = hex_vertices(x1, y1, 1.0, a1)
                v2 = hex_vertices(x2, y2, 1.0, a2)
                
                # Get normals of hex1
                n1_angles = np.radians(a1) + np.radians(60 * np.arange(6))
                n2_angles = np.radians(a2) + np.radians(60 * np.arange(6))
                
                all_normal_angles = np.concatenate([n1_angles, n2_angles])
                # Also include antiparallel? "12 edge normals" might just mean the 6 normals of each, so 12 total.
                # But a normal direction and its opposite define the same separation axis.
                # Let's just check the 12 unit vectors.
                
                separated = False
                for ang in all_normal_angles:
                    u = np.array([np.cos(ang), np.sin(ang)])
                    proj1 = v1 @ u
                    proj2 = v2 @ u
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                        separated = True
                        break
                if not separated:
                    return False

        # Check containment: every inner vertex p satisfies cross(p2 - p1, p - p1) >= -1e-6 for each outer edge
        for i in range(11):
            x, y, ang = shifted_inner[i]
            verts = hex_vertices(x, y, 1.0, ang)
            for p in verts:
                for k in range(6):
                    p1 = o_verts[k]
                    p2 = o_verts[(k + 1) % 6]
                    edge_vec = p2 - p1
                    vec_to_p = p - p1
                    cross_val = edge_vec[0] * vec_to_p[1] - edge_vec[1] * vec_to_p[0]
                    if cross_val < -1e-6:
                        return False
        return True

    def compute_min_side(inner, oa_deg):
        """Given inner hexagons (shifted so outer center is at origin) and outer angle, compute minimal side length."""
        # The outer hexagon is centered at origin with angle oa_deg.
        # Its edges are defined by 6 half-planes.
        # The vertices of the outer hexagon with side s and angle a are at distance s from center? No, circumradius = s.
        # Wait, for a regular hexagon with side s, the distance from center to vertex is s.
        # The vertices are at (s*cos(a+60k), s*sin(a+60k)).
        # The edges connect consecutive vertices.
        # The condition is: for each edge (p1, p2), and each inner vertex p, cross(p2-p1, p-p1) >= 0.
        # This means p is to the left of the directed edge p1->p2 (if we traverse CCW).
        # For a regular hexagon centered at origin with vertices at angles a, a+60, ..., the edges are:
        # Edge k connects vertex k to vertex k+1.
        # The half-plane constraint for edge k: the inner points must be inside.
        # Actually, it's simpler: the outer hexagon is the intersection of 6 half-planes.
        # Each half-plane is defined by an edge line. The distance from the center to each edge is s * cos(30) = s * sqrt(3)/2.
        # The inward normal to edge k points towards the center.
        # So the constraint is: for each edge k, the projection of all inner vertices onto the inward normal of edge k must be <= s * sqrt(3)/2.
        
        a_rad = np.radians(oa_deg)
        # Inward normals to the 6 edges
        # Edge k is between vertex k and k+1. The outward normal to edge k is at angle a + 60k + 30?
        # Let's verify. Vertex 0 is at angle a. Vertex 1 is at a+60.
        # The edge from V0 to V1. The outward normal is perpendicular to V1-V0, pointing away from center.
        # V1 - V0 = s*(cos(a+60)-cos(a), sin(a+60)-sin(a)).
        # The midpoint of V0 and V1 is at angle a+30, distance s*cos(30) = s*sqrt(3)/2.
        # So the outward normal direction is a+30.
        # Therefore, the inward normal is a+30+180 = a+210? No, the inward normal points from the edge to the center.
        # The vector from midpoint to center is in direction a+30+180.
        # So the constraint is: for all inner points p, p . (inward_normal_k) <= s * sqrt(3)/2.
        # Wait, if inward normal points from edge to center, then the distance from center to edge is s*sqrt(3)/2.
        # So the half-plane is { x : x . n_k <= s * sqrt(3)/2 } where n_k is the outward normal?
        # Let's re-derive.
        # Center is at origin. Edge midpoint is at M_k = (s*sqrt(3)/2) * (cos(a+30+60k), sin(a+30+60k)).
        # The outward normal is u_k = (cos(a+30+60k), sin(a+30+60k)).
        # The half-plane containing the origin is { x : x . u_k <= s*sqrt(3)/2 }.
        # Because for the origin, 0 . u_k = 0 <= s*sqrt(3)/2.
        # So for each inner vertex p, we need p . u_k <= s * sqrt(3)/2 for all k.
        # This means s >= max_k (max_p (p . u_k)) / (sqrt(3)/2) = (2/sqrt(3)) * max_k (max_p (p . u_k)).
        
        # Shift inner vertices so outer center is at origin.
        all_verts = []
        for i in range(11):
            x, y, ang = inner[i]
            # inner is already shifted in is_valid, but here we pass the raw inner and compute shifted.
            # Let's assume inner is raw, and we shift it by outer center.
            pass
        
        # Actually, let's compute the shifted vertices here.
        # The caller will pass inner in the final coordinate system.
        # Let's assume inner[:,0] and inner[:,1] are already relative to outer center.
        
        all_verts = []
        for i in range(11):
            x, y, ang = inner[i]
            v = hex_vertices(x, y, 1.0, ang)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        max_proj = 0
        for k in range(6):
            u_ang = a_rad + np.radians(30 + 60 * k)
            u = np.array([np.cos(u_ang), np.sin(u_ang)])
            projs = all_verts @ u
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        min_s = (2.0 / np.sqrt(3.0)) * max_proj
        return min_s

    # Generate candidate layouts
    best_score = -np.inf
    best_result = None
    
    # Layout 1: The 7-hexagon cluster (center + 6 around)
    # Center at (0,0). The 6 around are at distance 2*sqrt(3)/2? No.
    # Two unit hexagons touch if their centers are distance 2*sqrt(3)/2 apart? 
    # Wait, the distance between centers of two touching unit hexagons is 2 * (apothem) = 2 * sqrt(3)/2 = sqrt(3)? 
    # No, apothem is sqrt(3)/2. Distance between centers = 2 * apothem = sqrt(3) is for flat-to-flat.
    # But in a honeycomb, they are arranged such that centers are at distance 2*sqrt(3)/3 * something?
    # Let's just use the standard honeycomb lattice.
    # Lattice vectors: e1 = (3/2, sqrt(3)/2), e2 = (0, sqrt(3)). No, that's for hexagonal tiling with side 1?
    # The centers of a hexagonal tiling of unit hexagons form a triangular lattice.
    # The distance between nearest neighbors is 2 * (sqrt(3)/2) = sqrt(3)? No.
    # The width of a unit hexagon (flat to flat) is sqrt(3). So centers are sqrt(3) apart.
    # So the lattice basis can be:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5)
    # Check distance: |v2| = sqrt(3/4 + 2.25) = sqrt(3). Yes.
    
    sqrt3 = np.sqrt(3)
    v1 = np.array([sqrt3, 0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # 7-cluster: center + 6 neighbors
    cluster_centers = [np.array([0.0, 0.0])]
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if abs(i) + abs(j) <= 1 or (abs(i) == 1 and abs(j) == 1): # This is not the right condition for hex grid
                pass
    # Let's just list the 6 neighbors explicitly.
    # Neighbors of (0,0) in the lattice generated by v1, v2:
    # v1, -v1, v2, -v2, v1-v2, v2-v1
    neighbors = [v1, -v1, v2, -v2, v1 - v2, v2 - v1]
    cluster_centers.extend(neighbors)
    
    # The 12 next-layer neighbors (distance sqrt(3) from the cluster edge, i.e., distance 2*sqrt(3) from center? No)
    # The next layer in the hexagonal lattice around the 7-cluster.
    # These are at lattice coordinates (i,j) with max(|i|,|j|,|i+j|) = 2?
    # Let's generate all lattice points within radius 2 (in lattice distance).
    # Lattice point: i*v1 + j*v2.
    # The 7-cluster is i,j in {-1,0,1} with |i|+|j|+|i-j| <= 2? No, just the 7 points.
    # The 12 next layer:
    next_layer = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            # Check if it's in the 7-cluster
            in_cluster = False
            for c in cluster_centers:
                if np.allclose(p, c, atol=1e-6):
                    in_cluster = True
                    break
            if not in_cluster:
                # Check if it's at the next layer (distance 1 from cluster)
                # It should be adjacent to at least one cluster member.
                adj = False
                for c in cluster_centers:
                    if np.allclose(p - c, v1, atol=1e-6) or np.allclose(p - c, -v1, atol=1e-6) or \
                       np.allclose(p - c, v2, atol=1e-6) or np.allclose(p - c, -v2, atol=1e-6) or \
                       np.allclose(p - c, v1 - v2, atol=1e-6) or np.allclose(p - c, v2 - v1, atol=1e-6):
                        adj = True
                        break
                if adj:
                    next_layer.append(p)
    
    # We should have 12 next-layer points.
    # Let's verify.
    if len(next_layer) != 12:
        # Fallback: just take all lattice points within distance 2*sqrt(3) from origin, excluding the 7-cluster.
        next_layer = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i * v1 + j * v2
                if np.linalg.norm(p) > 1.5 * sqrt3: # Approximate
                    continue
                in_cluster = False
                for c in cluster_centers:
                    if np.allclose(p, c, atol=1e-6):
                        in_cluster = True
                        break
                if not in_cluster:
                    next_layer.append(p)
        # Deduplicate and take first 12
        unique_next = []
        for p in next_layer:
            found = False
            for q in unique_next:
                if np.allclose(p, q, atol=1e-6):
                    found = True
                    break
            if not found:
                unique_next.append(p)
        next_layer = unique_next[:12]

    # Candidate subsets of 4 from the 12 next-layer points
    # We can't iterate over all C(12,4)=495 subsets for many angles if it's slow, but it should be fine.
    # Let's try a subset of subsets to save time.
    # Or just try all 495 if it's fast enough.
    
    candidate_subsets = []
    # Symmetric subsets:
    # 1. All 4 from "gaps" (opposite pairs)
    # 2. All 4 from "direct" (adjacent)
    # 3. Mix
    # Let's just take all 495.
    idx_range = range(len(next_layer))
    candidate_subsets = list(combinations(idx_range, 4))
    
    # To save time, let's limit to 100 random subsets + some symmetric ones
    np.random.seed(42)
    if len(candidate_subsets) > 100:
        # Take first 20 and 80 random
        candidate_subsets = candidate_subsets[:20]
        random_idx = np.random.choice(len(candidate_subsets[20:]), 80, replace=False)
        candidate_subsets.extend([candidate_subsets[20 + i] for i in random_idx])
    
    angles_to_try = [0, 15, 30, 45, 60]
    
    for subset in candidate_subsets:
        if time.time() > deadline - 50:
            break
        
        # Build the 11 centers
        centers = cluster_centers.copy()
        for idx in subset:
            centers.append(next_layer[idx])
        
        # Center of mass
        com = np.mean(centers, axis=0)
        
        # Try to shift the cluster so that the enclosing hexagon is minimized.
        # We can try shifting by a small grid around the COM.
        # Or just use the COM as the outer center.
        
        for oa_deg in angles_to_try:
            if time.time() > deadline - 50:
                break
            
            # Shift centers so that COM is at origin
            shifted_centers = [c - com for c in centers]
            
            # Compute minimal side length
            inner_raw = np.array([[c[0], c[1], 0.0] for c in shifted_centers])
            side = compute_min_side(inner_raw, oa_deg)
            
            # Check validity
            outer = np.array([0.0, 0.0, float(oa_deg)])
            if is_valid(inner_raw, outer, side):
                score = 1.0 / side
                if score > best_score:
                    best_score = score
                    # Convert back to original coordinates
                    inner_final = np.array([[c[0] + com[0], c[1] + com[1], 0.0] for c in shifted_centers])
                    outer_final = np.array([com[0], com[1], float(oa_deg)])
                    best_result = (inner_final, outer_final, side)

    if best_result is None:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    return best_result

# EVOLVE-BLOCK-END
