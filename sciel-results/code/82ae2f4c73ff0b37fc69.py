# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

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

        # Check non-overlap
        for i in range(11):
            for j in range(i + 1, 11):
                x1, y1, a1 = shifted_inner[i]
                x2, y2, a2 = shifted_inner[j]
                v1 = hex_vertices(x1, y1, 1.0, a1)
                v2 = hex_vertices(x2, y2, 1.0, a2)
                
                # Normals for hex1 and hex2
                n1_angles = np.radians(a1) + np.radians(60 * np.arange(6))
                n2_angles = np.radians(a2) + np.radians(60 * np.arange(6))
                all_normal_angles = np.concatenate([n1_angles, n2_angles])
                
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

        # Check containment
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
        """Compute minimal side length for given inner layout (centered at 0,0) and outer angle."""
        a_rad = np.radians(oa_deg)
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

    # Hexagonal lattice basis for unit hexagons
    # Neighbors are at distance sqrt(3)
    sqrt3 = np.sqrt(3)
    v1 = np.array([sqrt3, 0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # Generate centers for the 7-cluster (center + 6 neighbors)
    cluster_centers = [np.array([0.0, 0.0])]
    neighbors = [v1, -v1, v2, -v2, v1 - v2, v2 - v1]
    cluster_centers.extend(neighbors)
    
    # Generate next layer (12 points)
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
                # Check if it's adjacent to the cluster
                adj = False
                for c in cluster_centers:
                    diff = p - c
                    if np.allclose(diff, v1, atol=1e-6) or np.allclose(diff, -v1, atol=1e-6) or \
                       np.allclose(diff, v2, atol=1e-6) or np.allclose(diff, -v2, atol=1e-6) or \
                       np.allclose(diff, v1 - v2, atol=1e-6) or np.allclose(diff, v2 - v1, atol=1e-6):
                        adj = True
                        break
                if adj:
                    next_layer.append(p)
    
    # Ensure we have 12 points, if not, deduplicate
    unique_next = []
    for p in next_layer:
        found = False
        for q in unique_next:
            if np.allclose(p, q, atol=1e-6):
                found = True
                break
        if not found:
            unique_next.append(p)
    next_layer = unique_next
    
    if len(next_layer) < 12:
        # Fallback: just use a larger ring if something went wrong
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i * v1 + j * v2
                in_cluster = False
                for c in cluster_centers:
                    if np.allclose(p, c, atol=1e-6):
                        in_cluster = True
                        break
                if not in_cluster and not any(np.allclose(p, q, atol=1e-6) for q in next_layer):
                    next_layer.append(p)
                if len(next_layer) >= 12:
                    break
            if len(next_layer) >= 12:
                break
        next_layer = next_layer[:12]

    # Try different subsets of 4 from the next layer
    # To avoid combinatorial explosion, we pick symmetric and random subsets
    indices = list(range(len(next_layer)))
    subsets = []
    
    # Symmetric subsets: opposite pairs
    # Indices 0, 3, 6, 9 might be opposite if ordered circularly, but our generation is grid-based.
    # Let's just take some deterministic subsets and some random ones safely.
    deterministic_subsets = [
        (0, 1, 2, 3),
        (0, 1, 4, 5),
        (0, 2, 4, 6),
        (1, 3, 5, 7),
        (0, 3, 6, 9) if len(next_layer) > 9 else (0, 1, 2, 3),
    ]
    subsets.extend(deterministic_subsets)
    
    # Add some random subsets safely
    if len(next_layer) >= 4:
        np.random.seed(42)
        for _ in range(50):
            subset = tuple(np.random.choice(len(next_layer), 4, replace=False))
            if subset not in subsets:
                subsets.append(subset)

    best_score = -np.inf
    best_result = None

    for subset in subsets:
        if time.time() > deadline - 50:
            break
        
        # Build the 11 centers
        centers = cluster_centers.copy()
        for idx in subset:
            if idx < len(next_layer):
                centers.append(next_layer[idx])
        
        if len(centers) != 11:
            continue

        # Center of mass
        com = np.mean(centers, axis=0)
        
        # Try different outer hexagon angles
        for oa_deg in [0, 15, 30, 45, 60, 75, 90]:
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
        # Fallback to a known valid configuration (seed-like)
        # This configuration might not be optimal but should be valid if constructed carefully
        # Let's use the 7-cluster + 4 more, centered at 0,0, with side length calculated
        centers = cluster_centers.copy()
        # Add 4 from next layer
        if len(next_layer) >= 4:
            centers.extend(next_layer[:4])
        
        com = np.mean(centers, axis=0)
        shifted_centers = [c - com for c in centers]
        inner_raw = np.array([[c[0], c[1], 0.0] for c in shifted_centers])
        
        # Try a few angles to find a valid one
        for oa_deg in [0, 30, 60]:
            outer = np.array([0.0, 0.0, float(oa_deg)])
            side = compute_min_side(inner_raw, oa_deg)
            if is_valid(inner_raw, outer, side):
                inner_final = np.array([[c[0] + com[0], c[1] + com[1], 0.0] for c in shifted_centers])
                outer_final = np.array([com[0], com[1], float(oa_deg)])
                return inner_final, outer_final, side

        # Absolute fallback: return a very large hexagon to ensure validity
        # 11 hexagons in a line? No, let's just use a large side length
        inner_hex_data = np.zeros((11, 3))
        for i in range(11):
            inner_hex_data[i] = [i * 2 * sqrt3, 0, 0]
        outer_hex_data = np.array([5 * 2 * sqrt3, 0, 0]) # Center of the line
        outer_hex_side_length = 5 * 2 * sqrt3 + 2 # Add margin
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    return best_result

# EVOLVE-BLOCK-END
