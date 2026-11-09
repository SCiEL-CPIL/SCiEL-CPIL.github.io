"""Four-cycle-free edge sets of Q7: maximize edges with no 2-face fully selected."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _build_edge_data():
    """Precompute all edges and face constraints for Q7."""
    d = 7
    n = 1 << d  # 128
    # Each edge: (u, v, dim) where u < v and they differ in bit 'dim'
    edges = []
    edge_index = {}  # (u, v) -> index
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y, i))
                edge_index[(x, y)] = idx
    
    E = len(edges)  # 448
    
    # For each edge, find the 6 faces (2D squares) it belongs to.
    # A face is defined by two dimensions (i, j) and a vertex.
    # Edge (x, y) with dim i belongs to a face with dim j (j != i) if
    # the face has edges in dims i and j.
    # A face is identified by (sorted dims, base vertex) where base vertex
    # is the one with both bits 0 in those dims.
    # Actually, a 2-face in Q7 is a set of 4 vertices. It's identified by
    # choosing 2 dimensions and fixing the other 5 bits.
    # Let's use: face_id = (i, j, fixed_bits_mask, fixed_bits_value)
    # But simpler: for edge (x, y, i), and each j != i, the face is the 4-cycle
    # x -> y (dim i), y -> y^bit_j (dim j), y^bit_j -> x^bit_j (dim i), x^bit_j -> x (dim j)
    # The 4 edges of this face are:
    #   (min(x,y), max(x,y)) = the edge itself
    #   (min(y, y^bit_j), max(y, y^bit_j))
    #   (min(x^bit_j, y^bit_j), max(x^bit_j, y^bit_j))
    #   (min(x, x^bit_j), max(x, x^bit_j))
    
    # For each edge, store the list of other 3 edges in each face it belongs to.
    # edge_faces[e_idx] = list of (face_id, [other_edge_idx1, other_edge_idx2, other_edge_idx3])
    # But we need: when we add edge e, which faces become "full"?
    # A face is full when all 4 of its edges are selected.
    # So for each face, track how many of its 4 edges are selected.
    
    # Let's build: for each face, the 4 edge indices.
    # Face identification: (i, j, x_base) where x_base has bits i and j set to 0.
    # Actually let's just enumerate faces directly.
    # A 2-face is determined by 2 dimensions (i, j) and the values of the other 5 bits.
    # Number of faces: C(7,2) * 2^5 = 21 * 32 = 672
    
    # For each face, store its 4 edge indices.
    face_edges = []  # list of 4 edge indices per face
    face_id_map = {}  # (i, j, fixed) -> face index
    
    for i in range(d):
        for j in range(i + 1, d):
            for fixed in range(1 << (d - 2)):
                # Reconstruct the base vertex with bits i, j = 0
                # and other bits from 'fixed'
                base = 0
                bit_pos = 0
                for k in range(d):
                    if k == i or k == j:
                        continue
                    if fixed & (1 << bit_pos):
                        base |= (1 << k)
                    bit_pos += 1
                # The 4 vertices of this face:
                v00 = base
                v10 = base | (1 << i)
                v01 = base | (1 << j)
                v11 = base | (1 << i) | (1 << j)
                # 4 edges:
                # (v00, v10) dim i, (v00, v01) dim j, (v10, v11) dim j, (v01, v11) dim i
                e1 = edge_index[(min(v00, v10), max(v00, v10))]
                e2 = edge_index[(min(v00, v01), max(v00, v01))]
                e3 = edge_index[(min(v10, v11), max(v10, v11))]
                e4 = edge_index[(min(v01, v11), max(v01, v11))]
                fid = len(face_edges)
                face_edges.append([e1, e2, e3, e4])
                face_id_map[(i, j, fixed)] = fid
    
    F = len(face_edges)  # 672
    
    # For each edge, list of faces it belongs to
    edge_to_faces = [[] for _ in range(E)]
    for fid, fedges in enumerate(face_edges):
        for eidx in fedges:
            edge_to_faces[eidx].append(fid)
    
    return edges, E, face_edges, F, edge_to_faces, d, n


def _greedy_init(edges, E, face_edges, F, edge_to_faces, rng):
    """Greedy construction with random edge ordering."""
    selected = np.zeros(E, dtype=np.int8)
    face_count = np.zeros(F, dtype=np.int8)
    
    order = list(range(E))
    rng.shuffle(order)
    
    count = 0
    for eidx in order:
        # Check if adding this edge would complete any face
        can_add = True
        for fid in edge_to_faces[eidx]:
            if face_count[fid] == 3:
                can_add = False
                break
        if can_add:
            selected[eidx] = 1
            count += 1
            for fid in edge_to_faces[eidx]:
                face_count[fid] += 1
    
    return selected, face_count, count


def _local_search(selected, face_count, count, edges, E, face_edges, F, edge_to_faces, rng, time_limit):
    """Local search: try to add unselected edges, remove selected ones if needed."""
    start_time = time.time()
    
    # Build list of unselected edges
    unselected = [i for i in range(E) if selected[i] == 0]
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        rng.shuffle(unselected)
        for eidx in unselected:
            if selected[eidx]:
                continue
            # Check if we can add this edge
            can_add = True
            for fid in edge_to_faces[eidx]:
                if face_count[fid] == 3:
                    can_add = False
                    break
            if can_add:
                selected[eidx] = 1
                count += 1
                for fid in edge_to_faces[eidx]:
                    face_count[fid] += 1
                improved = True
                # Remove from unselected list
                unselected.remove(eidx)
    
    # Now try to swap: remove an edge, add two others
    # This is more complex; let's try a simpler approach:
    # Try removing each selected edge and see if we can add more than 1 back
    selected_list = [i for i in range(E) if selected[i] == 1]
    improved2 = True
    while improved2 and (time.time() - start_time) < time_limit:
        improved2 = False
        rng.shuffle(selected_list)
        for rem_eidx in selected_list:
            if not selected[rem_eidx]:
                continue
            # Temporarily remove
            selected[rem_eidx] = 0
            count -= 1
            for fid in edge_to_faces[rem_eidx]:
                face_count[fid] -= 1
            
            # Try to add as many unselected edges as possible
            added = 0
            new_adds = []
            for eidx in range(E):
                if selected[eidx]:
                    continue
                can_add = True
                for fid in edge_to_faces[eidx]:
                    if face_count[fid] == 3:
                        can_add = False
                        break
                if can_add:
                    selected[eidx] = 1
                    count += 1
                    added += 1
                    new_adds.append(eidx)
                    for fid in edge_to_faces[eidx]:
                        face_count[fid] += 1
            
            if added <= 1:
                # Revert: remove the newly added ones, restore rem_eidx
                for eidx in new_adds:
                    selected[eidx] = 0
                    count -= 1
                    for fid in edge_to_faces[eidx]:
                        face_count[fid] -= 1
                selected[rem_eidx] = 1
                count += 1
                for fid in edge_to_faces[rem_eidx]:
                    face_count[fid] += 1
            else:
                improved2 = True
                # Update selected_list
                selected_list = [i for i in range(E) if selected[i] == 1]
    
    return selected, face_count, count


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    edges, E, face_edges, F, edge_to_faces, d, n = _build_edge_data()
    
    best_count = 0
    best_selected = None
    
    # Multiple random restarts with greedy + local search
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
        
        rng = random.Random(attempt * 7919 + 13)
        
        selected, face_count, count = _greedy_init(edges, E, face_edges, F, edge_to_faces, rng)
        
        # Local search with time limit
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
        
        selected, face_count, count = _local_search(
            selected, face_count, count, edges, E, face_edges, F, edge_to_faces, rng, remaining
        )
        
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
        
        if time.time() > deadline - 15:
            break
    
    # If we have a best solution, convert to edge list
    if best_selected is not None:
        result = []
        for i in range(E):
            if best_selected[i]:
                u, v, dim = edges[i]
                result.append([u, v])
        return result
    else:
        # Fallback: simple greedy
        result = []
        chosen = set()
        for x in range(n):
            for i in range(d):
                y = x ^ (1 << i)
                if x < y:
                    ok = True
                    for j in range(d):
                        if j != i:
                            c, e = x ^ (1 << j), y ^ (1 << j)
                            if all(p in chosen for p in ((min(x, c), max(x, c)), (min(y, e), max(y, e)), (min(c, e), max(c, e)))):
                                ok = False
                                break
                    if ok:
                        chosen.add((x, y))
                        result.append([x, y])
        return result

# EVOLVE-BLOCK-END
