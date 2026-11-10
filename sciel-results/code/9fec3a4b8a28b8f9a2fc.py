"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    d = 7
    n = 1 << d  # 128 vertices
    
    # Precompute all edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v))
    
    m = len(all_edges)  # 448 edges
    
    edge_to_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx
    
    # For each edge, find all 4-cycles containing it
    # face_conflicts[edge_idx] = list of tuples (e1, e2, e3)
    face_conflicts = [[] for _ in range(m)]
    
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = u ^ (1 << j)
            e = v ^ (1 << j)
            e1 = (min(u, c), max(u, c))
            e2 = (min(v, e), max(v, e))
            e3 = (min(c, e), max(c, e))
            face_conflicts[idx].append((edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3]))
    
    # Precompute reverse mapping: for each edge, which faces does it participate in
    # conflict_count[e] = number of faces where e is the "missing" edge that would complete a C4
    # Actually, we need: for each edge e, the list of (face_id, position) where it's one of the 3 others
    # Better: maintain a count per edge of how many "blocking triples" are fully selected
    
    # Strategy: maintain conflict_count[idx] = number of 4-cycles where all OTHER 3 edges are chosen
    # When we add edge idx, for each face containing idx, increment the count of the 3 other edges
    # When we remove edge idx, decrement those counts
    
    # Precompute: for each edge idx, for each face it's in, which 3 edges are the "others"
    # face_conflicts[idx] already gives us this
    
    # Also precompute: for each edge, the list of (edge_idx, face_idx) pairs
    # where this edge is one of the 3 "others" in that face
    # This lets us efficiently update conflict counts
    
    # For edge e, in face f (which has edges e, a, b, c), e is "other" for the remaining 3
    # So: others_in_face[e][f_idx] = list of the other 3 edge indices
    
    # Let's build: for each edge, a list of (the 3 edges it blocks when all 3 are present)
    # Actually simpler: 
    #   conflict_count[idx] = number of faces where the other 3 edges of that face are all chosen
    #   To check if we can add idx: conflict_count[idx] == 0
    #   When adding edge e: for each face containing e, for each of the other 3 edges in that face,
    #     if the remaining 2 others are also chosen, increment conflict_count of the face's "target"
    #   This is getting complex. Let me use a different approach.
    
    # Alternative: for each edge e, precompute the list of faces it belongs to
    # face_of_edge[e] = list of face_ids
    # Each face has 4 edges. face_edges[face_id] = (e0, e1, e2, e3)
    
    # Build faces:
    # A face is determined by choosing 2 directions (i, j) and a base vertex with those bits fixed
    # For directions i < j, the face vertices are: base, base^i, base^j, base^(i|j)
    # where base has bits i and j set to 0
    
    # But easier: each edge (u,v) with direction i participates in faces with direction j (j!=i)
    # The face is: {u, v, u^j, v^j}
    
    # Let's just use the face_conflicts structure directly with a count-based approach
    
    # For each edge idx, conflict_count[idx] = number of 4-cycles where all 3 other edges are selected
    # can_add(idx) iff conflict_count[idx] == 0
    
    # When we select edge e:
    #   For each face containing e (with other edges a, b, c):
    #     If b and c are both selected, then conflict_count[a] += 1 (and vice versa for other pairs)
    #     Actually: the face has edges {e, a, b, c}. 
    #     Adding e means: for edge a, the "other 3" are {e, b, c}. If b and c were already chosen, 
    #     then a's conflict_count increases by 1. Similarly for b and c.
    
    # So: when adding edge e, for each face (e, a, b, c):
    #   if chosen[b] and chosen[c]: conflict_count[a] += 1
    #   if chosen[a] and chosen[c]: conflict_count[b] += 1
    #   if chosen[a] and chosen[b]: conflict_count[c] += 1
    
    # When removing edge e, do the reverse.
    
    def try_greedy(seed):
        rng = random.Random(seed)
        edge_order = list(range(m))
        rng.shuffle(edge_order)
        
        chosen = np.zeros(m, dtype=bool)
        conflict_count = np.zeros(m, dtype=np.int32)
        count = 0
        
        for idx in edge_order:
            if conflict_count[idx] == 0:
                chosen[idx] = True
                count += 1
                # Update conflict counts for edges in faces containing idx
                for (a, b, c) in face_conflicts[idx]:
                    if chosen[b] and chosen[c]:
                        conflict_count[a] += 1
                    if chosen[a] and chosen[c]:
                        conflict_count[b] += 1
                    if chosen[a] and chosen[b]:
                        conflict_count[c] += 1
        
        return chosen, count, conflict_count
    
    def local_improve(chosen, count, conflict_count):
        """Try to add more edges, then try swaps."""
        chosen = chosen.copy()
        conflict_count = conflict_count.copy()
        
        # First pass: add any edge that can be added
        for idx in range(m):
            if not chosen[idx] and conflict_count[idx] == 0:
                chosen[idx] = True
                count += 1
                for (a, b, c) in face_conflicts[idx]:
                    if chosen[b] and chosen[c]:
                        conflict_count[a] += 1
                    if chosen[a] and chosen[c]:
                        conflict_count[b] += 1
                    if chosen[a] and chosen[b]:
                        conflict_count[c] += 1
        
        # Second pass: try 1-for-2 swaps
        # Remove edge e, see if we can add 2+ edges
        for idx in range(m):
            if not chosen[idx]:
                continue
            # Temporarily remove
            chosen[idx] = False
            count -= 1
            # Update conflict counts (decrement)
            for (a, b, c) in face_conflicts[idx]:
                if chosen[b] and chosen[c]:
                    conflict_count[a] -= 1
                if chosen[a] and chosen[c]:
                    conflict_count[b] -= 1
                if chosen[a] and chosen[b]:
                    conflict_count[c] -= 1
            
            # Try to add edges
            gained = 0
            added = []
            for jdx in range(m):
                if chosen[jdx] or conflict_count[jdx] > 0:
                    continue
                chosen[jdx] = True
                count += 1
                gained += 1
                added.append(jdx)
                for (a, b, c) in face_conflicts[jdx]:
                    if chosen[b] and chosen[c]:
                        conflict_count[a] += 1
                    if chosen[a] and chosen[c]:
                        conflict_count[b] += 1
                    if chosen[a] and chosen[b]:
                        conflict_count[c] += 1
            
            if gained >= 2:
                # Keep the improvement
                pass
            else:
                # Revert
                for jdx in added:
                    chosen[jdx] = False
                    count -= 1
                    for (a, b, c) in face_conflicts[jdx]:
                        if chosen[b] and chosen[c]:
                            conflict_count[a] -= 1
                        if chosen[a] and chosen[c]:
                            conflict_count[b] -= 1
                        if chosen[a] and chosen[b]:
                            conflict_count[c] -= 1
                # Restore idx
                chosen[idx] = True
                count += 1
                for (a, b, c) in face_conflicts[idx]:
                    if chosen[b] and chosen[c]:
                        conflict_count[a] += 1
                    if chosen[a] and chosen[c]:
                        conflict_count[b] += 1
                    if chosen[a] and chosen[b]:
                        conflict_count[c] += 1
        
        return chosen, count, conflict_count
    
    # Run multiple greedy restarts with different seeds
    best_count = 0
    best_chosen = None
    best_conflict = None
    
    num_restarts = 0
    for seed in range(5000):
        if time.time() > deadline - 15.0:
            break
        chosen, count, conflict_count = try_greedy(seed)
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
            best_conflict = conflict_count.copy()
        num_restarts += 1
    
    # Local improvement on best result
    if best_chosen is not None and time.time() < deadline - 10.0:
        best_chosen, best_count, best_conflict = local_improve(best_chosen, best_count, best_conflict)
    
    # Convert back to list of edges
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            out.append(list(all_edges[idx]))
    
    return out
# EVOLVE-BLOCK-END
