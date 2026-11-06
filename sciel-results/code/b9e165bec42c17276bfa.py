import random
import time
import sys

def construct():
    start = time.time()
    deadline = start + 118.0  # safety margin

    d = 7
    n = 1 << d  # 128 vertices

    # 1. Build the conflict graph
    # Vertices are edges of Q7. 
    # An edge in Q7 is (u, v) where u^v is a power of 2.
    # Two edges (u,v) and (x,y) conflict if they form a C4 in Q7.
    # This happens if {u,v} and {x,y} are in the same 2D face.
    # A 2D face is defined by two distinct bits i, j.
    # Vertices of the face are z, z^(1<<i), z^(1<<j), z^(1<<i)^(1<<j).
    # The edges of the face are the pairs of these 4 vertices that differ by one bit.
    # There are 4 edges in a face. All pairs of these 4 edges conflict.

    # Total edges in Q7: 448.
    # Let's assign an index to each edge.
    all_edges = []
    edge_id = {}
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_id[(u, v)] = idx
                all_edges.append((u, v))
                idx += 1
    
    E = len(all_edges) # 448

    # Build adjacency list for the conflict graph
    # adj[e] = set of edges conflicting with e
    adj = [set() for _ in range(E)]
    
    # For each 2D face, add conflicts
    # A face is determined by a base vertex z and two bits i, j (i < j)
    # z can be any vertex such that z has 0 in bits i and j? No, just pick a canonical representation.
    # To avoid duplicates, iterate z from 0 to n-1, and i < j.
    # But we must ensure we don't process the same face 4 times (once for each vertex).
    # A face is unique if we fix the smallest vertex in the face as the base?
    # Or simply: for each pair of bits (i, j), the faces partition the cube into 2^(d-2) faces.
    # For fixed i, j, the faces are defined by the values of the other d-2 bits.
    # Let's iterate i, j and the "other bits" configuration.
    
    for i in range(d):
        for j in range(i + 1, d):
            bit_i = 1 << i
            bit_j = 1 << j
            # Iterate over all combinations of other bits
            # There are 2^(d-2) such combinations.
            # We can iterate z from 0 to n-1, but only consider z where z & bit_i == 0 and z & bit_j == 0?
            # No, that's not unique.
            # Better: Iterate k from 0 to 2^(d-2)-1. Map k to the bits excluding i and j.
            # This is complex. 
            # Alternative: Iterate all z in 0..n-1. For each z, for each pair i,j, 
            # if z is the "minimum" vertex of the face (i.e., z has 0 in i and j, and for any k < z in the face... no).
            # Simplest: Iterate all z. For each i, j.
            # The face vertices are z, z^bi, z^bj, z^bi^bj.
            # To avoid duplicates, only process if z is the smallest vertex in the face.
            # Smallest vertex means z < z^bi, z < z^bj, z < z^bi^bj.
            # This implies bit_i is 0 in z and bit_j is 0 in z.
            # So, for each i, j, we only consider z such that (z & bit_i) == 0 and (z & bit_j) == 0.
            
            for z in range(n):
                if (z & bit_i) != 0 or (z & bit_j) != 0:
                    continue
                
                v1 = z
                v2 = z ^ bit_i
                v3 = z ^ bit_j
                v4 = z ^ bit_i ^ bit_j
                
                # The 4 edges in this face
                e1 = edge_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_id[(min(v1, v3), max(v1, v3))]
                e3 = edge_id[(min(v2, v4), max(v2, v4))]
                e4 = edge_id[(min(v3, v4), max(v3, v4))]
                
                face_edges = [e1, e2, e3, e4]
                
                # All pairs conflict
                for k in range(4):
                    for l in range(k + 1, 4):
                        adj[face_edges[k]].add(face_edges[l])
                        adj[face_edges[l]].add(face_edges[k])

    # Convert to lists for faster iteration
    adj_lists = [list(s) for s in adj]
    degrees = [len(l) for l in adj_lists]

    # Heuristic: Greedy Independent Set
    def greedy_mis(order):
        chosen = [False] * E
        chosen_set = set()
        # order is a permutation of 0..E-1
        for v in order:
            if not chosen[v]:
                # Check if any neighbor is chosen
                can_add = True
                for u in adj_lists[v]:
                    if chosen[u]:
                        can_add = False
                        break
                if can_add:
                    chosen[v] = True
                    chosen_set.add(v)
        return chosen_set

    # Local Search: Hill Climbing
    # State: chosen_set (set of indices)
    # Move: Add v, remove N(v) intersect chosen_set
    # Gain: 1 - |N(v) intersect chosen_set|
    
    def local_search(initial_set, rng, time_limit):
        chosen = set(initial_set)
        chosen_bool = [False] * E
        for v in chosen:
            chosen_bool[v] = True
        
        best_set = set(chosen)
        best_size = len(chosen)
        
        # Precompute neighbors for fast access
        # We need to know, for each v not in chosen, how many neighbors are in chosen.
        # Let's maintain an array `neighbor_count` where neighbor_count[v] = |N(v) ∩ chosen|
        neighbor_count = [0] * E
        for v in chosen:
            for u in adj_lists[v]:
                neighbor_count[u] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_gain = 0
            best_v = -1
            
            # Find the best single move
            for v in range(E):
                if chosen_bool[v]:
                    continue
                gain = 1 - neighbor_count[v]
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v != -1:
                # Apply move
                # Remove neighbors
                for u in adj_lists[best_v]:
                    if chosen_bool[u]:
                        chosen.discard(u)
                        chosen_bool[u] = False
                        # Update neighbor counts for neighbors of u
                        for w in adj_lists[u]:
                            neighbor_count[w] -= 1
                # Add best_v
                chosen.add(best_v)
                chosen_bool[best_v] = True
                # Update neighbor counts for neighbors of best_v
                for w in adj_lists[best_v]:
                    neighbor_count[w] += 1
                
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_set = set(chosen)
                improved = True
            else:
                # No improving single move. Try a 2-move or break.
                # For efficiency, we might just break or try a random perturbation if we have time.
                # Let's try to find a 2-move: add u, v. Remove N(u) U N(v) intersect chosen.
                # This is expensive. O(E^2). E=448. 448^2 ~ 200k. 
                # We can sample a few pairs.
                
                if time.time() < time_limit - 1:
                    # Sample 100 random pairs
                    non_chosen = [v for v in range(E) if not chosen_bool[v]]
                    if len(non_chosen) >= 2:
                        for _ in range(100):
                            u1 = rng.choice(non_chosen)
                            u2 = rng.choice(non_chosen)
                            if u1 == u2:
                                continue
                            
                            # Calculate removal set
                            rem_set = set()
                            for w in adj_lists[u1]:
                                if chosen_bool[w]:
                                    rem_set.add(w)
                            for w in adj_lists[u2]:
                                if chosen_bool[w]:
                                    rem_set.add(w)
                            
                            gain = 2 - len(rem_set)
                            if gain > 0:
                                # Apply 2-move
                                for w in rem_set:
                                    chosen.discard(w)
                                    chosen_bool[w] = False
                                    for x in adj_lists[w]:
                                        neighbor_count[x] -= 1
                                chosen.add(u1)
                                chosen_bool[u1] = True
                                for x in adj_lists[u1]:
                                    neighbor_count[x] += 1
                                chosen.add(u2)
                                chosen_bool[u2] = True
                                for x in adj_lists[u2]:
                                    neighbor_count[x] += 1
                                
                                if len(chosen) > best_size:
                                    best_size = len(chosen)
                                    best_set = set(chosen)
                                improved = True
                                break
                            
                # If no improvement even with 2-moves, we are stuck in a local optimum.
                # We could break, or use Simulated Annealing.
                # Given the small size, let's break and rely on multiple restarts.
                break
                
        return best_set

    # Main Loop
    best_global = set()
    best_global_size = 0
    rng = random.Random(42)
    
    # Multiple restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Generate ordering
        # Strategy: Sort by degree descending, then randomize ties
        # Or just random order
        # Let's mix: 
        if r % 2 == 0:
            # Degree based
            order = list(range(E))
            order.sort(key=lambda x: -degrees[x])
            # Add some randomness to the sort key to break ties differently
            # Actually, stable sort with random tie-breaker
            # Simpler: shuffle, then stable sort by -degree? No.
            # Let's just use random order for some, degree for others.
            pass
        else:
            order = list(range(E))
            rng.shuffle(order)
            
        # If even, do degree sort with random perturbation
        if r % 2 == 0:
            # Perturbed degree
            keys = [(-degrees[i], rng.random()) for i in range(E)]
            order = [i for i, _ in sorted(enumerate(keys), key=lambda x: x[1])]
        
        init_set = greedy_mis(order)
        
        # Local Search
        time_limit = deadline - 20
        ls_set = local_search(init_set, rng, time_limit)
        
        if len(ls_set) > best_global_size:
            best_global_size = len(ls_set)
            best_global = ls_set
            
        # Check if we found 304
        if best_global_size >= 304:
             # We can try to optimize further or just break if we have time
             # But 304 is the target. Let's keep searching for 305+ if possible, 
             # but 304 is the "best known".
             pass

    # Final check
    if time.time() < deadline - 5:
         # One more local search on the best global
         ls_final = local_search(best_global, rng, deadline - 5)
         if len(ls_final) > best_global_size:
             best_global = ls_final
             best_global_size = len(ls_final)

    # Convert to list of [u, v]
    out = []
    for idx in sorted(best_global):
        u, v = all_edges[idx]
        out.append([u, v])
    return out
