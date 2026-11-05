def construct():
    import time
    import itertools
    import numpy as np
    import random
    
    start_time = time.time()
    DEADLINE = 115
    
    n = 16
    R = list(range(3, 16))
    nR = 13
    
    # Precompute Q3 subgraph check using bitmasks
    # Q3 has 8 vertices, 12 edges. It's a bipartite graph with parts of size 4.
    # A Q3 exists iff there's a set of 8 vertices where the induced subgraph
    # is isomorphic to Q3.
    
    # Faster approach: enumerate all 8-subsets of R, check if any is Q3
    # But that's C(13,8) = 1287 subsets, each check is O(8^2) = manageable
    
    # Even faster: for each pair of 4-sets (A, B) that partition an 8-set,
    # check if the bipartite graph between A and B is exactly Q3's bipartite structure
    # Q3 bipartite: each vertex in A has degree 3 to B, each in B has degree 3 to A
    
    # Precompute all ways to partition 13 elements into two 4-sets
    # Actually, we need to check: does adding edge (i,j) create a Q3?
    # A Q3 containing edge (i,j) means i,j are in the same 8-vertex set
    # and the induced subgraph on those 8 vertices is Q3.
    
    # Strategy: For each 6-subset S of the other 11 vertices, form 8-set = S ∪ {i,j}
    # Check if induced subgraph on 8-set is Q3.
    # C(11,6) = 462 checks per edge addition, each O(64) ops = fast
    
    def is_q3(adj_bits, eight_set):
        """Check if induced subgraph on eight_set is isomorphic to Q3.
        adj_bits: list of 13 uint16 bitmasks for R vertices.
        eight_set: tuple of 8 indices (0..12) into R.
        Q3 is bipartite with parts of size 4, each vertex degree 3.
        """
        # Q3 has exactly 12 edges. Check edge count first.
        edge_count = 0
        for a_idx in range(8):
            for b_idx in range(a_idx+1, 8):
                if adj_bits[eight_set[a_idx]] & (1 << eight_set[b_idx]):
                    edge_count += 1
        if edge_count != 12:
            return False
        # Check bipartite: try all 70 ways to split 8 into 4+4
        # C(8,4) = 70
        for part1 in itertools.combinations(range(8), 4):
            part2 = tuple(v for v in range(8) if v not in part1)
            ok = True
            for a in part1:
                deg = 0
                for b in part2:
                    if adj_bits[eight_set[a]] & (1 << eight_set[b]):
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if not ok:
                continue
            for b in part2:
                deg = 0
                for a in part1:
                    if adj_bits[eight_set[a]] & (1 << eight_set[b]):
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if ok:
                return True
        return False
    
    def check_q3_after_add(adj_bits, i, j):
        """Check if adding edge (i,j) creates a Q3 in R subgraph."""
        others = [k for k in range(nR) if k != i and k != j]
        for six in itertools.combinations(others, 6):
            eight_set = tuple(sorted([i, j] + list(six)))
            if is_q3(adj_bits, eight_set):
                return True
        return False
    
    def build_adj_bits(adj_R):
        bits = []
        for i in range(nR):
            mask = 0
            for j in range(nR):
                if adj_R[i, j]:
                    mask |= (1 << j)
            bits.append(mask)
        return bits
    
    def count_edges(adj_R):
        return int(np.sum(adj_R)) // 2
    
    def full_q3_check(adj_R):
        """Verify no Q3 exists in entire graph (R subgraph + K3,R connections)."""
        adj_bits = build_adj_bits(adj_R)
        # Check all 8-subsets of R
        for eight in itertools.combinations(range(nR), 8):
            if is_q3(adj_bits, eight):
                return False
        # Check Q3s with some L vertices (0,1,2) and some R vertices
        # L vertices are all connected to each other? No, L={0,1,2} has no edges among them.
        # Each L vertex connects to all R vertices.
        # A Q3 with k L vertices and (8-k) R vertices:
        # Since L vertices have no edges among themselves, and each connects to all R,
        # in Q3 each vertex has degree 3.
        # If k=1: the L vertex connects to 3 R vertices. Those 3 R vertices + 4 more R = 7 R total?
        # No, Q3 has 8 vertices. k L + (8-k) R = 8 vertices.
        # k=1: 1 L, 7 R. L connects to 3 of the 7 R. In Q3, each vertex deg 3.
        #   The L vertex has deg 3 (to 3 R). Each of those 3 R vertices needs deg 3 total.
        #   They connect to L (1 edge) + need 2 more from other R vertices.
        #   The other 4 R vertices need deg 3 from R-R edges.
        #   This is complex. Let me just check all combinations.
        
        # For k L vertices (k=1,2,3), check if there's a Q3
        for k in range(1, 4):
            for L_subset in itertools.combinations(range(3), k):
                for R_subset in itertools.combinations(range(nR), 8-k):
                    # Check if induced subgraph is Q3
                    # Build adjacency for these 8 vertices
                    # Vertex indices: L_subset[0..k-1], R_subset[0..8-k-1]
                    total = 8
                    adj8 = np.zeros((total, total), dtype=bool)
                    # L-L edges: none (L is independent)
                    # L-R edges: all present
                    for li in range(k):
                        for ri in range(k, total):
                            adj8[li, ri] = True
                            adj8[ri, li] = True
                    # R-R edges: from adj_R
                    for ri in range(k, total):
                        for rj in range(ri+1, total):
                            if adj_R[R_subset[ri-k], R_subset[rj-k]]:
                                adj8[ri, rj] = True
                                adj8[rj, ri] = True
                    # Check if this is Q3: 12 edges, bipartite 4-4, each deg 3
                    ec = int(np.sum(adj8)) // 2
                    if ec != 12:
                        continue
                    for part1 in itertools.combinations(range(total), 4):
                        part2 = tuple(v for v in range(total) if v not in part1)
                        ok = True
                        for a in part1:
                            deg = sum(1 for b in part2 if adj8[a, b])
                            if deg != 3:
                                ok = False
                                break
                        if not ok:
                            continue
                        for b in part2:
                            deg = sum(1 for a in part1 if adj8[a, b])
                            if deg != 3:
                                ok = False
                                break
                        if ok:
                            return False
        return True
    
    best_edge_count = 0
    best_adj_R = None
    
    for seed in range(20):
        if time.time() > DEADLINE - 10:
            break
        random.seed(seed)
        
        # Start with K3,R complete (all edges between L and R)
        # R subgraph: start with a random sparse graph or specific structure
        adj_R = np.zeros((nR, nR), dtype=bool)
        
        # Initial: add a cycle or matching
        if seed % 2 == 0:
            # Cycle
            for i in range(nR):
                adj_R[i, (i+1) % nR] = True
                adj_R[(i+1) % nR, i] = True
        else:
            # Random sparse
            for i in range(nR):
                for j in range(i+1, nR):
                    if random.random() < 0.15:
                        adj_R[i, j] = True
                        adj_R[j, i] = True
        
        adj_bits = build_adj_bits(adj_R)
        current_edges = count_edges(adj_R)
        
        # Phase 1: Greedy edge addition
        candidates = [(i, j) for i in range(nR) for j in range(i+1, nR) if not adj_R[i, j]]
        random.shuffle(candidates)
        
        for (i, j) in candidates:
            if time.time() > DEADLINE - 15:
                break
            if adj_R[i, j]:
                continue
            adj_R[i, j] = True
            adj_R[j, i] = True
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
            if check_q3_after_add(adj_bits, i, j):
                adj_R[i, j] = False
                adj_R[j, i] = False
                adj_bits[i] &= ~(1 << j)
                adj_bits[j] &= ~(1 << i)
            else:
                current_edges += 1
        
        # Phase 2: Local search with swaps
        for iteration in range(500):
            if time.time() > DEADLINE - 15:
                break
            
            # Try adding a random missing edge
            missing = [(i, j) for i in range(nR) for j in range(i+1, nR) if not adj_R[i, j]]
            if not missing:
                break
            
            added_any = False
            for (i, j) in random.sample(missing, min(len(missing), 20)):
                if time.time() > DEADLINE - 15:
                    break
                adj_R[i, j] = True
                adj_R[j, i] = True
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
                if check_q3_after_add(adj_bits, i, j):
                    # Try to remove an existing edge to make room
                    existing = [(a, b) for a in range(nR) for b in range(a+1, nR) 
                               if adj_R[a, b] and a != i and b != j]
                    if existing:
                        for (a, b) in random.sample(existing, min(len(existing), 10)):
                            if time.time() > DEADLINE - 15:
                                break
                            # Remove (a,b), keep (i,j)
                            adj_R[a, b] = False
                            adj_R[b, a] = False
                            adj_bits[a] &= ~(1 << b)
                            adj_bits[b] &= ~(1 << a)
                            # Check if Q3-free now
                            adj_bits2 = build_adj_bits(adj_R)
                            if not check_q3_after_add(adj_bits2, i, j):
                                added_any = True
                                current_edges += 0  # net zero change (removed one, added one)
                                break
                            else:
                                # Restore
                                adj_R[a, b] = True
                                adj_R[b, a] = True
                                adj_bits[a] |= (1 << b)
                                adj_bits[b] |= (1 << a)
                        if added_any:
                            break
                    if not added_any:
                        # Revert
                        adj_R[i, j] = False
                        adj_R[j, i] = False
                        adj_bits[i] &= ~(1 << j)
                        adj_bits[j] &= ~(1 << i)
                else:
                    current_edges += 1
                    added_any = True
                    break
            
            if not added_any and iteration > 50:
                break
        
        # Track best
        if current_edges > best_edge_count:
            best_edge_count = current_edges
            best_adj_R = adj_R.copy()
    
    # Final verification and edge construction
    if best_adj_R is None:
        best_adj_R = np.zeros((nR, nR), dtype=bool)
    
    # Verify no Q3
    if not full_q3_check(best_adj_R):
        # Fallback: use a known safe construction
        # K3,13 with a perfect matching in R (6 edges)
        best_adj_R = np.zeros((nR, nR), dtype=bool)
        for i in range(0, 12, 2):
            best_adj_R[i, i+1] = True
            best_adj_R[i+1, i] = True
    
    edges = []
    for a in range(3):
        for b in range(3, n):
            edges.append([a, b])
    for i in range(nR):
        for j in range(i+1, nR):
            if best_adj_R[i, j]:
                edges.append([R[i], R[j]])
    return edges
