import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    n = 16
    
    # Q3 cube structure
    # Vertices labeled 0-7 as 3-bit binary: 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
    # Two bipartitions:
    #   Even parity (0,2,2,2 ones): {0, 3, 5, 6}
    #   Odd parity (1,1,1,3 ones):  {1, 2, 4, 7}
    # Edges connect vertices differing in exactly one bit.
    
    # Bipartite adjacency: for each vertex in even part, which odd-part vertices it connects to
    # 0=000 -> 1=001, 2=010, 4=100
    # 3=011 -> 1=001, 2=010, 7=111
    # 5=101 -> 1=001, 4=100, 7=111
    # 6=110 -> 2=010, 4=100, 7=111
    
    # We'll represent the bipartite adjacency as a 4x4 matrix where
    # rows = even part vertices [0,3,5,6] (indices 0,1,2,3)
    # cols = odd part vertices [1,2,4,7] (indices 0,1,2,3)
    
    # Matrix (1 if edge exists):
    #        odd0(1)  odd1(2)  odd2(4)  odd3(7)
    # even0(0):   1       1       1       0
    # even1(3):   1       1       0       1
    # even2(5):   1       0       1       1
    # even3(6):   0       1       1       1
    
    # This is the 4x4 all-ones matrix minus the identity!
    
    def has_q3(adj_bits, n=16):
        """
        Check if graph contains Q3 as a subgraph.
        adj_bits[u] is a bitmask of neighbors of u.
        Returns True if Q3 found.
        """
        # For each 8-subset, check if it contains Q3.
        # Q3 is bipartite 4+4. For each 8-subset S, try each 4-subset A of S as one part.
        # B = S \ A.
        # Check if there's a bijection f: {0,1,2,3} -> A and g: {0,1,2,3} -> B
        # such that for all i,j: edge(f(i), g(j)) exists iff matrix[i][j] = 1.
        
        # The matrix is J - I (all ones minus identity).
        # So we need: for each i, f(i) is connected to exactly g(j) for j != i,
        # and NOT connected to g(i).
        
        # Equivalently: for each i, f(i) is connected to all of B except g(i).
        
        # Strategy: for each 8-subset, for each 4-subset A (and B = complement):
        #   For each vertex a in A, compute its neighbors in B. 
        #   We need to assign g such that for each i, g(i) is the unique vertex in B
        #   that is NOT a neighbor of f(i).
        #   So: f(i) must have exactly 3 neighbors in B (all except one).
        #   And the "missing" neighbor must be distinct for each i.
        
        # More precisely:
        #   For each a in A: let N_B(a) = {b in B : edge(a,b)}.
        #   We need |N_B(a)| = 3 for all a in A (i.e., a is connected to exactly 3 of 4 vertices in B).
        #   And the missing vertex m(a) = B \ N_B(a) must be a bijection from A to B.
        
        # So the check is:
        #   1. For all a in A, a has exactly 3 neighbors in B.
        #   2. The map a -> (the unique b in B not adjacent to a) is a bijection.
        
        # This is much faster than trying all 576 permutations!
        
        for S in combinations(range(n), 8):
            S_set = set(S)
            for A in combinations(S, 4):
                A_set = set(A)
                B = tuple(v for v in S if v not in A_set)
                B_set = set(B)
                
                # Check condition 1: each a in A has exactly 3 neighbors in B
                missing = []
                valid = True
                for a in A:
                    # Count neighbors of a in B
                    cnt = 0
                    miss = -1
                    for b in B:
                        if adj_bits[a] & (1 << b):
                            cnt += 1
                        else:
                            miss = b
                    if cnt != 3:
                        valid = False
                        break
                    missing.append(miss)
                
                if not valid:
                    continue
                
                # Check condition 2: missing is a permutation of B
                if sorted(missing) == sorted(B):
                    return True
        
        return False
    
    def has_q3_fast(adj_bits):
        """
        Optimized Q3 checker using the structural insight.
        """
        n = 16
        for S in combinations(range(n), 8):
            S_set = set(S)
            for A in combinations(S, 4):
                A_set = set(A)
                B = [v for v in S if v not in A_set]
                
                missing = []
                valid = True
                for a in A:
                    cnt = 0
                    miss = -1
                    for b in B:
                        if adj_bits[a] & (1 << b):
                            cnt += 1
                        else:
                            miss = b
                    if cnt != 3:
                        valid = False
                        break
                    missing.append(miss)
                
                if valid and len(set(missing)) == 4:
                    return True
        
        return False
    
    # Start with K3,13 (39 edges) - definitely Q3-free
    # Left: {0,1,2}, Right: {3,...,15}
    adj_bits = [0] * n
    edges = []
    for a in range(3):
        for b in range(3, 16):
            adj_bits[a] |= (1 << b)
            adj_bits[b] |= (1 << a)
            edges.append([a, b])
    
    # Try to add edges within the right side {3,...,15}
    # and between left and right (already complete)
    # and within left (none currently)
    
    # Actually, let's try a broader approach:
    # Start with various initial graphs and do local search.
    
    best_edges = edges[:]
    best_count = len(edges)
    
    # Try adding edges one by one
    # All possible edges: C(16,2) = 120
    all_possible = list(combinations(range(n), 2))
    current_edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    
    # Try to add edges in random order
    import random
    random.seed(42)
    
    # First, try all edges that don't create Q3
    # For speed, we'll check Q3 only when adding an edge
    # The key insight: when adding edge (u,v), we only need to check 8-subsets containing both u and v
    # But that's still C(14,6) = 3003 subsets per edge, times the inner check...
    
    # Let's use a smarter approach: 
    # For adding edge (u,v), a Q3 containing this edge must have u and v as adjacent vertices in the cube.
    # In Q3, adjacent vertices differ in exactly one bit. So u and v must be in different bipartition parts.
    # 
    # Let's try: for each possible new edge (u,v) not already present,
    # check if adding it creates a Q3.
    
    # To make the check fast when adding edge (u,v):
    # A Q3 containing edge (u,v): u and v are adjacent in the cube.
    # WLOG, say u is in even part, v is in odd part (or vice versa).
    # The even part has 4 vertices including u, the odd part has 4 including v.
    # 
    # Let's enumerate: for each 4-subset A of {0..15} containing u (if u is even) or v (if v is even),
    # and 4-subset B containing the other, check if the bipartite graph between A and B 
    # (in the graph WITH the new edge) is isomorphic to Q3.
    
    # Actually, let's just use the full checker but only on 8-subsets containing {u,v}.
    # C(14,6) = 3003 subsets. For each, C(8,4)=70 ways to split. For each split, O(4*4)=16 work.
    # Total: 3003 * 70 * 16 ≈ 3.4M operations per edge. For 120 edges, that's 400M. Too slow.
    
    # Better: precompute which 8-subsets could contain a Q3 given the current graph.
    # Or: use the structural property more directly.
    
    # Alternative approach: use a different initial construction that's denser.
    
    # Let me try: what if we use a 4-partite structure?
    # Or: try to find the maximum Q3-free graph via iterative edge addition with a fast checker.
    
    # Let me optimize the Q3 checker significantly.
    
    def has_q3_incremental(adj_bits, u, v):
        """
        Check if adding edge (u,v) to the graph creates a Q3.
        Only checks 8-subsets containing both u and v.
        """
        n = 16
        others = [x for x in range(n) if x != u and x != v]
        
        for S6 in combinations(others, 6):
            S = (u, v) + S6
            S_set = set(S)
            
            for A in combinations(S, 4):
                A_set = set(A)
                B = tuple(x for x in S if x not in A_set)
                
                # Check if u and v are in different parts (required for them to be adjacent in Q3)
                u_in_A = u in A_set
                v_in_A = v in A_set
                if u_in_A == v_in_A:
                    continue  # u and v in same part, can't be adjacent in bipartite Q3
                
                # Now check the structural condition
                missing = []
                valid = True
                for a in A:
                    cnt = 0
                    miss = -1
                    for b in B:
                        # Check if edge (a,b) exists (including the new edge if applicable)
                        if a == u and b == v or a == v and b == u:
                            cnt += 1  # new edge
                        elif adj_bits[a] & (1 << b):
                            cnt += 1
                        else:
                            miss = b
                    if cnt != 3:
                        valid = False
                        break
                    missing.append(miss)
                
                if valid and len(set(missing)) == 4:
                    return True
        
        return False
    
    # Now do local search
    # Try to add edges greedily
    for u, v in all_possible:
        if time.time() > deadline - 5:
            break
        if (min(u,v), max(u,v)) in current_edge_set:
            continue
        if has_q3_incremental(adj_bits, u, v):
            continue
        # Add the edge
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        current_edge_set.add((min(u,v), max(u,v)))
        edges.append([u, v])
    
    # Now try to add more edges (second pass, different order)
    random.shuffle(all_possible)
    for u, v in all_possible:
        if time.time() > deadline - 5:
            break
        if (min(u,v), max(u,v)) in current_edge_set:
            continue
        if has_q3_incremental(adj_bits, u, v):
            continue
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        current_edge_set.add((min(u,v), max(u,v)))
        edges.append([u, v])
    
    # Third pass
    random.seed(123)
    random.shuffle(all_possible)
    for u, v in all_possible:
        if time.time() > deadline - 5:
            break
        if (min(u,v), max(u,v)) in current_edge_set:
            continue
        if has_q3_incremental(adj_bits, u, v):
            continue
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        current_edge_set.add((min(u,v), max(u,v)))
        edges.append([u, v])
    
    # Fourth pass
    random.seed(456)
    random.shuffle(all_possible)
    for u, v in all_possible:
        if time.time() > deadline - 5:
            break
        if (min(u,v), max(u,v)) in current_edge_set:
            continue
        if has_q3_incremental(adj_bits, u, v):
            continue
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        current_edge_set.add((min(u,v), max(u,v)))
        edges.append([u, v])
    
    # Final verification
    if has_q3_fast(adj_bits):
        # Fallback to K3,13
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return edges
