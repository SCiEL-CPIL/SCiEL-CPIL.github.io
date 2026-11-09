import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    A = list(range(8))
    B = list(range(8, 16))
    
    adj = [0] * n
    
    for u in A:
        for v in B:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    def get_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask ^= lsb
        return edges

    def contains_q3(S, adj):
        # S is a tuple of 8 vertices
        # Precompute masks for all 4-subsets of S?
        # There are 70 such subsets.
        # For each L, R is the complement.
        # We can just iterate.
        for L in combinations(S, 4):
            L_set = set(L)
            R = tuple(v for v in S if v not in L_set)
            
            # Calculate mask for R
            mask_R = 0
            for v in R:
                mask_R |= (1 << v)
            
            ok = True
            for u in L:
                d = bin(adj[u] & mask_R).count('1')
                if d < 3:
                    ok = False
                    break
            if not ok:
                continue
                
            mask_L = 0
            for u in L:
                mask_L |= (1 << u)
                
            for v in R:
                d = bin(adj[v] & mask_L).count('1')
                if d < 3:
                    ok = False
                    break
            if ok:
                return True
        return False

    def find_q3(adj):
        for S in combinations(range(n), 8):
            if contains_q3(S, adj):
                return S
        return None

    # Phase 1: Remove edges to destroy Q3s
    # Randomized removal to avoid local minima?
    # Just find one and remove a random edge from it.
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        q3 = find_q3(adj)
        if q3 is None:
            break
        # Remove a random edge from the Q3
        # The Q3 is "contained" in q3.
        # We can pick any edge that is part of the 3-regular subgraph.
        # But we don't know which specific edges form the Q3.
        # However, removing ANY edge from the induced subgraph G[q3] might not break the Q3
        # if the Q3 uses other edges.
        # But if we remove an edge that is in ALL possible Q3s in q3, it's safe.
        # Simpler: Just remove a random edge from the complete set of edges in G[q3]?
        # No, that might not break the specific Q3.
        
        # Better: Identify a specific Q3 subgraph.
        # My contains_q3 returns True if there EXISTS a split.
        # We need to find the actual edges.
        
        # Let's modify contains_q3 to return the edges of the Q3?
        # Or just remove a random edge from the 12 edges of a specific Q3.
        
        # Let's re-implement to find one specific Q3 edge set.
        
        break # Placeholder, need to fix logic

    # Re-implementation of find_q3 to return edges
    def find_q3_edges(adj):
        for S in combinations(range(n), 8):
            # Check splits
            for L in combinations(S, 4):
                L_set = set(L)
                R = tuple(v for v in S if v not in L_set)
                mask_R = 0
                for v in R:
                    mask_R |= (1 << v)
                
                ok = True
                for u in L:
                    if bin(adj[u] & mask_R).count('1') < 3:
                        ok = False
                        break
                if not ok: continue
                
                mask_L = 0
                for u in L:
                    mask_L |= (1 << u)
                
                for v in R:
                    if bin(adj[v] & mask_L).count('1') < 3:
                        ok = False
                        break
                if not ok: continue
                
                # We found a split L, R where min degree >= 3.
                # We need to extract a 3-regular subgraph.
                # This is a bipartite graph with min degree 3.
                # We can find a perfect matching in the complement?
                # Or just greedily pick edges.
                # Since it's small, we can just return the list of all edges between L and R?
                # No, we need to remove ONE edge that breaks this specific Q3.
                # Actually, if we remove ANY edge from the set of edges between L and R,
                # does it break the Q3?
                # Not necessarily. The Q3 might not use that edge.
                
                # To be safe, we can find the specific 3-regular subgraph.
                # But for simplicity, let's just remove a random edge between L and R
                # and hope it breaks it?
                # No, that's risky.
                
                # Let's just return the first edge found in the dense part?
                # Actually, we can just remove a random edge from the entire graph
                # and re-check.
                
                # Let's return a random edge from L x R that exists.
                possible_edges = []
                for u in L:
                    for v in R:
                        if (adj[u] >> v) & 1:
                            possible_edges.append((u, v))
                if possible_edges:
                    return random.choice(possible_edges)
        return None

    # Phase 1
    for _ in range(5000):
        if time.time() > deadline:
            break
        edge = find_q3_edges(adj)
        if edge is None:
            break
        u, v = edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Phase 2: Add intra edges
    def creates_q3_with_edge(u, v, adj):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        other_vertices = [x for x in range(n) if x != u and x != v]
        for S6 in combinations(other_vertices, 6):
            S = tuple(list(S6) + [u, v])
            if contains_q3(S, adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                return True
        return False

    intra_pairs = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            intra_pairs.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            intra_pairs.append((B[i], B[j]))
            
    random.shuffle(intra_pairs)
    
    for u, v in intra_pairs:
        if time.time() > deadline:
            break
        if not creates_q3_with_edge(u, v, adj):
            pass # Edge kept

    # Phase 3: Add back bipartite edges
    bipartite_pairs = []
    for u in A:
        for v in B:
            if not ((adj[u] >> v) & 1):
                bipartite_pairs.append((u, v))
                
    random.shuffle(bipartite_pairs)
    
    for u, v in bipartite_pairs:
        if time.time() > deadline:
            break
        if not creates_q3_with_edge(u, v, adj):
            pass # Edge kept

    return get_edges(adj)
