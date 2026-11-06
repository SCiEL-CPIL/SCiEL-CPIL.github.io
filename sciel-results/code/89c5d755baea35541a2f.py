import time
from math import gcd

def construct():
    deadline = time.time() + 85
    n = 255
    
    # Candidates: numbers in [1,255] with gcd(v,255)>1
    # i.e., divisible by 3, 5, or 17
    candidates = [v for v in range(1, n+1) if v % 3 == 0 or v % 5 == 0 or v % 17 == 0]
    
    m = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency matrix as bitmasks
    # adj_mask[i] is an integer with bit j set if i and j are adjacent
    adj_mask = [0] * m
    for i in range(m):
        for j in range(i+1, m):
            if gcd(candidates[i], candidates[j]) > 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    # We must include 255.
    # The clique must be a subset of {255} U neighbors(255).
    # Let's find the maximum clique in the graph induced by neighbors(255) plus 255.
    # Since 255 is connected to all candidates (by definition of candidates),
    # any clique in the candidate graph that includes 255 is just a clique in the candidate graph
    # that includes the index of 255.
    # Actually, 255 is connected to ALL candidates because candidates are defined as having gcd > 1 with 255.
    # So we just need to find the maximum clique in the graph of `candidates` that includes `idx_255`.
    # This is equivalent to finding the maximum clique in the graph induced by `adj_mask[idx_255]` and adding 255.
    
    # Let P be the set of nodes compatible with 255 (which is all other nodes).
    # We want to find a max clique in the subgraph induced by P, and add 255 to it.
    # Size = 1 + max_clique_size(P).
    
    # Use a branch and bound algorithm for maximum clique with bitmasks.
    # Reference: Tomita's algorithm or similar.
    
    best_size = 0
    best_clique_mask = 0
    
    # Initial greedy heuristic to find a good lower bound
    # Greedy: pick node with max degree in remaining set
    def greedy_clique(P):
        clique = 0
        while P:
            # Pick v with max popcount(adj_mask[v] & P)
            # Since m is small (127), we can iterate
            best_v = -1
            best_deg = -1
            # Optimization: just pick the first one? No, greedy by degree is better.
            # To speed up, we can just pick a random one or the one with highest index?
            # Let's try to pick the one with max connections.
            # Iterate through bits in P
            p = P
            while p:
                # Extract lowest set bit
                lsb = p & -p
                v = lsb.bit_length() - 1
                deg = (adj_mask[v] & P).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                p ^= lsb
            
            if best_v == -1:
                break
            clique |= (1 << best_v)
            P &= adj_mask[best_v]
        return clique
    
    # Run greedy on the full set of candidates (excluding 255 for a moment, or including it? 
    # The graph includes 255. Let's run greedy on the full graph but force 255?
    # Actually, since 255 is connected to all, we can just find max clique in the full graph.
    # But we MUST include 255.
    # So we find max clique in the subgraph induced by neighbors of 255.
    # Neighbors of 255 are all other candidates.
    
    P_initial = adj_mask[idx_255]
    greedy_res = greedy_clique(P_initial)
    best_size = greedy_res.bit_count()
    best_clique_mask = greedy_res
    
    # Branch and Bound
    # We search for a clique in P_initial larger than best_size.
    
    def max_clique_search(P, X, current_clique, depth):
        nonlocal best_size, best_clique_mask
        if time.time() > deadline - 5:
            return
        
        # If P is empty, check if we have a better clique
        if not P:
            if depth > best_size:
                best_size = depth
                best_clique_mask = current_clique
            return
        
        # Upper bound: depth + popcount(P)
        if depth + P.bit_count() <= best_size:
            return
            
        # Pivot selection
        # Choose u in P U X that maximizes |P \cap N(u)|
        # To simplify, we can just pick any u in P or X.
        # Standard Tomita uses P U X.
        # Here X is the set of vertices already processed (not in P, not in clique).
        # For simplicity and speed, let's just iterate P.
        
        # Optimization: Order vertices in P
        # We can sort vertices in P by degree or just iterate.
        
        # Pivot u
        if P:
            # Pick u from P U X with max degree in P
            # Since X is usually small or we can ignore it for a simpler bound, 
            # let's just pick u from P.
            u = (P & -P).bit_length() - 1
            Q = P & ~adj_mask[u]
        else:
            Q = P
            
        # Iterate through Q
        # If Q is empty, we must iterate P? No, if Q is empty, it means u is connected to all P.
        # Then we can just add u and recurse on P \cap N(u)?
        # The Bron-Kerbosch with pivot says: iterate v in Q.
        # If Q is empty, the loop does nothing? No, that's not right.
        # If Q is empty, it means N(u) contains P.
        # Then we can just pick any v in P?
        # Actually, if Q is empty, we can just proceed with the first element of P?
        # Let's stick to the standard: iterate v in Q.
        # If Q is empty, we might miss cliques?
        # No, if Q is empty, it means u is adjacent to all vertices in P.
        # Any maximal clique must contain u or not?
        # If a maximal clique C is in P and doesn't contain u, it must be a clique in P \ N(u) = Q.
        # If Q is empty, there are no such cliques. So any maximal clique must contain u.
        # So we can just force u into the clique.
        
        if not Q:
            # Force u
            v = u
            new_P = P & adj_mask[v]
            new_X = X & adj_mask[v]
            max_clique_search(new_P, new_X, current_clique | (1 << v), depth + 1)
            return
            
        for v_bit in list(Q):
            # Extract v
            v = v_bit.bit_length() - 1
            if not (P & (1 << v)):
                continue
                
            new_P = P & adj_mask[v]
            new_X = X & adj_mask[v]
            
            max_clique_search(new_P, new_X, current_clique | (1 << v), depth + 1)
            
            P &= ~(1 << v)
            X |= (1 << v)
            
            # Re-calculate Q? 
            # In Tomita, Q is recalculated or maintained.
            # For simplicity, we re-calculate the loop condition?
            # Actually, the standard algorithm iterates over a list generated from Q.
            # But as P changes, Q changes.
            # A simpler version:
            # While P:
            #   v = pick from P
            #   ...
            #   P.remove(v)
            # This is O(N^2) per level which is fine for N=127.
            
    # Restart with a simpler loop structure for clarity and correctness
    def search(P, X, clique, depth):
        nonlocal best_size, best_clique_mask
        if time.time() > deadline - 5:
            return
            
        if not P:
            if depth > best_size:
                best_size = depth
                best_clique_mask = clique
            return
            
        if depth + P.bit_count() <= best_size:
            return
            
        # Pivot
        # Select u from P U X
        # For speed, just pick first from P
        u_bit = P & -P
        u = u_bit.bit_length() - 1
        Q = P & ~adj_mask[u]
        
        # If Q is empty, all in P are neighbors of u.
        # We can just add u?
        # If Q is empty, any maximal clique in P must contain u?
        # Yes, because if a clique C in P does not contain u, C must be in P \ N(u) = Q.
        # If Q is empty, no such C exists.
        if not Q:
            search(P & adj_mask[u], X & adj_mask[u], clique | u_bit, depth + 1)
            return
            
        # Iterate over Q
        # We need to be careful with the loop as P changes.
        # We iterate over a snapshot of Q?
        # No, if we pick v in Q, we remove v from P.
        # The other elements in Q remain in P?
        # Yes.
        # But the set of candidates for the next pivot might change.
        
        # To avoid complex updates, we can just iterate through the bits of Q.
        # But we must ensure we only process v if it is still in P.
        
        q = Q
        while q:
            v_bit = q & -q
            v = v_bit.bit_length() - 1
            q ^= v_bit
            
            if not (P & v_bit):
                continue
                
            search(P & adj_mask[v], X & adj_mask[v], clique | v_bit, depth + 1)
            
            P &= ~v_bit
            X |= v_bit
            
            # Early termination?
            # If P is empty, break?
            # The while loop handles P check? No, we check `if not (P & v_bit)`.
            # We should also check if depth + P.bit_count() <= best_size inside the loop?
            # The recursive call will handle it.
            
    # Initial call
    # P is all neighbors of 255
    # X is empty
    # Clique is empty (we add 255 at the end)
    search(adj_mask[idx_255], 0, 0, 0)
    
    # Construct result
    result_indices = []
    mask = best_clique_mask
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        result_indices.append(idx)
        mask ^= lsb
        
    result_indices.append(idx_255)
    
    result = sorted([candidates[i] for i in result_indices])
    
    # Verify
    assert 255 in result
    assert all(1 <= x <= 255 for x in result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            assert gcd(result[i], result[j]) > 1
            
    return result
