import numpy as np
import time
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    p1, p2 = 5, 41
    
    # Compute QR sets
    qr5 = set()
    for x in range(1, p1):
        qr5.add((x*x) % p1)
    # qr5 = {1, 4}
    
    qr41 = set()
    for x in range(1, p2):
        qr41.add((x*x) % p2)
    # 20 elements
    
    # Forbidden differences mod 205
    # d is forbidden iff d%5 in {0,1,4} and d%41 in qr41 ∪ {0}, d != 0
    
    # Build conflict graph on Z_205
    # But use CRT structure
    
    # First, find maximum square-difference-free set mod 41
    # Graph: vertices 0..40, edge between x,y if (x-y)%41 in qr41
    # (symmetric since 41 ≡ 1 mod 4, so -1 is QR)
    
    # Verify symmetry
    assert all(((-d) % p2) in qr41 for d in qr41), "QR41 not symmetric"
    assert all(((-d) % p1) in qr5 for d in qr5), "QR5 not symmetric"
    
    # Build adjacency for mod 41
    adj41 = [0] * p2  # bitmask
    for x in range(p2):
        mask = 0
        for d in qr41:
            y = (x + d) % p2
            mask |= (1 << y)
            y2 = (x - d) % p2
            mask |= (1 << y2)
        adj41[x] = mask
    
    # Maximum independent set on 41-vertex graph
    # Use branch and bound with bitsets
    
    def max_independent_set(adj, n):
        """Find maximum independent set using B&B with bitsets"""
        best = [0]  # size
        best_set = [0]  # bitmask
        
        # Greedy coloring for upper bound (Tomita's algorithm)
        def greedy_color(adj, n):
            """Return upper bound on independence number via greedy coloring"""
            # Simple greedy: color vertices in order
            color = [0] * n
            for v in range(n):
                used = set()
                for u in range(v):
                    if adj[v] & (1 << u):
                        used.add(color[u])
                c = 1
                while c in used:
                    c += 1
                color[v] = c
            return max(color)
        
        def dfs(candidates, current, depth):
            if not candidates:
                if len(current) > best[0]:
                    best[0] = len(current)
                    best_set[0] = current
                return
            
            # Upper bound: |candidates| + |current|
            cand_list = [i for i in range(n) if candidates & (1 << i)]
            if len(current) + len(cand_list) <= best[0]:
                return
            
            # Pick a vertex with maximum degree in candidate subgraph
            # Or just pick the first one
            v = cand_list[0]
            
            # Branch 1: include v
            new_candidates = candidates & ~(1 << v) & ~adj[v]
            dfs(new_candidates, current | (1 << v), depth + 1)
            
            # Branch 2: exclude v
            dfs(candidates & ~(1 << v), current, depth + 1)
        
        all_mask = (1 << n) - 1
        dfs(all_mask, 0, 0)
        return best_set[0]
    
    # This naive B&B might be slow for 41 vertices. Let me use a better approach.
    # Actually, let's use a smarter B&B with the maximum degree vertex selection.
    
    def max_independent_set_v2(adj, n):
        best_size = [0]
        best_mask = [0]
        
        def dfs(candidates, current_mask, current_size):
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_mask[0] = current_mask
            
            if candidates == 0:
                return
            
            # Upper bound
            cand_count = bin(candidates).count('1')
            if current_size + cand_count <= best_size[0]:
                return
            
            # Select vertex with max degree in candidate subgraph
            best_v = -1
            best_deg = -1
            for v in range(n):
                if candidates & (1 << v):
                    deg = bin(adj[v] & candidates).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
            
            # Include best_v
            new_cand = candidates & ~(1 << best_v) & ~adj[best_v]
            dfs(new_cand, current_mask | (1 << best_v), current_size + 1)
            
            # Exclude best_v
            dfs(candidates & ~(1 << best_v), current_mask, current_size)
        
        all_mask = (1 << n) - 1
        dfs(all_mask, 0, 0)
        return best_mask[0]
    
    deadline = time.time() + 600.0
    
    # Solve mod 41 MIS
    mis41_mask = max_independent_set_v2(adj41, p2)
    mis41_size = bin(mis41_mask).count('1')
    mis41_list = [i for i in range(p2) if mis41_mask & (1 << i)]
    
    # Now combine using CRT structure
    # For each subset A of Z_5, and assignment of S_a for each a in A,
    # check if the combined set is valid in Z_205.
    
    # The combined set: for a in A, for s in S_a, the element is the CRT combination.
    # CRT: x ≡ a (mod 5), x ≡ s (mod 41)
    # x = a * 41 * inv(41 mod 5, 5) + s * 5 * inv(5 mod 41, 41) mod 205
    # 41 mod 5 = 1, inv(1,5) = 1
    # 5 mod 41 = 5, inv(5,41): 5*? ≡ 1 mod 41. 5*33=165=4*41+1, so inv=33
    # x = a * 41 * 1 + s * 5 * 33 mod 205 = 41a + 165s mod 205
    
    def crt(a, s):
        return (41 * a + 165 * s) % 205
    
    # For the combined set to be valid:
    # 1. Within each row a: S_a must be square-diff-free mod 41 (already satisfied by mis41)
    # 2. Between rows a, a': if (a-a')%5 in {1,4}, then for all s in S_a, t in S_a', (s-t)%41 must be non-square mod 41
    #    If (a-a')%5 in {2,3}, no constraint.
    
    # To maximize size, I want to use the largest possible S_a sets.
    # The best case: use A = {0, 2} where (0-2)%5 = 3 ∈ {2,3}, so no cross-constraint.
    # Then I can use the maximum MIS for both rows: size = 2 * mis41_size
    
    # But can I do better with a different A?
    # If A has size 2 with diff in {2,3}: 2 * mis41_size
    # If A has size 2 with diff in {1,4}: need S_0 and S_1 such that all cross-diffs are non-squares
    # If A has size 1: mis41_size
    # If A has size 3: need all pairwise diffs in {2,3}, impossible as shown
    
    # So the best is likely 2 * mis41_size with A = {0,2} or {0,3} etc.
    
    # Let's also try: can I use different MIS sets for different rows to allow larger total?
    # For A = {0, 2} (diff 3, non-square mod 5): no cross constraint, so use max MIS for both.
    # Total = 2 * mis41_size
    
    # Let's verify this is valid and compute the set
    A = [0, 2]
    S = []
    for a in A:
        for s in mis41_list:
            S.append(crt(a, s))
    
    # Verify
    forbidden205 = set()
    for x in range(1, m):
        forbidden205.add(pow(x, 2, m))
    forbidden205.discard(0)
    
    S_set = set(S)
    valid = True
    for i, a in enumerate(S):
        for b in S[i+1:]:
            d = (a - b) % m
            if d in forbidden205:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        return np.array(S, dtype=np.int64)
    
    # Fallback: just the MIS mod 41 lifted
    S = [crt(0, s) for s in mis41_list]
    return np.array(S, dtype=np.int64)
