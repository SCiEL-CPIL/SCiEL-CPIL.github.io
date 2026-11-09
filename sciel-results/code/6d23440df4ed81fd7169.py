import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    p1, p2 = 5, 41
    
    # 1. Compute forbidden differences (quadratic residues)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency bitsets for G (edges where diff is in forbidden)
    adj = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i != j and (i - j) % m in forbidden:
                mask |= (1 << j)
        adj[i] = mask
    
    # 3. Precompute a static ordering.
    degrees = [bin(adj[i]).count('1') for i in range(m)]
    order = list(range(m))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    best_size = 0
    best_set = 0
    deadline = time.time() + 590 # 10s margin
    
    # CRT-based construction:
    # Z_205 ≅ Z_5 × Z_41
    # A difference d mod 205 is forbidden iff d is a non-zero square mod 205.
    # By CRT, d is a square mod 205 iff d mod 5 is a square mod 5 AND d mod 41 is a square mod 41.
    # Non-zero squares mod 5: {1, 4}
    # Non-zero squares mod 41: {1, 4, 5, 9, 16, 25, 36} (7 elements)
    
    # Strategy: Choose a set S in Z_205 such that for any a, b in S:
    # (a-b) mod 5 is NOT in {1,4} OR (a-b) mod 41 is NOT in {1,4,5,9,16,25,36}
    
    # Simple construction: Fix the Z_5 component to a single value, and choose Z_41 values
    # such that no two differ by a square mod 41.
    # If we fix a mod 5 = c, then (a-b) mod 5 = 0, which is NOT in {1,4}.
    # So ANY set of values with the same residue mod 5 is valid!
    # We can choose all 41 values with a given residue mod 5.
    
    # Let's verify: If a ≡ b (mod 5), then (a-b) ≡ 0 (mod 5).
    # 0 is not a non-zero square mod 5, so (a-b) cannot be a square mod 205
    # (since a square mod 205 must reduce to a square mod 5, and 0 is only 0^2 mod 5,
    #  but we need (a-b) mod 205 ≠ 0 for it to be in the forbidden set).
    # Wait, the forbidden set removes 0. So if (a-b) ≡ 0 mod 5, then (a-b) mod 205 is either 0
    # or a multiple of 5. If it's 0, it's not forbidden. If it's non-zero multiple of 5,
    # say 5k, then (5k)^2 = 25k^2 mod 205. Is 5k a square mod 205?
    # For 5k to be a square mod 205, we need 5k ≡ x^2 mod 205 for some x.
    # Then x^2 ≡ 5k mod 5 => 0 ≡ 0, ok.
    # x^2 ≡ 5k mod 41. So 5k must be a square mod 41.
    # So multiples of 5 CAN be squares mod 205 if k*5 is a square mod 41.
    
    # So fixing mod 5 is NOT sufficient. We need to be more careful.
    
    # Better: Fix mod 41 to a single value. If a ≡ b (mod 41), then (a-b) ≡ 0 (mod 41).
    # 0 is not in the non-zero squares mod 41. So (a-b) cannot be a square mod 205.
    # Because if d is a square mod 205, then d mod 41 must be a square mod 41 (including 0? No, 
    # non-zero squares mod 205 reduce to squares mod 41, but could be 0 mod 41? 
    # If d ≡ 0 mod 41 and d is a square mod 205, then d = 41k. (41k)^2 = 1681k^2 ≡ 0 mod 205? 
    # 1681 = 8*205 + 31, so 1681 ≡ 31 mod 205. So (41k)^2 ≡ 31k^2 mod 205.
    # For 41k to be a square mod 205, we need 41k ≡ x^2 mod 205.
    # Then x^2 ≡ 41k mod 41 => 0 ≡ 0.
    # x^2 ≡ 41k mod 5 => x^2 ≡ k mod 5. So k must be a square mod 5.
    # So multiples of 41 CAN be squares mod 205.
    
    # Hmm, so fixing one component isn't enough. Let me think differently.
    
    # The forbidden set F = {x^2 mod 205 : x=1..204} \ {0}.
    # |F| = ? Let's compute.
    
    # Actually, let's just use the branch-and-bound with the existing approach,
    # but add a CRT-based initial solution to seed the search.
    
    # Initial solution: Try to find a large set via greedy with CRT insights.
    # For now, just run B&B.
    
    def greedy_color_upper_bound(candidates):
        if candidates == 0:
            return 0
        remaining = candidates
        colors = 0
        while remaining:
            colors += 1
            v = -1
            for u in order:
                if (remaining >> u) & 1:
                    v = u
                    break
            if v == -1:
                break
            clique = 1 << v
            compatible = remaining & adj[v]
            while compatible:
                u = -1
                for w in order:
                    if (compatible >> w) & 1:
                        u = w
                        break
                if u == -1:
                    break
                clique |= (1 << u)
                compatible &= adj[u]
            remaining &= ~clique
        return colors

    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        ub = greedy_color_upper_bound(candidates)
        if current_size + ub <= best_size:
            return
        
        v = -1
        for u in order:
            if (candidates >> u) & 1:
                v = u
                break
        
        if v == -1:
            return
        
        include_candidates = candidates & ~(adj[v] | (1 << v))
        branch_and_bound(include_candidates, current_set | (1 << v), current_size + 1)
        
        exclude_candidates = candidates & ~(1 << v)
        branch_and_bound(exclude_candidates, current_set, current_size)

    all_mask = (1 << m) - 1
    branch_and_bound(all_mask, 0, 0)
    
    result = []
    for i in range(m):
        if (best_set >> i) & 1:
            result.append(i)
            
    if not result:
        result = [0]
        
    return np.array(result, dtype=np.int64)
