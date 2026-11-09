import time, math, random

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    def now(): return time.time()
    
    all_m = list(range(2, 129))
    n = len(all_m)
    
    # Precompute gcd matrix
    G = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(all_m[i], all_m[j])
            G[i][j] = g
            G[j][i] = g
            
    # Precompute conflict graph:
    # Two moduli i, j are "conflicting" if they cannot coexist in any valid set.
    # This happens if:
    # 1. gcd(m_i, m_j) == 1 (coprime), OR
    # 2. The constraints on residues are so tight that no pair of residues exists.
    #    Specifically, for g = gcd(m_i, m_j), we need r_i % g != r_j % g.
    #    This is always possible if g >= 2.
    #    Wait, is it?
    #    Let's check m=2, m=4. g=2.
    #    r_2 in {0,1}, r_4 in {0,1,2,3}.
    #    Need r_2 % 2 != r_4 % 2.
    #    If r_2=0, r_4 must be odd (1 or 3). Possible.
    #    If r_2=1, r_4 must be even (0 or 2). Possible.
    #    So {2,4} is compatible.
    #    
    #    What about m=2, m=6? g=2.
    #    r_2 in {0,1}, r_6 in {0..5}.
    #    Need r_2 % 2 != r_6 % 2.
    #    Possible.
    #    
    #    What about m=4, m=6? g=2.
    #    r_4 in {0..3}, r_6 in {0..5}.
    #    Need r_4 % 2 != r_6 % 2.
    #    Possible.
    #    
    #    So it seems pairwise compatibility only fails if gcd == 1?
    #    Let's check if there are cases where gcd > 1 but no residues exist.
    #    We need r_i % g != r_j % g.
    #    Since r_i can be any value in [0, m_i), r_i % g can be any value in [0, g)
    #    (because m_i is a multiple of g, so the residues modulo g are uniform).
    #    Same for r_j.
    #    So we just need to pick a != b from [0, g).
    #    This is possible iff g >= 2.
    #    
    #    CONCLUSION: Two moduli are compatible iff gcd(m_i, m_j) >= 2.
    #    
    #    This simplifies the problem significantly!
    #    The problem reduces to: Find the largest subset of {2, ..., 128} such that
    #    every pair has gcd >= 2.
    #    Then, for any such subset, we can ALWAYS find valid residues.
    #    (We can even find them greedily or with a simple backtracking, but the
    #    existence is guaranteed by the pairwise argument above).
    
    # So the core problem is: Maximize |S| such that for all i, j in S, gcd(m_i, m_j) >= 2.
    # This is equivalent to finding the largest subset of {2, ..., 128} with no coprime pair.
    
    # Let's build the graph of "coprime conflicts".
    # Node i is connected to node j if gcd(m_i, m_j) == 1.
    # We want the Maximum Independent Set of this graph.
    
    # Since n=127, we can use a branch and bound solver for MIS.
    # However, 2^127 is too big. We need a good heuristic.
    
    # Observation: All even numbers have gcd >= 2 with each other.
    # Count of evens in 2..128: 63.
    # Can we add any odd numbers?
    # An odd number m must share a factor with ALL other numbers in the set.
    # If the set contains an even number not divisible by any prime factor of m,
    # then gcd(m, even) might be 1.
    
    # Let's use a greedy approach to find a large independent set, then try to improve.
    # Or better, use a specific structure.
    
    # The largest such set is likely the set of all even numbers.
    # Are there any odd numbers that can be added to the set of all evens?
    # If we add an odd number m, it must not be coprime with any even number in the set.
    # Since the set contains 2, m must share a factor with 2. But m is odd, so gcd(m,2)=1.
    # Therefore, NO odd number can be added to a set containing 2.
    # 
    # What if we remove 2?
    # Then we can potentially add odd numbers that are not coprime with the remaining evens.
    # But removing 2 costs 1. Can we gain more?
    # If we remove 2, we have evens {4, 6, ..., 128}.
    # Can we add 3?
    # 3 must not be coprime with any even in {4, 6, ...}.
    # gcd(3, 4) = 1. So 3 cannot be added if 4 is in the set.
    # If we remove 4 as well?
    # This suggests the set of all evens is a strong local optimum.
    
    # Let's verify if there's a larger set.
    # By Turan's theorem or similar, the largest subset of {1..N} with no coprime pair
    # is usually the set of numbers divisible by the smallest prime (2).
    # Size approx N/2.
    # For N=128, size 64 (evens 2..128).
    # Actually, 2, 4, ..., 128 is 63 numbers.
    # Wait, 128/2 = 64. The evens are 2, 4, ..., 128.
    # 2*1, 2*2, ..., 2*64.
    # So there are 64 even numbers.
    
    # Let's double check if we can do better by mixing.
    # Suppose we take all multiples of 3.
    # 3, 6, 9, ..., 126.
    # Count: floor(128/3) = 42.
    # 42 < 64.
    
    # What about a mix?
    # The set of all evens is size 64.
    # Is it possible to have a set of size 65?
    # If we have 65 numbers, by Pigeonhole Principle on prime factors?
    # Not directly.
    
    # Let's assume the maximum is 64 (all evens).
    # Now we just need to construct the residues for these 64 moduli.
    # We know a solution exists. We need to find it.
    
    # Residue assignment strategy:
    # We need r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j) for all pairs.
    # Since all m_i are even, gcd(m_i, m_j) is at least 2.
    
    # Let's try a simple backtracking for the residues, given the set of moduli.
    # Order: descending modulus (more freedom first) or ascending?
    # Ascending modulus has fewer choices, so it might prune faster?
    # Or descending?
    
    # Let's try to assign residues for the set of all evens.
    # S = {2, 4, 6, ..., 128}
    
    def solve_residues(moduli, G, all_m):
        """
        Find residues for the given moduli (indices into all_m) such that
        r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j) for all pairs.
        Returns list of (index, residue) or None if failed.
        """
        # Order moduli by increasing size (fewer choices first)
        # This is often better for CSPs
        ordered = sorted(moduli, key=lambda i: all_m[i])
        
        assigned = {}
        
        def backtrack(pos):
            if pos == len(ordered):
                return True
            
            i = ordered[pos]
            m_i = all_m[i]
            
            # Try all residues for i
            for r in range(m_i):
                # Check against all previously assigned
                ok = True
                for j, r_j in assigned.items():
                    g = G[i][j]
                    if (r % g) == (r_j % g):
                        ok = False
                        break
                
                if ok:
                    assigned[i] = r
                    if backtrack(pos + 1):
                        return True
                    del assigned[i]
            
            return False
        
        if backtrack(0):
            return [(i, assigned[i]) for i in ordered]
        return None

    # First, let's try the set of all evens.
    even_indices = [i for i in range(n) if all_m[i] % 2 == 0]
    
    # Verify pairwise gcd >= 2 for evens
    valid_set = True
    for a in range(len(even_indices)):
        for b in range(a+1, len(even_indices)):
            i = even_indices[a]
            j = even_indices[b]
            if G[i][j] < 2:
                valid_set = False
                break
        if not valid_set:
            break
    
    if valid_set:
        res = solve_residues(even_indices, G, all_m)
        if res:
            # Verify
            def verify(chosen):
                for a in range(len(chosen)):
                    for b in range(a + 1, len(chosen)):
                        i, r = chosen[a]
                        j, r2 = chosen[b]
                        g = G[i][j]
                        if g < 2:
                            return False
                        if (r % g) == (r2 % g):
                            return False
                return True
            
            if verify(res):
                return [[r, all_m[i]] for (i, r) in res]
    
    # Fallback: Greedy MIS construction
    # This is just in case the even set fails residue assignment (which shouldn't happen)
    # or if we find a larger set.
    
    # Build adjacency list for coprime graph
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if G[i][j] == 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy MIS
    remaining = set(range(n))
    chosen_indices = []
    
    # Sort by degree (descending) to pick high-degree nodes? 
    # Or ascending? 
    # For MIS, picking nodes with high degree removes many edges, which is good.
    # But we want to KEEP nodes.
    # So we should pick nodes that have FEW conflicts?
    # No, we want to pick a set of nodes with NO conflicts.
    # So we pick a node, add it to set, remove it and its neighbors from remaining.
    
    # Heuristic: Pick node with minimum degree in the induced subgraph?
    # Or maximum?
    # Let's try minimum degree (least constrained).
    
    while remaining:
        # Find node with min degree in remaining
        min_deg = float('inf')
        best_node = None
        for u in remaining:
            deg = len(adj[u] & remaining)
            if deg < min_deg:
                min_deg = deg
                best_node = u
        
        chosen_indices.append(best_node)
        # Remove best_node and its neighbors
        to_remove = {best_node} | (adj[best_node] & remaining)
        remaining -= to_remove
        
    # Solve residues for this set
    res = solve_residues(chosen_indices, G, all_m)
    if res:
        return [[r, all_m[i]] for (i, r) in res]
        
    # Absolute fallback
    return [[0, 2]]
