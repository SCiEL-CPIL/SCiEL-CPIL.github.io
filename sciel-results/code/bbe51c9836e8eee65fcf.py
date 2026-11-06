"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
            
    # Precompute bad triples
    # For each element x, precompute a bitmask of other elements y such that
    # there exists z > max(x,y) with gcd(x,y) == gcd(x,z) == gcd(y,z)
    # Actually, let's precompute for each pair (x,y) with x < y, whether they form a bad triple with any z.
    # A pair (x,y) is "bad" if there exists z != x,y such that gcd(x,y)==gcd(x,z)==gcd(y,z).
    # If we select x and y, we must NOT select any z that completes a bad triple with them.
    # So for each pair (x,y), we can compute the set of "conflict" elements z.
    
    # Let's precompute conflict_mask[x][y] = bitmask of z such that {x,y,z} is a bad triple.
    # But storing 128*128 bitmasks of 128 bits each is 128*128*16 bytes = 256KB. Fine.
    
    # Actually, let's think about the structure of bad triples.
    # gcd(a,b) = gcd(a,c) = gcd(b,c) = g.
    # This implies g divides a, b, c.
    # Let a = g*a', b = g*b', c = g*c' with gcd(a',b')=gcd(a',c')=gcd(b',c')=1? 
    # No, gcd(a,b)=g means gcd(a',b')=1.
    # gcd(a,c)=g means gcd(a',c')=1.
    # gcd(b,c)=g means gcd(b',c')=1.
    # So a', b', c' are pairwise coprime.
    # And g is a common divisor.
    
    # Precompute for each pair (i,j) (i<j), the list of k such that gcd(i,j)==gcd(i,k)==gcd(j,k).
    # Let's call this bad_with[i][j] = set of k.
    
    bad_with = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd_table[a][b]
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[a][c] == g and gcd_table[b][c] == g:
                    bad_with[a][b].add(c)
                    bad_with[b][a].add(c)
    
    # Now, a set S is valid iff for all distinct x,y,z in S, z is NOT in bad_with[x][y].
    # Equivalently, for all pairs (x,y) in S, bad_with[x][y] is disjoint from S.
    
    # We want to maximize |S|.
    # This is equivalent to finding a maximum set S such that for all x,y in S, S does not intersect bad_with[x][y].
    
    # Let's try a greedy approach with random restarts and local search.
    # Or, we can model it as: 
    # Start with empty set.
    # Add elements one by one.
    # When we add x, we remove from consideration any y such that there exists z in S with y in bad_with[x][z] or x in bad_with[y][z]?
    # No, the condition is symmetric. If we have S and we want to add x, we need to check:
    # For all pairs (a,b) in S, x is not in bad_with[a][b].
    # AND for all a in S, bad_with[x][a] must be disjoint from S (excluding a? No, bad_with[x][a] contains elements c such that {x,a,c} is bad. So if c is in S, it's bad).
    # So adding x is valid iff:
    # 1. For all a,b in S with a!=b, x not in bad_with[a][b].
    # 2. For all a in S, bad_with[x][a] is disjoint from S.
    
    # Let's implement a function to check if x can be added to S.
    
    def can_add(x, S):
        # S is a set of integers
        # Check 1: For all pairs in S, x not in bad_with[a][b]
        for a in S:
            for b in S:
                if a == b:
                    continue
                if x in bad_with[a][b]:
                    return False
        # Check 2: For all a in S, bad_with[x][a] disjoint from S
        for a in S:
            if a == x:
                continue
            # Check intersection of bad_with[x][a] and S
            # Since S is a set, we can just iterate
            for c in bad_with[x][a]:
                if c in S:
                    return False
        return True
    
    # This is O(|S|^2) per check, which is slow for large S.
    # Let's optimize by maintaining for each x, a count of how many elements in S are in bad_with[x][a] for some a in S?
    # Actually, let's precompute for each x, a bitmask of elements y such that x is in bad_with[a][b] for some pair (a,b)? No.
    
    # Let's use a different representation.
    # Represent S as a bitmask.
    # Precompute for each x, a bitmask conflict_mask[x] where bit y is set if x is in bad_with[a][b] for some... no.
    
    # Let's precompute for each pair (x,y), the bitmask of z in bad_with[x][y].
    # Let conflict_bitmask[x][y] be a bitmask of length N+1 (bits 0..N) where bit k is 1 if k is in bad_with[x][y].
    
    # Then, given S (as a bitmask), and we want to add x:
    # Condition 1: For all a,b in S, bit x of conflict_bitmask[a][b] is 0.
    # This is equivalent to: (conflict_bitmask[a][b] & (1 << x)) == 0 for all a,b in S.
    # We can precompute for each x, a bitmask "forbidden_by_pairs[x]"? No, it depends on the specific pairs in S.
    
    # Alternative: 
    # For a given S (bitmask), the set of elements that CANNOT be added is the union over all pairs (a,b) in S of (bad_with[a][b] intersect complement of S? No, just bad_with[a][b]).
    # Wait, if x is in bad_with[a][b] for some a,b in S, then x cannot be added.
    # Also, if we add x, then for each a in S, bad_with[x][a] must be disjoint from S.
    # This second condition means: for each a in S, (conflict_bitmask[x][a] & S) == 0.
    
    # So, to add x to S:
    # 1. For all a,b in S: (conflict_bitmask[a][b] & (1 << x)) == 0
    # 2. For all a in S: (conflict_bitmask[x][a] & S) == 0
    
    # Let's precompute conflict_bitmask.
    
    conflict_bitmask = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            mask = 0
            for k in bad_with[i][j]:
                mask |= (1 << k)
            conflict_bitmask[i][j] = mask
            conflict_bitmask[j][i] = mask
    
    # Now, implement can_add with bitmasks.
    # S_mask is an integer bitmask.
    # x is an integer.
    
    def can_add_bitmask(x, S_mask):
        # Condition 1: For all a,b in S_mask, bit x of conflict_bitmask[a][b] is 0.
        # Iterate over all pairs in S_mask.
        # We can extract elements from S_mask.
        # To speed this up, we can precompute for each x, a list of pairs (a,b) such that x is in bad_with[a][b].
        # Let's call this pairs_containing[x] = list of (a,b) such that x in bad_with[a][b].
        # Then Condition 1 is: for all (a,b) in pairs_containing[x], it is NOT the case that (a in S_mask and b in S_mask).
        # i.e., for all (a,b) in pairs_containing[x], (S_mask & (1 << a)) == 0 or (S_mask & (1 << b)) == 0.
        
        # Condition 2: For all a in S_mask, (conflict_bitmask[x][a] & S_mask) == 0.
        # i.e., for all a in S_mask, conflict_bitmask[x][a] has no bits in common with S_mask.
        
        # Let's precompute pairs_containing[x].
        # This is O(N^3) to build, which is 128^3 = 2M. Fine.
        
        # For now, let's just iterate.
        # Condition 1:
        for a in range(1, N + 1):
            if not (S_mask & (1 << a)):
                continue
            for b in range(a + 1, N + 1):
                if not (S_mask & (1 << b)):
                    continue
                if conflict_bitmask[a][b] & (1 << x):
                    return False
        
        # Condition 2:
        for a in range(1, N + 1):
            if not (S_mask & (1 << a)):
                continue
            if conflict_bitmask[x][a] & S_mask:
                return False
        
        return True
    
    # This is still O(N^2) per check.
    # Let's optimize Condition 1 by precomputing pairs_containing.
    
    pairs_containing = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for k in bad_with[a][b]:
                pairs_containing[k].add((a, b))
    
    def can_add_fast(x, S_mask):
        # Condition 1:
        for (a, b) in pairs_containing[x]:
            if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                return False
        # Condition 2:
        for a in range(1, N + 1):
            if not (S_mask & (1 << a)):
                continue
            if conflict_bitmask[x][a] & S_mask:
                return False
        return True
    
    # Even Condition 2 is O(N). Total O(N + |pairs_containing[x]|).
    # |pairs_containing[x]| can be up to N^2/2.
    # But in practice, it's much smaller.
    
    # Let's try a greedy construction with random order.
    
    def greedy_random():
        S_mask = 0
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        for x in elements:
            if can_add_fast(x, S_mask):
                S_mask |= (1 << x)
        return S_mask
    
    # Run multiple greedy constructions and keep the best.
    best_mask = 0
    best_size = 0
    
    num_greedy = 100
    for _ in range(num_greedy):
        mask = greedy_random()
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Now, try to improve with local search (add/remove/swap).
    # We want to maximize size.
    # Let's try adding elements that are not in best_mask.
    # If we can't add any, try removing one element to allow adding two others?
    
    current_mask = best_mask
    current_size = best_size
    
    # Local search
    improved = True
    while improved:
        improved = False
        # Try to add
        for x in range(1, N + 1):
            if current_mask & (1 << x):
                continue
            if can_add_fast(x, current_mask):
                current_mask |= (1 << x)
                current_size += 1
                improved = True
    
    # Try swaps: remove one, add two?
    # This is complex. Let's try a simpler 1-move local search:
    # Try removing one element and see if we can add two new ones.
    # Or just try all single removals and then try to fill.
    
    # Let's try a more systematic local search with random restarts.
    # But time is limited.
    
    # Let's also try a "fill" strategy:
    # Start with empty set, add elements in a specific order that minimizes conflicts.
    # Order by degree in the hypergraph?
    # Degree of x = number of bad triples containing x.
    
    deg = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in bad_with[a][b]:
                deg[a] += 1
                deg[b] += 1
                deg[c] += 1
    
    # Greedy by degree
    def greedy_degree():
        S_mask = 0
        elements = sorted(range(1, N + 1), key=lambda x: deg[x])
        for x in elements:
            if can_add_fast(x, S_mask):
                S_mask |= (1 << x)
        return S_mask
    
    mask_deg = greedy_degree()
    size_deg = bin(mask_deg).count('1')
    if size_deg > best_size:
        best_size = size_deg
        best_mask = mask_deg
    
    # Final polish: try to add any remaining elements
    for x in range(1, N + 1):
        if not (best_mask & (1 << x)):
            if can_add_fast(x, best_mask):
                best_mask |= (1 << x)
                best_size += 1
    
    # Convert to list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    return result

# EVOLVE-BLOCK-END
