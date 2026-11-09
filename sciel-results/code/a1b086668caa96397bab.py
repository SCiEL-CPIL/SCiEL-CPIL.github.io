import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit (or 90s if available, but 60 is stated in contract)
    
    N = 128
    
    # 1. Precomputation
    # gcd_table[i][j] = gcd(i, j)
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # bad[i][j] is a bitmask of k such that {i, j, k} is a bad triple
    # i.e., gcd(i,j) == gcd(i,k) == gcd(j,k)
    bad = [[0] * (N + 1) for _ in range(N + 1)]
    
    # Precompute for each pair (i, j), the mask of k that conflict
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g_ij = gcd_table[i][j]
            mask = 0
            for k in range(1, N + 1):
                if k == i or k == j:
                    continue
                if gcd_table[i][k] == g_ij and gcd_table[j][k] == g_ij:
                    mask |= (1 << k)
            bad[i][j] = mask
            bad[j][i] = mask

    # Helper: popcount
    def popcount(x):
        return x.bit_count()

    # Helper: Get list of set bits
    def get_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            bits.append(idx)
            mask ^= lsb
        return bits

    # 2. Greedy Initialization
    def greedy_build(order):
        s = 0
        for x in order:
            # Check if x can be added to s
            # x conflicts with s if there exists u, v in s such that {u, v, x} is bad
            # This is equivalent to: for all u in s, x is NOT in bad[u][v] for any v in s?
            # Actually, we just need to check if adding x creates a bad triple with any pair in s.
            # Let S be the current set.
            # We need to check all pairs (u, v) in S.
            # This is O(|S|^2).
            
            # Optimization: Maintain a mask of "forbidden" vertices for the current set S.
            # forbidden_mask = union of bad[u][v] for all u,v in S.
            # If x is in forbidden_mask, it cannot be added.
            
            # However, in a simple greedy, we don't maintain this efficiently across restarts easily without passing state.
            # For initialization, O(N^3) is fine (128^3 ~ 2M ops).
            
            valid = True
            bits_s = get_bits(s)
            for u in bits_s:
                for v in bits_s:
                    if u < v:
                        if (bad[u][v] >> x) & 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                s |= (1 << x)
        return s

    # Better Greedy with Forbidden Mask
    def greedy_build_fast(order):
        s = 0
        forbidden = 0
        for x in order:
            if (forbidden >> x) & 1:
                continue
            
            # Check if x is in s? (It shouldn't be if we process unique order, but order might have dups? No, it's a permutation)
            # If x is not forbidden by any pair in s, we can add it.
            # Wait, the forbidden mask logic:
            # If we add x to s, the new forbidden mask will be:
            # new_forbidden = forbidden | (union of bad[x][u] for u in s)
            # But we must first ensure x itself is not in `forbidden`.
            # If x is not in `forbidden`, it means no pair (u,v) in s has x in bad[u][v].
            # This is exactly the condition for x to be compatible with s.
            
            s |= (1 << x)
            
            # Update forbidden mask
            bits_s_old = get_bits(s & ~(1 << x)) # s without x
            # Actually, we can just update incrementally if we stored it, but here we recalculate or update from previous.
            # Let's just update:
            # new_forbidden = forbidden
            # for u in bits_s_old:
            #     new_forbidden |= bad[x][u]
            
            # To avoid get_bits overhead in the loop, let's just do it.
            temp_s = s & ~(1 << x)
            while temp_s:
                lsb = temp_s & -temp_s
                u = lsb.bit_length() - 1
                forbidden |= bad[x][u]
                temp_s ^= lsb
                
        return s

    # Initial best
    best_set = 0
    best_size = 0
    
    random.seed(42)
    
    # Try a few random orders for greedy
    for _ in range(20):
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_build_fast(order)
        sz = popcount(s)
        if sz > best_size:
            best_size = sz
            best_set = s

    # 3. Branch and Bound
    # State: (current_set_mask, candidates_mask, current_size)
    # We use recursion.
    
    # To speed up B&B, we order candidates by "impact".
    # Precompute degree of each vertex in the hypergraph?
    # deg[i] = number of bad triples involving i.
    deg = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # i is in bad triples defined by (i, j, k)
            deg[i] += popcount(bad[i][j])
            deg[j] += popcount(bad[i][j])
            # k is also in these triples
            mask = bad[i][j]
            while mask:
                lsb = mask & -mask
                k = lsb.bit_length() - 1
                deg[k] += 1
                mask ^= lsb

    # Sort vertices by degree descending for branching order
    vertices_by_deg = sorted(range(1, N + 1), key=lambda x: -deg[x])
    
    # We will use a stack-based B&B to avoid recursion depth issues and allow time checks.
    # Stack items: (current_set_mask, candidates_mask, current_size)
    
    # To make it efficient, we should pick the next vertex from candidates based on some heuristic.
    # Simple heuristic: Pick the vertex in candidates with highest degree.
    # But calculating highest degree in candidates every time is O(N).
    # We can just pick the first one in a pre-sorted list that is in candidates.
    
    # Let's use a recursive function with a time check.
    
    import sys
    sys.setrecursionlimit(10000)

    def branch_and_bound(current_set, candidates, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
            
        if candidates == 0:
            return
            
        # Upper bound
        ub = current_size + popcount(candidates)
        if ub <= best_size:
            return

        # Pick a vertex v from candidates
        # Heuristic: Pick v that appears in the most bad triples *within* candidates?
        # Or just pick the one with highest global degree.
        # Let's iterate through vertices_by_deg and find the first one in candidates.
        v = 0
        for cand in vertices_by_deg:
            if (candidates >> cand) & 1:
                v = cand
                break
        
        if v == 0:
            return

        # Branch 1: Include v
        # Check if v is compatible with current_set
        # v is compatible if v is not in the "forbidden" mask of current_set.
        # We need to calculate the forbidden mask of current_set.
        # To avoid recalculating every time, we can pass it? 
        # But the state is (set, candidates). 
        # Recalculating forbidden mask is O(|S|^2).
        
        # Let's check compatibility directly:
        # For all pairs (u, w) in current_set, is v in bad[u][w]?
        compatible = True
        bits_s = get_bits(current_set)
        for i in range(len(bits_s)):
            for j in range(i + 1, len(bits_s)):
                u = bits_s[i]
                w = bits_s[j]
                if (bad[u][w] >> v) & 1:
                    compatible = False
                    break
            if not compatible:
                break
        
        if compatible:
            new_set = current_set | (1 << v)
            # New candidates: 
            # 1. Remove v
            # 2. Remove any x in candidates such that x forms a bad triple with v and any u in new_set \ {v}
            #    i.e., x in bad[v][u] for any u in current_set.
            #    Also, x cannot form a bad triple with two elements in current_set? 
            #    No, current_set is already valid.
            #    The new constraint is that x must not complete a bad triple with v and any u in current_set.
            #    So, forbidden_new = union of bad[v][u] for u in current_set.
            
            forbidden_new = 0
            for u in bits_s:
                forbidden_new |= bad[v][u]
                
            new_candidates = (candidates & ~(1 << v)) & ~forbidden_new
            
            branch_and_bound(new_set, new_candidates, current_size + 1)

        # Branch 2: Exclude v
        new_candidates = candidates & ~(1 << v)
        branch_and_bound(current_set, new_candidates, current_size)

    # Run B&B
    # We need a good initial candidate set.
    # Let's start with all vertices.
    initial_candidates = 0
    for i in range(1, N + 1):
        initial_candidates |= (1 << i)
        
    # To make B&B faster, we can start from the greedy solution?
    # No, B&B needs to explore the space.
    # However, starting with an empty set and exploring 128 nodes is huge.
    # We can use the greedy solution as a "seed" for the lower bound, but the search starts from scratch.
    # To prune effectively, we need the LB to be high. We have that.
    
    # One optimization: If we start with the greedy set, we can't just "start" B&B from there because we've already made choices.
    # We must start from empty set to be rigorous.
    # But we can use the greedy set to order vertices or something?
    
    # Let's just run it.
    # To avoid deep recursion on the "exclude" branch which is deep but unproductive, 
    # we can prioritize "include" if it looks promising?
    # Standard B&B usually does include first to find good solutions early.
    
    branch_and_bound(0, initial_candidates, 0)

    # Convert best_set to list
    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    
    if not result:
        return [1]
        
    return result
