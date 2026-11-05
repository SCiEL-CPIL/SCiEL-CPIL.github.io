def construct():
    import time
    import sys
    sys.setrecursionlimit(10000)

    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 256
    
    # We use 1-indexed for logic, 0-indexed for bitmasks.
    # Bit i corresponds to number i+1.
    
    # Precompute weights
    weights = [1.0 / (i + 1) for i in range(n)]
    
    # Precompute "forbidden" masks for each element.
    # If we select element i (value v = i+1), which other elements j cannot be selected?
    # Actually, the constraint is: no {a, b, c} in S such that a+c=2b.
    # This is a 3-uniform hypergraph. 
    # For B&B, we typically work with independent sets in graphs. 
    # However, we can view this as: 
    # If we select a set of elements, we must ensure no 3-AP is formed.
    
    # A standard B&B for 3-AP free sets often uses the property that 
    # if we select a set S, the remaining candidates must not complete any AP with elements in S.
    
    # Let's use a recursive B&B on the elements sorted by weight (descending).
    # State: current_index, current_set (bitmask), current_score.
    # Upper bound: current_score + sum of weights of all remaining elements that could potentially be added.
    # To make the bound tight, we should precompute for each index i, the sum of weights from i to n-1.
    # But simply summing all remaining is a loose bound. 
    # A better bound: For each remaining element, check if it is "compatible" with the current set?
    # That's expensive.
    
    # Alternative Strategy: 
    # Since n=256 is small, we can use a specific B&B structure.
    # Sort elements by weight descending: 1, 2, ..., 256.
    # At each step, we decide to include or exclude the next element.
    # If we include element v, we must remove all elements that would form a 3-AP with v and any previously selected element?
    # No, we must ensure that no 3-AP is formed by the *entire* set.
    
    # Let's maintain the set of *available* elements.
    # Initially, all 1..256 are available.
    # When we select an element v, we must remove from `available` any element u such that {u, v, w} is a 3-AP for some w in `available` OR w already selected?
    # Actually, if we select v, we cannot select any pair (u, w) that forms a 3-AP with v.
    # This implies that the constraint is global.
    
    # Efficient B&B for 3-AP:
    # We process elements in order of decreasing weight.
    # We maintain a bitmask `candidates` of elements that can still be added.
    # When we add element v:
    #   1. Remove v from candidates.
    #   2. Identify all pairs (a, b) in candidates such that {a, v, b} is a 3-AP. 
    #      If we add v, we cannot have both a and b.
    #      This doesn't immediately remove a or b, it just creates a conflict between them.
    #      This turns the problem into a Maximum Weight Independent Set (MWIS) on a graph of remaining conflicts?
    #      Yes! 
    #      
    # Let's reframe:
    # The set of all 3-APs defines a hypergraph. 
    # If we fix a subset S of selected elements, the remaining elements C must form a set such that:
    # 1. No 3-AP is contained entirely in C.
    # 2. No 3-AP is formed by 2 elements in C and 1 in S.
    # 3. No 3-AP is formed by 1 element in C and 2 in S.
    #
    # Condition 3 is static given S. If S contains two elements that are part of a 3-AP, the third is forbidden from C.
    # Condition 2: For each v in S, and each pair (a, b) in C, if {a, v, b} is a 3-AP, then we cannot have both a and b in C.
    # This means that within C, we have a graph G_S where edges connect pairs (a, b) that would form a 3-AP with some v in S.
    # Additionally, C itself cannot contain any 3-AP (edges from original hypergraph restricted to C).
    #
    # So, given S, the problem reduces to finding a Maximum Weight Independent Set in a graph G_C induced by C, 
    # where G_C has edges:
    # - (a, b) if {a, b, c} is a 3-AP for some c in S.
    # - (a, b) if {a, b, c} is a 3-AP for some c in C (this is dynamic, but effectively, we just need C to be 3-AP free).
    #
    # Wait, if C is 3-AP free, it means no 3 elements in C form a 3-AP. 
    # This is equivalent to saying that in the graph where edges connect pairs that are ends of a 3-AP (with middle in C or S?), 
    # it's getting complicated.
    
    # Simpler B&B approach for small n:
    # Use the fact that the maximum size of a 3-AP free set in [1, N] is small (related to Roth's theorem, but for N=256 it's around 100-120?).
    # Actually, the density is low.
    
    # Let's use a standard Branch and Bound with a tight upper bound.
    # Sort elements by weight descending.
    # Precompute suffix sums of weights for a loose upper bound.
    # To make it tighter, we can precompute for each index i, the maximum weight independent set of the subgraph induced by elements i..n-1? 
    # That's NP-hard, so we can't precompute it exactly for all i.
    
    # However, we can use a greedy upper bound or a simple relaxation.
    # Given the time limit, a simple B&B might be too slow if the bound is loose.
    
    # Alternative: Use the "Cayley" or "Moser" construction? 
    # The set of numbers with only 0s and 1s in base 3 is 3-AP free.
    # For N=256, we can take the largest such set.
    # Base 3 representation of 256 is 100101 (3^5=243, 3^4=81, 3^3=27, 3^2=9, 3^1=3, 3^0=1).
    # 256 = 1*243 + 0*81 + 0*27 + 1*9 + 0*3 + 1*1.
    # The Moser set includes numbers with digits 0,1 in base 3.
    # Let's generate the Moser set for [1, 256] and see its score.
    # Then, we can try to improve it with local search.
    
    def is_moser(x):
        while x > 0:
            if x % 3 == 2:
                return False
            x //= 3
        return True

    moser_set = [x for x in range(1, 257) if is_moser(x)]
    moser_score = sum(1.0 / x for x in moser_set)
    
    # Now, let's do a local search starting from the Moser set to see if we can improve it.
    # The Moser set is 3-AP free.
    # We can try to swap elements to increase the reciprocal sum.
    
    S = set(moser_set)
    current_score = moser_score
    
    def can_add(x, S):
        # Check if x can be added to S without forming a 3-AP
        # S is a set of ints
        max_d_mid = min(x - 1, 256 - x)
        for d in range(1, max_d_mid + 1):
            if (x - d) in S and (x + d) in S:
                return False
        max_d_start = (256 - x) // 2
        for d in range(1, max_d_start + 1):
            if (x + d) in S and (x + 2 * d) in S:
                return False
        max_d_end = x // 2
        for d in range(1, max_d_end + 1):
            if (x - 2 * d) in S and (x - d) in S:
                return False
        return True

    # Local Search: 1-Swap and 2-Swap
    # Since n=256, we can afford to check all swaps.
    # We want to maximize sum(1/x).
    # So we want to replace larger x with smaller y.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try all possible swaps: remove x from S, add y not in S.
        # Only consider if 1/y > 1/x, i.e., y < x.
        # To be efficient, iterate x in S (largest first) and y in not_S (smallest first).
        
        S_list = sorted(S, reverse=True) # Largest x first (lowest weight)
        not_S_list = sorted([x for x in range(1, 257) if x not in S]) # Smallest y first (highest weight)
        
        # We can break early if y >= x, because 1/y <= 1/x.
        best_swap = None
        best_gain = 0
        
        for x in S_list:
            if time.time() > deadline - 5: break
            for y in not_S_list:
                if y >= x:
                    break
                # Check if removing x and adding y is valid
                # This is equivalent to: can y be added to S - {x}?
                S_temp = S - {x}
                if can_add(y, S_temp):
                    gain = 1.0 / y - 1.0 / x
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (x, y)
        
        if best_swap:
            x, y = best_swap
            S.remove(x)
            S.add(y)
            current_score += best_gain
            improved = True
        else:
            # Try simple additions? 
            # If we can add y without removing x, that's better.
            # But Moser set is maximal? Not necessarily.
            # Let's check if any y in not_S can be added.
            for y in not_S_list:
                if can_add(y, S):
                    S.add(y)
                    current_score += 1.0 / y
                    improved = True
                    break # Restart loop to re-evaluate
    
    # If local search didn't improve much, or if we have time, try a random restart with SA?
    # But the Moser set is a strong starting point.
    # Let's return the best found.
    
    return list(S)

# EVOLVE-BLOCK-END
