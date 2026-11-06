import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    n = 256
    
    # Precompute: for each a, which sums s are multiples of a
    # For triple (a,b,c) with a<b<=c: violation if a | (b+c)
    # Precompute multiples: mult[a] = set of sums that are multiples of a
    # But we need to check b+c for specific b,c
    
    # Better: for each a, precompute a boolean array div_a[s] = (s % a == 0) for s in range(2, 513)
    # Then check div_a[b+c]
    
    # For branch-and-bound, process elements in decreasing order
    # State: which elements > current have been included
    # When considering element a, check all pairs (b,c) from included elements > a
    # where a < b <= c and a | (b+c)
    
    # Precompute for speed: for each a, and for each possible pair (b,c) with a<b<=c<=256,
    # whether a divides b+c. Store as a 2D boolean or use modular arithmetic.
    
    # Actually, let's just use modular arithmetic directly - it's fast enough.
    
    # Branch and bound:
    # Process elements from 256 down to 1
    # Maintain: included = set of elements already decided to include (all > current)
    # best_set = best valid set found so far
    # 
    # At element a:
    #   Option 1: Include a. Check if {a} ∪ included is valid (only need to check triples involving a)
    #     If valid, recurse
    #   Option 2: Exclude a, recurse
    #   Prune: if len(included) + (256 - a + 1) <= len(best_set), prune (can't beat best)
    #   Also prune if including a is impossible (conflict)
    
    # But 2^256 is too large. We need good pruning.
    # Key: process in decreasing order. When we're at element a, all elements > a are decided.
    # The number of elements > a is 256-a. We need to check conflicts with those.
    
    # Let's use a greedy initialization first, then branch-and-bound around it.
    
    # Greedy: start with all elements, remove conflicts
    # Or: start from top, greedily add if no conflict
    
    # Greedy initialization:
    S = set()
    for a in range(256, 0, -1):
        # Check if adding a to S creates any violation
        valid = True
        # For all b,c in S with a < b <= c: check a | (b+c)
        S_list = sorted(S)
        for i in range(len(S_list)):
            b = S_list[i]
            if b <= a:
                break
            for j in range(i, len(S_list)):
                c = S_list[j]
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(a)
    
    best_set = set(S)
    best_size = len(S)
    
    # Now branch-and-bound to improve
    # Process elements in decreasing order, but allow both include and exclude
    # Use the greedy solution as a lower bound for pruning
    
    elements = list(range(256, 0, -1))
    
    def can_add(a, current_set):
        """Check if adding a to current_set (where all elements > a) creates a violation."""
        # Need: for all b,c in current_set with a < b <= c, a does not divide b+c
        cs = sorted(current_set)
        # Only consider elements > a
        for i in range(len(cs)):
            b = cs[i]
            if b <= a:
                break
            for j in range(i, len(cs)):
                c = cs[j]
                if (b + c) % a == 0:
                    return False
        return True
    
    def branch_and_bound(idx, current_set, best_set_ref):
        """Branch and bound. idx is index in elements list."""
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if idx == len(elements):
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        a = elements[idx]
        
        # Pruning: even if we include all remaining, can we beat best?
        remaining = len(elements) - idx
        if len(current_set) + remaining <= best_size:
            return
        
        # Option 1: Include a
        if can_add(a, current_set):
            current_set.add(a)
            branch_and_bound(idx + 1, current_set, best_set_ref)
            current_set.discard(a)
        
        # Option 2: Exclude a
        branch_and_bound(idx + 1, current_set, best_set_ref)
    
    # The above is too slow for 256 elements. We need a smarter approach.
    # 
    # Better: Use the greedy solution as a starting point, then do local search
    # with conflict-based reasoning.
    #
    # Alternative: Process in chunks. The key insight is that for large a,
    # the constraint a | (b+c) is restrictive. For small a, it's less so.
    #
    # Let's try a different approach: 
    # 1. Start with greedy solution
    # 2. For each element not in S, try to add it by removing minimal conflicting elements
    # 3. Use the precomputed conflict structure
    
    # Precompute conflicts: for each a, which (b,c) pairs conflict
    # conflict[a] = set of (b,c) with a < b <= c and a | (b+c)
    
    # Actually, let's do a smarter local search:
    # For each element a not in S, compute the set of elements in S that conflict with a
    # (i.e., there exists b in S with b > a such that a | (a+b) or a | (b+c) for some c)
    # Wait, the conflict is: a,b,c in S with a<b<=c and a|(b+c)
    # So for fixed a, conflicting pairs are (b,c) in S with a<b<=c and a|(b+c)
    
    # For local search: try swapping elements
    # For each a in S and each x not in S, try S' = S - {a} + {x}
    # Check if S' is valid
    
    # This is O(|S| * (n-|S|) * |S|²) which is too slow.
    
    # Let me optimize the validity check and do more iterations.
    
    # Optimized validity check using precomputed data:
    # For a set S, sort it. For each a in S (smallest), check all pairs (b,c) in S with b>c>a.
    # Precompute: for each a, a numpy array div[a][s] = (s % a == 0) for s in range(2, 513)
    
    div_table = {}
    for a in range(1, 257):
        div_table[a] = np.zeros(513, dtype=bool)
        for s in range(2*a, 513, a):
            div_table[a][s] = True
    
    def is_valid_fast(S):
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                for k in range(j, m):
                    c = S_list[k]
                    if div_table[a][b+c]:
                        return False
        return True
    
    # Try to improve the greedy solution with local search
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        S_list = sorted(S)
        for i, elem in enumerate(S_list):
            if time.time() > deadline:
                break
            # Try replacing elem with a smaller element
            for a in range(1, elem):
                if time.time() > deadline:
                    break
                S_new = (S - {elem}) | {a}
                if is_valid_fast(S_new):
                    if len(S_new) > len(S):
                        S = S_new
                        improved = True
                        break
                    elif len(S_new) == len(S):
                        # Same size, but maybe better for future improvements
                        # Only accept if it allows more additions later
                        # For now, skip to keep it simple
                        pass
            # Try adding a new element (if S isn't full)
            for a in range(1, 257):
                if a in S:
                    continue
                if time.time() > deadline:
                    break
                S_new = S | {a}
                if is_valid_fast(S_new):
                    S = S_new
                    improved = True
                    break
    
    if len(S) > len(best_set):
        best_set = S
        best_size = len(S)
    
    # Final: try to maximize by adding more elements
    # Sort candidates by how "safe" they are (fewer conflicts)
    candidates = [x for x in range(1, 257) if x not in S]
    candidates.sort()
    for a in candidates:
        if time.time() > deadline:
            break
        S_new = S | {a}
        if is_valid_fast(S_new):
            S = S_new
            best_set = S
            best_size = len(S)
    
    return list(best_set)
