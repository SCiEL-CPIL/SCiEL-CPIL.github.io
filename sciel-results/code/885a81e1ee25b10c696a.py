"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_valid_set(S):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            for k in range(j+1, len(S)):
                a, b, c = S[i], S[j], S[k]
                if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                    return False
    return True

def _bitmask_construct(n=128):
    """
    Bit-parallel greedy construction with conflict masks.
    
    For each vertex x, precompute:
      - pair_sum_mask[x]: bitmask of all y such that (x+y) <= n
      - For adding x to current set S, x is invalid if there exist a,b in S
        such that (a+b) in S, (a+x) in S, (b+x) in S.
    
    We use the insight: x conflicts with pair (a,b) iff:
      a+b in S AND a+x in S AND b+x in S
    
    Precompute for each x: conflict_pairs[x] = set of (a,b) pairs that would
    form a forbidden triple with x, given the current S.
    
    Actually, let's use a different approach:
    For a candidate x, check if there exist a,b in S where:
      - a+b is in S (pair sum already selected)
      - a+x is in S
      - b+x is in S
    
    Precompute: for each x, the set of a in S such that (a+x) in S.
    Call this neighbors_x. Then x is invalid if there exist a,b in neighbors_x
    such that a+b in S.
    """
    n = 128
    # Precompute: for each x, which a values have (a+x) <= n
    # We'll use bitmasks where bit i represents value i+1 (1-indexed)
    
    S = set()
    S_mask = 0  # bit i set means (i+1) in S
    
    # For greedy: try elements in a specific order
    # Try multiple orders and keep the best
    
    def greedy(order):
        S_local = set()
        S_mask_local = 0
        
        for x in order:
            if x in S_local:
                continue
            # Check if adding x creates a forbidden triple
            # x conflicts if there exist a,b in S_local such that:
            #   (a+b) in S_local AND (a+x) in S_local AND (b+x) in S_local
            #
            # Let N_x = {a in S_local : (a+x) in S_local}
            # x is invalid iff there exist a,b in N_x with (a+b) in S_local
            #
            # Efficient check: for each a in N_x, check if any b in N_x 
            # (b != a) has (a+b) in S_local
            
            # Build N_x: all a in S_local where (a+x) is also in S_local
            N_x = []
            for a in S_local:
                if a + x <= n and (a + x) in S_local:
                    N_x.append(a)
            
            # Check if any pair in N_x has their sum in S_local
            valid = True
            for i in range(len(N_x)):
                a = N_x[i]
                for j in range(i+1, len(N_x)):
                    b = N_x[j]
                    if a + b in S_local:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S_local.add(x)
                S_mask_local |= (1 << x)
        
        return S_local
    
    # Try different orderings
    best = set()
    
    # Order 1: increasing
    S1 = greedy(list(range(1, n+1)))
    if len(S1) > len(best):
        best = S1
    
    # Order 2: decreasing
    S2 = greedy(list(range(n, 0, -1)))
    if len(S2) > len(best):
        best = S2
    
    # Order 3: only large numbers first (upper half)
    S3 = greedy(list(range(65, n+1)) + list(range(1, 65)))
    if len(S3) > len(best):
        best = S3
    
    # Order 4: only odd numbers
    S4 = greedy([x for x in range(1, n+1) if x % 2 == 1])
    if len(S4) > len(best):
        best = S4
    
    # Order 5: only even numbers
    S5 = greedy([x for x in range(1, n+1) if x % 2 == 0])
    if len(S5) > len(best):
        best = S5
    
    # Order 6: numbers > 64 (upper half only)
    S6 = greedy(list(range(65, n+1)))
    if len(S6) > len(best):
        best = S6
    
    return best

def _local_search_bitmask(S, n=128, deadline=None):
    """Local search using bitmask-based validity checks."""
    S = set(S)
    improved = True
    
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try adding each missing element
        for x in range(1, n+1):
            if x in S:
                continue
            
            # Check validity: does adding x create a forbidden triple?
            # x is invalid if there exist a,b in S such that:
            #   (a+b) in S AND (a+x) in S AND (b+x) in S
            
            # Build N_x = {a in S : (a+x) in S}
            N_x = [a for a in S if a + x <= n and (a + x) in S]
            
            valid = True
            for i in range(len(N_x)):
                a = N_x[i]
                for j in range(i+1, len(N_x)):
                    b = N_x[j]
                    if a + b in S:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.add(x)
                improved = True
                break  # Restart after adding
        
        if improved:
            continue
        
        # Try swaps: remove one element, add another
        S_list = list(S)
        found_swap = False
        for a in S_list:
            S_without_a = S - {a}
            for x in range(1, n+1):
                if x in S_without_a:
                    continue
                
                # Check if S_without_a ∪ {x} is valid
                # Only need to check triples involving x
                N_x = [b for b in S_without_a if b + x <= n and (b + x) in S_without_a]
                
                valid = True
                for i in range(len(N_x)):
                    b = N_x[i]
                    for j in range(i+1, len(N_x)):
                        c = N_x[j]
                        if b + c in S_without_a:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    S = S_without_a | {x}
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s limit
    
    # Get initial solution via bitmask-based greedy
    best_set = _bitmask_construct(128)
    
    # Improve with local search
    if time.time() < deadline:
        S_improved = _local_search_bitmask(best_set, 128, deadline=deadline)
        if len(S_improved) > len(best_set):
            best_set = S_improved
    
    # Verify validity
    S_final = list(best_set)
    if not _is_valid_set(S_final):
        # Fallback to known valid set
        S_final = list(range(65, 129))
    
    # Ensure all are Python ints in 1..128
    S_final = [int(x) for x in S_final if 1 <= x <= 128]
    
    return S_final
# EVOLVE-BLOCK-END
