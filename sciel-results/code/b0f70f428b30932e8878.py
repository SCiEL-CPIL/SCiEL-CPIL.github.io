import random
import time

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin (90s limit, leave 10s)
    
    def is_valid_4ap_free(S):
        """Check if set S is 4-AP-free."""
        S_list = sorted(S)
        S_set = set(S_list)
        for i, a in enumerate(S_list):
            for d in range(1, N // 3 + 1):
                if a + 3 * d <= N:
                    if (a + d in S_set) and (a + 2 * d in S_set) and (a + 3 * d in S_set):
                        return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to S without creating a 4-AP."""
        # Check all 4 positions where x could be in a 4-AP
        for d in range(1, N // 3 + 1):
            # x is 1st term: x, x+d, x+2d, x+3d
            if x + 3 * d <= N:
                if (x + d in S) and (x + 2 * d in S) and (x + 3 * d in S):
                    return False
            # x is 2nd term: x-d, x, x+d, x+2d
            if x - d >= 1 and x + 2 * d <= N:
                if (x - d in S) and (x + d in S) and (x + 2 * d in S):
                    return False
            # x is 3rd term: x-2d, x-d, x, x+d
            if x - 2 * d >= 1 and x + d <= N:
                if (x - 2 * d in S) and (x - d in S) and (x + d in S):
                    return False
            # x is 4th term: x-3d, x-2d, x-d, x
            if x - 3 * d >= 1:
                if (x - 3 * d in S) and (x - 2 * d in S) and (x - d in S):
                    return False
        return True
    
    def greedy_with_order(order, S_init=None):
        """Greedy: add elements in given order if they don't create a 4-AP."""
        S = S_init.copy() if S_init else set()
        for x in order:
            if x not in S and can_add(S, x):
                S.add(x)
        return S
    
    def local_search(S, max_time=5.0):
        """Local search: try to add elements, then try swaps."""
        S = S.copy()
        ls_start = time.time()
        
        # Phase 1: Try to add any valid elements
        changed = True
        while changed and (time.time() - ls_start) < max_time * 0.3:
            changed = False
            for x in range(1, N + 1):
                if x not in S and can_add(S, x):
                    S.add(x)
                    changed = True
                    break
        
        # Phase 2: Try swaps - remove one element, add two
        improved = True
        while improved and (time.time() - ls_start) < max_time:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            for x_remove in S_list:
                if (time.time() - ls_start) >= max_time:
                    break
                S_temp = S.copy()
                S_temp.discard(x_remove)
                # Try to add two elements
                additions = 0
                candidates = [x for x in range(1, N + 1) if x not in S_temp]
                random.shuffle(candidates)
                for x_add in candidates:
                    if additions >= 2:
                        break
                    if can_add(S_temp, x_add):
                        S_temp.add(x_add)
                        additions += 1
                if len(S_temp) > len(S):
                    S = S_temp
                    improved = True
        
        # Phase 3: Try single removals + single additions (net 0, but might enable future additions)
        # Actually, let's just do a final add pass
        changed = True
        while changed and (time.time() - ls_start) < max_time:
            changed = False
            for x in range(1, N + 1):
                if x not in S and can_add(S, x):
                    S.add(x)
                    changed = True
                    break
        
        return S
    
    # Generate initial solutions
    best_set = set()
    best_size = 0
    
    # Try ternary construction as initial
    ternary_set = set()
    for i in range(1, N + 1):
        # Check if ternary representation has no '1'
        n = i
        valid = True
        while n > 0:
            if n % 3 == 1:
                valid = False
                break
            n //= 3
        if valid:
            ternary_set.add(i)
    
    if len(ternary_set) > best_size:
        best_size = len(ternary_set)
        best_set = ternary_set.copy()
    
    # Greedy with random orders
    elements = list(range(1, N + 1))
    
    num_restarts = 0
    while time.time() < deadline - 10.0:  # Leave 10s for local search
        random.shuffle(elements)
        S = greedy_with_order(elements)
        num_restarts += 1
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Local search on best solution
    if time.time() < deadline - 5.0:
        best_set = local_search(best_set, max_time=min(5.0, deadline - time.time() - 2.0))
        best_size = len(best_set)
    
    # Final verification
    if not is_valid_4ap_free(best_set):
        # Fallback to ternary
        best_set = ternary_set
    
    return sorted(best_set)

