import time
import random

def construct():
    N = 128
    start_time = time.time()
    time_limit = 80  # safety margin before 90s kill
    
    # Initial solution: upper half
    S = set(range(N//2 + 1, N + 1))
    
    def is_valid_set(S):
        """Check if S has no forbidden triple."""
        items = sorted(S)
        for i in range(len(items)):
            a = items[i]
            for j in range(i+1, len(items)):
                b = items[j]
                if a + b in S:
                    for k in range(len(items)):
                        c = items[k]
                        if c == a or c == b:
                            continue
                        if a + c in S and b + c in S:
                            return False
        return True
    
    def can_add(S, x):
        """Check if adding x to S keeps it valid."""
        if x in S:
            return True
        # Need: no distinct a,b,c in S∪{x} with a+b,a+c,b+c all in S∪{x}
        # Since x is new, the sums involving x (a+x, b+x) must already be in S
        # Case 1: x is one of {a,b,c}. WLOG x=a. Then need b+c, b+x, c+x all in S∪{x}
        #   b+x and c+x must be in S (since they're > x typically, and x is new)
        #   b+c must be in S
        #   So: exists distinct b,c in S such that b+c ∈ S, b+x ∈ S, c+x ∈ S
        # Case 2: x is not in {a,b,c}. Then a+b,a+c,b+c all in S, which would already be invalid in S.
        #   So only Case 1 matters.
        for b in S:
            if b + x not in S:
                continue
            for c in S:
                if c == b:
                    continue
                if c + x not in S:
                    continue
                if b + c in S:
                    return False
        return True
    
    def can_remove_and_add(S, x_remove, x_add):
        """Check if removing x_remove and adding x_add keeps S valid."""
        if x_remove not in S or x_add in S:
            return False
        new_S = S - {x_remove} | {x_add}
        return is_valid_set(new_S)
    
    best_S = S.copy()
    best_size = len(S)
    
    # Greedy: try to add elements in increasing order
    for x in range(1, N//2 + 1):
        if can_add(best_S, x):
            best_S.add(x)
    
    # Local search with swaps
    random.seed(42)
    current_S = best_S.copy()
    current_size = len(current_S)
    
    # Precompute for faster checking
    def fast_can_add(S, x):
        if x in S:
            return True
        # For each b in S where b+x in S, check if there's c in S, c!=b, with c+x in S and b+c in S
        candidates_b = [b for b in S if (b + x) in S]
        for b in candidates_b:
            for c in S:
                if c == b:
                    continue
                if (c + x) in S and (b + c) in S:
                    return False
        return True
    
    # Iterative improvement
    no_improve_count = 0
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try adding single elements
        for x in range(1, N + 1):
            if x not in current_S and fast_can_add(current_S, x):
                current_S.add(x)
                improved = True
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = current_S.copy()
        
        # Try swaps: remove one, add another
        if not improved:
            items = list(current_S)
            random.shuffle(items)
            for x_remove in items:
                if time.time() - start_time > time_limit - 5:
                    break
                for x_add in range(1, N + 1):
                    if x_add in current_S:
                        continue
                    new_S = current_S - {x_remove} | {x_add}
                    if is_valid_set(new_S) and len(new_S) >= len(current_S):
                        current_S = new_S
                        improved = True
                        if len(current_S) > best_size:
                            best_size = len(current_S)
                            best_S = current_S.copy()
                        break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 5:
                break
        else:
            no_improve_count = 0
    
    # Final validation
    if not is_valid_set(best_S):
        return list(range(N//2 + 1, N + 1))
    
    return list(best_S)
