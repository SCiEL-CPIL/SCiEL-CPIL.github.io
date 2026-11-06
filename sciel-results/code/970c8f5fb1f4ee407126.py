import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    
    def is_valid_addition(candidate, S):
        """Check if adding candidate to S maintains the property."""
        # Case 1: candidate is 'a' (smallest in triple)
        # Need: for all b, c in S with b > candidate, b <= c: candidate ∤ (b+c)
        for b in S:
            if b <= candidate:
                continue
            # b > candidate, need to check all c in S with c >= b
            # candidate divides b+c iff c ≡ -b (mod candidate)
            # So c = k*candidate - b for some k, and c >= b, c in S
            # b <= k*candidate - b <= n
            # 2b <= k*candidate <= n+b
            k_min = (2 * b + candidate - 1) // candidate
            k_max = (n + b) // candidate
            for k in range(k_min, k_max + 1):
                c = k * candidate - b
                if c >= b and c in S:
                    return False
        
        # Case 2: candidate is 'b' (middle in triple)
        # Need: for all a in S with a < candidate, and c in S with c >= candidate: a ∤ (candidate+c)
        for a in S:
            if a >= candidate:
                continue
            # a < candidate, need c >= candidate, a divides candidate+c
            # c = k*a - candidate, c >= candidate, c <= n
            # candidate <= k*a - candidate <= n
            # 2*candidate <= k*a <= n + candidate
            k_min = (2 * candidate + a - 1) // a
            k_max = (n + candidate) // a
            for k in range(k_min, k_max + 1):
                c = k * a - candidate
                if c >= candidate and c in S:
                    return False
        
        # Case 3: candidate is 'c' (largest in triple)
        # Need: for all a, b in S with a < b <= candidate: a ∤ (b+candidate)
        for a in S:
            if a >= candidate:
                continue
            # a < candidate, need b in S with a < b <= candidate, a divides b+candidate
            # b = k*a - candidate, a < b <= candidate
            # a < k*a - candidate <= candidate
            # candidate + a < k*a <= 2*candidate
            # k > 1 + candidate/a, k <= 2*candidate/a
            k_min = (candidate // a) + 2
            k_max = (2 * candidate) // a
            for k in range(k_min, k_max + 1):
                b = k * a - candidate
                if a < b <= candidate and b in S:
                    return False
        
        return True
    
    def greedy(order, S_initial=None):
        """Greedy construction with given order."""
        S = set(S_initial) if S_initial else set()
        for x in order:
            if time.time() > deadline:
                break
            if x not in S and is_valid_addition(x, S):
                S.add(x)
        return S
    
    def local_improve(S, max_iters=5000):
        """Try to improve S by swapping or adding elements."""
        S = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iters and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try adding any element not in S
            for x in range(1, n + 1):
                if x in S:
                    continue
                if is_valid_addition(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try swapping: remove one element, try to add two others
                S_list = sorted(S)
                for i, remove_x in enumerate(S_list):
                    if time.time() > deadline:
                        return S
                    S_new = S - {remove_x}
                    added_count = 0
                    for add_x in range(1, n + 1):
                        if add_x in S_new or add_x == remove_x:
                            continue
                        if is_valid_addition(add_x, S_new):
                            S_new.add(add_x)
                            added_count += 1
                            if added_count >= 2:
                                break
                    if added_count >= 2 and len(S_new) > len(S):
                        S = S_new
                        improved = True
                        break
        
        return S
    
    # Strategy 1: Descending greedy (prioritize large numbers)
    S1 = greedy(range(n, 0, -1))
    best_set = S1
    
    # Strategy 2: Ascending greedy
    if time.time() < deadline:
        S2 = greedy(range(1, n + 1))
        if len(S2) > len(best_set):
            best_set = S2
    
    # Strategy 3: Random shuffles with different seeds
    random.seed(123)
    for seed in range(50):
        if time.time() > deadline:
            break
        random.seed(seed * 7919 + 13)
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_rand = greedy(order)
        if len(S_rand) > len(best_set):
            best_set = S_rand
    
    # Strategy 4: Weighted ordering - try to include numbers with fewer conflicts
    if time.time() < deadline:
        # Compute a "conflict score" for each number
        # Numbers that divide many sums are more problematic
        conflict_score = [0] * (n + 1)
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                for c in range(b, n + 1):
                    if (b + c) % a == 0:
                        conflict_score[a] += 1
        
        # Sort by conflict score (ascending - fewer conflicts first)
        order_by_conflict = sorted(range(1, n + 1), key=lambda x: conflict_score[x])
        S_conflict = greedy(order_by_conflict)
        if len(S_conflict) > len(best_set):
            best_set = S_conflict
    
    # Strategy 5: Local improvement on best set
    if time.time() < deadline:
        best_set = local_improve(best_set)
    
    # Strategy 6: Try local improvement on other good candidates
    if time.time() < deadline:
        S1_imp = local_improve(S1)
        if len(S1_imp) > len(best_set):
            best_set = S1_imp
    
    if time.time() < deadline:
        S2_imp = local_improve(S2)
        if len(S2_imp) > len(best_set):
            best_set = S2_imp
    
    if not best_set:
        return [256]
    
    return sorted(best_set)
