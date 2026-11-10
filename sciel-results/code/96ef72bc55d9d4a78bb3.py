import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 256
    # We want to maximize sum of 1/x for 3-AP-free subset of [1, N]
    
    # Precompute reciprocal weights
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
    
    # Check if adding x to current set creates a 3-AP
    # A set is 3-AP-free if for no a < b < c in set do we have a + c = 2b
    # When adding x, we need to check:
    #   For all a in S with a < x: if 2*x - a is in S and 2*x - a != a and 2*x - a != x, conflict
    #   Actually: for all pairs (a, b) in S, check if adding x creates a 3-AP
    #   The three possibilities for a 3-AP involving x:
    #     x, a, b: x + b = 2a => b = 2a - x, need a in S, b in S, a != x, b != x
    #     a, x, b: a + b = 2x => b = 2x - a, need a in S, b in S, a != x, b != x  
    #     a, b, x: a + x = 2b => a = 2b - x, need a in S, b in S, a != x, b != x
    
    def is_valid_addition(x, in_set):
        """Check if x can be added to in_set without creating a 3-AP."""
        # For each a in in_set with a < x:
        #   Check if 2*x - a is in in_set (and != a, != x) -> this would make a, x, 2*x-a a 3-AP
        #   Check if 2*a - x is in in_set (and >= 1, != a, != x) -> this would make 2*a-x, a, x a 3-AP
        # For each b in in_set with b < x:
        #   Check if x + b = 2*a for some a... wait, let me think more carefully.
        # 
        # A 3-AP is (u, v, w) with u + w = 2v, u < v < w.
        # If x is the smallest: x, v, w with x + w = 2v => w = 2v - x. Need v, w in S, v > x, w > v.
        # If x is the middle: u, x, w with u + w = 2x => w = 2x - u. Need u, w in S, u < x, w > x.
        # If x is the largest: u, v, x with u + x = 2v => u = 2v - x. Need u, v in S, u < v < x.
        
        for a in in_set:
            if a >= x:
                continue
            # Case: a is smallest, x is middle: need w = 2x - a in S, w > x
            w = 2 * x - a
            if w != x and w > x and w <= N and w in in_set:
                return False
            # Case: a is middle, x is largest: need u = 2a - x in S, u < a
            u = 2 * a - x
            if u != x and u != a and u >= 1 and u < a and u in in_set:
                return False
        # Case: x is smallest: need v, w in S with x + w = 2v, v > x, w > v
        # For each v in S with v > x: w = 2v - x, need w in S, w > v
        for v in in_set:
            if v <= x:
                continue
            w = 2 * v - x
            if w != x and w > v and w <= N and w in in_set:
                return False
        return True
    
    # Better: use a set for O(1) lookup
    def is_valid_addition_fast(x, in_set):
        """Check if x can be added to in_set (a Python set) without creating a 3-AP."""
        for a in in_set:
            if a >= x:
                continue
            # a smallest, x middle: w = 2x - a
            w = 2 * x - a
            if w != x and w > x and w <= N and w in in_set:
                return False
            # a middle, x largest: u = 2a - x
            u = 2 * a - x
            if u != x and u != a and 1 <= u < a and u in in_set:
                return False
        # x smallest: for v in S with v > x, w = 2v - x
        for v in in_set:
            if v <= x:
                continue
            w = 2 * v - x
            if w != x and w > v and w <= N and w in in_set:
                return False
        return True
    
    # Greedy baseline
    greedy_set = set()
    for x in range(1, N + 1):
        if is_valid_addition_fast(x, greedy_set):
            greedy_set.add(x)
    
    greedy_sum = sum(weights[x] for x in greedy_set)
    best_set = set(greedy_set)
    best_sum = greedy_sum
    
    # Backtracking search with pruning
    # Try to improve by searching
    candidates = list(range(1, N + 1))  # in order of decreasing weight
    
    # Upper bound helper
    def upper_bound(current_sum, remaining_candidates):
        return current_sum + sum(weights[x] for x in remaining_candidates)
    
    # DFS with backtracking
    # We'll try to build a set by considering each candidate in order
    # At each step, we can include or exclude
    
    # For efficiency, we'll use a list and a set
    current_set = set()
    current_sum = 0.0
    
    # Precompute suffix sums of weights for upper bound
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    def dfs(idx, current_set, current_sum):
        nonlocal best_set, best_sum, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        # Update best
        if current_sum > best_sum + 1e-15:
            best_sum = current_sum
            best_set = set(current_set)
        
        if idx >= N:
            return
        
        # Upper bound pruning
        ub = current_sum + suffix_sum[idx + 1]
        if ub <= best_sum + 1e-15:
            return
        
        x = candidates[idx]
        
        # Try including x
        if is_valid_addition_fast(x, current_set):
            current_set.add(x)
            dfs(idx + 1, current_set, current_sum + weights[x])
            current_set.discard(x)
        
        # Try excluding x
        dfs(idx + 1, current_set, current_sum)
    
    # The full DFS over 256 candidates is too large. 
    # Instead, let's do a smarter search: start from greedy, then try local improvements.
    
    # Local search: try to add elements not in set, or swap
    # First, try to add any element that can be added
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if x not in best_set and is_valid_addition_fast(x, best_set):
                best_set.add(x)
                best_sum += weights[x]
                improved = True
                if time.time() - start_time > DEADLINE:
                    return sorted(best_set)
    
    # Try swapping: remove one element, add another (or more)
    # For each element in the set, try removing it and then greedily adding others
    for remove_x in list(best_set):
        if time.time() - start_time > DEADLINE:
            break
        temp_set = set(best_set)
        temp_set.discard(remove_x)
        temp_sum = best_sum - weights[remove_x]
        
        # Try to add elements
        for add_x in range(1, N + 1):
            if add_x in temp_set:
                continue
            if is_valid_addition_fast(add_x, temp_set):
                temp_set.add(add_x)
                temp_sum += weights[add_x]
        
        if temp_sum > best_sum + 1e-15:
            best_set = temp_set
            best_sum = temp_sum
    
    # Another round of adding
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if x not in best_set and is_valid_addition_fast(x, best_set):
                best_set.add(x)
                best_sum += weights[x]
                improved = True
                if time.time() - start_time > DEADLINE:
                    break
        if time.time() - start_time > DEADLINE:
            break
    
    # Verify no 3-AP
    def verify_no_3ap(s):
        s = sorted(s)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                c = 2 * s[j] - s[i]
                if c in s and c != s[i] and c != s[j]:
                    return False
                c2 = 2 * s[i] - s[j]
                if c2 in s and c2 != s[i] and c2 != s[j]:
                    return False
        return True
    
    # Final check
    if not verify_no_3ap(best_set):
        # Fallback to greedy
        return sorted(greedy_set)
    
    return sorted(best_set)
