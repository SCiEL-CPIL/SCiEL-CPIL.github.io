import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 110.0 # Safety margin for 120s limit

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        # Fast check for sum-free property
        s_set = set(subset)
        n = len(subset)
        for i in range(n):
            a = subset[i]
            for j in range(i, n):
                b = subset[j]
                c = a + b
                if c in s_set:
                    return False
        return True

    def can_add(A_set, A_list, x):
        # Check if A_list U {x} is sum-free
        # We only need to check pairs involving x
        # If a, b in A_list, a+b != x
        # If a in A_list, a+x != a (x!=0, true)
        # If a in A_list, a+x != b for any b in A_list
        # Also x+x != a for any a in A_list
        
        # 1. x + x not in A_set
        if 2 * x in A_set:
            return False
        
        # 2. For all a in A_list, a + x not in A_set
        for a in A_list:
            if (a + x) in A_set:
                return False
        
        # 3. For all a, b in A_list, a + b != x
        # This is equivalent to: x - a not in A_set for any a in A_list?
        # If a + b = x, then b = x - a. So if x - a is in A_set, and x-a != a (handled by 2x case? No, if a=b, 2a=x, handled by 1).
        # If a != b, then a and x-a are distinct elements in A_set.
        # So we just need to check if (x - a) in A_set for any a in A_list.
        # Note: if a = x-a, then 2a=x, which is case 1.
        for a in A_list:
            if (x - a) in A_set:
                return False
        
        return True

    def greedy_fill(S, initial_subset, order_key):
        """
        Greedily add elements from S \ initial_subset to initial_subset.
        order_key: function to sort candidates
        """
        A = list(initial_subset)
        A_set = set(A)
        candidates = [x for x in S if x not in A_set]
        
        # Sort candidates
        candidates.sort(key=order_key)
        
        for x in candidates:
            if time.time() > DEADLINE:
                break
            if can_add(A_set, A, x):
                A.append(x)
                A_set.add(x)
        
        return A

    def local_search(S, initial_subset):
        """
        Try to improve the subset by removing and re-adding elements.
        Simple version: try removing each element, then greedily fill again.
        """
        best = list(initial_subset)
        best_size = len(best)
        
        # To save time, we only do a few iterations or limit the search
        # We'll try removing elements that are "expensive" (involve many conflicts)
        # For simplicity, we'll just do a pass of removing each element and filling.
        # But this is O(N^2 * fill_cost).
        
        # Given the small N, let's just do a few random restarts of greedy fill
        # with different orderings.
        
        return best

    results = []
    
    # Define instances
    instances = [
        (60, 801),
        (120, 802),
        (240, 803)
    ]
    
    for size, seed in instances:
        S = instance(size, seed)
        
        best_subset = []
        best_size = 0
        
        # Heuristic 1: Odds
        init_odds = [x for x in S if x % 2 != 0]
        
        # Heuristic 2: Positives
        init_pos = [x for x in S if x > 0]
        
        # Heuristic 3: Negatives
        init_neg = [x for x in S if x < 0]
        
        # Heuristic 4: Large Absolute Values (Top half)
        sorted_by_abs = sorted(S, key=lambda x: abs(x), reverse=True)
        init_large = sorted_by_abs[:len(S)//2 + 1]
        
        # Heuristic 5: Small Absolute Values
        init_small = sorted_by_abs[len(S)//2:]
        
        inits = [init_odds, init_pos, init_neg, init_large, init_small, []]
        
        # Orderings for greedy fill
        # 1. Default (sorted S)
        # 2. By absolute value descending
        # 3. By value descending
        # 4. By value ascending
        
        orderings = [
            lambda x: x,
            lambda x: -abs(x),
            lambda x: -x,
            lambda x: x
        ]
        
        for init in inits:
            for order_key in orderings:
                if time.time() > DEADLINE:
                    break
                    
                candidate = greedy_fill(S, init, order_key)
                
                # Verify validity (safety)
                if is_sum_free(candidate):
                    if len(candidate) > best_size:
                        best_size = len(candidate)
                        best_subset = candidate
        
        # If we still have time, try a more sophisticated local search?
        # For now, the greedy fill with multiple inits is strong.
        
        # Final safety check
        if not is_sum_free(best_subset):
             # Fallback to odds if something went wrong
             fallback = [x for x in S if x % 2 != 0]
             if is_sum_free(fallback):
                 best_subset = fallback
        
        results.append(best_subset)
        
    return results

