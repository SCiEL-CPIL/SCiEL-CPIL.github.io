import time
import random

def construct():
    """
    Construct a 4-term-progression-free subset of [1, 512].
    Uses a local search strategy (greedy add + swap) to maximize cardinality.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    n = 512
    
    def creates_4ap(S, x):
        """
        Check if adding x to set S creates a nonconstant 4-term arithmetic progression.
        S is a set of integers.
        """
        # x can be the 1st, 2nd, 3rd, or 4th term of a 4-AP (a, a+d, a+2d, a+3d)
        
        # Case 1: x is the first term (a)
        # Need x+d, x+2d, x+3d in S
        # Max d: x + 3d <= n => d <= (n - x) // 3
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                return True
        
        # Case 2: x is the second term (a+d)
        # Need x-d, x+d, x+2d in S
        # Constraints: x-d >= 1 => d <= x-1
        #             x+2d <= n => d <= (n - x) // 2
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                return True
        
        # Case 3: x is the third term (a+2d)
        # Need x-2d, x-d, x+d in S
        # Constraints: x-2d >= 1 => d <= (x - 1) // 2
        #             x+d <= n => d <= n - x
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                return True
        
        # Case 4: x is the fourth term (a+3d)
        # Need x-3d, x-2d, x-d in S
        # Constraints: x-3d >= 1 => d <= (x - 1) // 3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d) in S and (x - 2*d) in S and (x - d) in S:
                return True
        
        return False

    def local_search(S, deadline):
        """
        Improve set S using local search (add valid elements, then swap).
        """
        S = set(S)
        improved = True
        
        while improved and (time.time() - start_time) < deadline - 2.0:
            improved = False
            
            # Phase 1: Try to add any valid element
            # We iterate through all numbers not in S
            # To make it faster, we can just check all 1..n and skip if in S
            for x in range(1, n + 1):
                if x in S:
                    continue
                if not creates_4ap(S, x):
                    S.add(x)
                    improved = True
            
            if improved:
                continue
            
            # Phase 2: Try to swap an element
            # If we can't add any new element, try swapping an existing element x
            # with a new element y such that S \ {x} U {y} is valid.
            # This is valid if y does not create a 4-AP with S \ {x}.
            
            # Optimization: Only try a limited number of swaps per epoch to avoid getting stuck
            # in a long loop of non-improving swaps (though we only swap if valid, 
            # so it always maintains validity, but might cycle in size? No, size is constant in swap.
            # But it might change the structure to allow future adds).
            
            candidates = [x for x in range(1, n + 1) if x not in S]
            # Shuffle candidates to avoid bias
            random.shuffle(candidates)
            
            swapped = False
            # Limit the number of swap attempts per phase to save time
            # If we find one, we restart the whole loop (Add phase)
            for y in candidates:
                if (time.time() - start_time) > deadline - 2.0:
                    break
                
                # Find if there's an x in S that can be removed to allow y
                # We iterate through a copy of S
                for x in list(S):
                    # Check if S - {x} + {y} is valid
                    # This is equivalent to: does y create a 4-AP with S - {x}?
                    # Note: S is valid. Removing x keeps it valid.
                    # Adding y is valid if no 4-AP involves y and elements of S-{x}.
                    
                    # We can't just call creates_4ap(S - {x}, y) directly because
                    # creating a new set S-{x} every time is expensive.
                    # Instead, we can modify the check or just accept the cost.
                    # Given n=512, S is small. S-{x} creation is O(|S|).
                    # creates_4ap is O(n).
                    # Total cost per check: O(|S| + n).
                    # With |S|~100, n=512, it's fast.
                    
                    S_temp = S - {x}
                    if not creates_4ap(S_temp, y):
                        # Perform swap
                        S.remove(x)
                        S.add(y)
                        swapped = True
                        break
                
                if swapped:
                    improved = True
                    break
                    
        return S

    def greedy_construction(n, order=None):
        selected = set()
        if order is None:
            order = range(1, n + 1)
        for x in order:
            if not creates_4ap(selected, x):
                selected.add(x)
        return selected

    def base3_construction(n):
        result = set()
        for x in range(1, n + 1):
            temp = x
            valid = True
            while temp > 0:
                if temp % 3 == 2:
                    valid = False
                    break
                temp //= 3
            if valid:
                result.add(x)
        return result

    best_set = set()
    best_size = 0
    
    # Strategy 1: Standard Greedy
    if time.time() - start_time < deadline - 10:
        S1 = greedy_construction(n)
        S1_opt = local_search(S1, deadline)
        if len(S1_opt) > best_size:
            best_size = len(S1_opt)
            best_set = S1_opt

    # Strategy 2: Randomized Greedy (multiple seeds)
    if time.time() - start_time < deadline - 10:
        for seed in [42, 123, 999]:
            if time.time() - start_time > deadline - 5:
                break
            random.seed(seed)
            order = list(range(1, n + 1))
            random.shuffle(order)
            S2 = greedy_construction(n, order)
            S2_opt = local_search(S2, deadline)
            if len(S2_opt) > best_size:
                best_size = len(S2_opt)
                best_set = S2_opt

    # Strategy 3: Base-3 Construction
    if time.time() - start_time < deadline - 10:
        S3 = base3_construction(n)
        S3_opt = local_search(S3, deadline)
        if len(S3_opt) > best_size:
            best_size = len(S3_opt)
            best_set = S3_opt

    # Final check
    if not best_set:
        best_set = greedy_construction(n)

    return sorted(best_set)
