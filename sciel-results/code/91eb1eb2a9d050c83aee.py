"""Construct maximum-size set with cardinality-separated subset sums."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    max_val = 256
    max_size = 60

    def try_add(elements, x, dp, current_size):
        """Try to add x to elements, check if cardinality separation holds."""
        if x in elements:
            return None

        new_dp = [dp[k] for k in range(len(dp))]
        while len(new_dp) <= current_size + 1:
            new_dp.append(0)

        for k in range(current_size, -1, -1):
            new_dp[k+1] |= (dp[k] << x)

        n_k = len(new_dp)
        for i in range(n_k):
            for j in range(i+1, n_k):
                if new_dp[i] & new_dp[j]:
                    return None

        return elements + [x], new_dp

    def verify_disjoint(dp, n):
        """Check that all dp[i] for i in 0..n are pairwise disjoint."""
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if dp[i] & dp[j]:
                    return False
        return True

    def count_bits(dp, n):
        """Count total number of achievable sums across all cardinalities."""
        return sum(bin(dp[i]).count('1') for i in range(n + 1))

    def build_dp(elements):
        """Build DP from scratch for a given set of elements."""
        n = len(elements)
        dp = [0] * (n + 1)
        dp[0] = 1
        for x in elements:
            for k in range(n, 0, -1):
                dp[k] |= (dp[k-1] << x)
        return dp

    def greedy_construct(ordering, time_limit):
        """Greedy construction with a given ordering."""
        elements = []
        dp = [1]
        in_set = set()
        
        for x in ordering:
            if time.time() > time_limit:
                break
            if len(elements) >= max_size:
                break
            if x in in_set:
                continue
            result = try_add(elements, x, dp, len(elements))
            if result:
                elements, dp = result
                in_set.add(x)
        
        return elements, dp

    def local_search(elements, dp, time_limit):
        """Local search to improve the set."""
        in_set = set(elements)
        n = len(elements)
        
        # Phase 1: Try to add any missing element
        for x in range(1, max_val + 1):
            if time.time() > time_limit:
                break
            if x in in_set:
                continue
            result = try_add(elements, x, dp, len(elements))
            if result:
                elements, dp = result
                in_set.add(x)
        
        if len(elements) >= max_size:
            return elements, dp
        
        # Phase 2: Systematic swap search
        # For each position, try swapping with each candidate
        candidates = [x for x in range(1, max_val + 1) if x not in in_set]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(elements)
            
            for i in range(n):
                if time.time() > time_limit:
                    break
                
                old_elem = elements[i]
                
                for new_elem in candidates:
                    if time.time() > time_limit:
                        break
                    
                    # Try the swap
                    new_elements = elements[:i] + [new_elem] + elements[i+1:]
                    new_dp = build_dp(new_elements)
                    
                    if verify_disjoint(new_dp, n):
                        # Check if this allows adding more elements
                        in_set_new = set(new_elements)
                        best_add = None
                        for x in range(1, max_val + 1):
                            if x in in_set_new:
                                continue
                            result = try_add(new_elements, x, new_dp, n)
                            if result:
                                best_add = x
                                break
                        
                        if best_add is not None:
                            # Accept the swap and add the new element
                            elements, dp = try_add(new_elements, best_add, new_dp, n)
                            in_set = set(elements)
                            candidates = [x for x in range(1, max_val + 1) if x not in in_set]
                            improved = True
                            break
                
                if improved:
                    break
        
        return elements, dp

    # Try multiple orderings
    orderings = [
        ("inc", list(range(1, max_val + 1))),
        ("dec", list(range(max_val, 0, -1))),
        ("sparse", [1, 3, 7, 15, 31, 63, 127, 255, 256] + list(range(2, 128))),
        ("prime", [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251] + list(range(4, 257, 2))),
        # New: power-of-2 based ordering
        ("powers", [1, 2, 4, 8, 16, 32, 64, 128, 256] + [x for x in range(1, 257) if x not in [1, 2, 4, 8, 16, 32, 64, 128, 256]]),
        # New: Fibonacci-like
        ("fib", [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 255, 256] + [x for x in range(1, 257) if x not in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 255, 256]]),
    ]

    best_elements = []
    best_dp = [1]

    for ordering_name, ordering in orderings:
        if time.time() > deadline:
            break

        elements, dp = greedy_construct(ordering, deadline)
        
        if len(elements) > len(best_elements):
            best_elements = list(elements)
            best_dp = dp
    
    # Local search on the best solution
    if len(best_elements) < max_size and time.time() < deadline:
        best_elements, best_dp = local_search(best_elements, best_dp, deadline)

    # Final verification
    if not best_elements:
        return [1]

    n = len(best_elements)
    verify_dp = [0] * (n + 1)
    verify_dp[0] = 1
    for x in best_elements:
        for k in range(n, 0, -1):
            verify_dp[k] |= (verify_dp[k-1] << x)

    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if verify_dp[i] & verify_dp[j]:
                return [1, 3, 7]

    return [int(x) for x in best_elements]
