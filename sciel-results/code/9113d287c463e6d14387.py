def construct():
    import time
    import math
    import random
    
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, leaving 5s safety margin
    
    def largest_prime_factor(n):
        """Return the largest prime factor of n."""
        if n < 2:
            return 1
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp
    
    def is_valid_set(S):
        """Check if set S is valid (no three distinct a,b,c with a|b*c)."""
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                b = S_list[j]
                for k in range(n):
                    if i == k or j == k:
                        continue
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    def get_violation(S_list):
        """Return a violating triple or None."""
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                b = S_list[j]
                for k in range(n):
                    if i == k or j == k:
                        continue
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return (a, b, c)
        return None
    
    def can_add(S, x):
        """Check if x can be added to S without violating the constraint."""
        S_list = list(S)
        n = len(S_list)
        # For each pair (b, c) in S, check if x divides b*c
        for i in range(n):
            b = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                c = S_list[j]
                if (b * c) % x == 0:
                    return False
        return True
    
    # Strategy: Start with numbers that have large prime factors (they're "safer")
    # A number with a large prime factor is less likely to divide products of others
    
    all_candidates = list(range(2, 257))
    
    # Sort by largest prime factor descending, then by value descending
    all_candidates.sort(key=lambda x: (largest_prime_factor(x), x), reverse=True)
    
    # Greedy construction: add numbers in order of safety
    S = set()
    for x in all_candidates:
        if time.time() > deadline - 10:  # Leave time for local search
            break
        if x in S:
            continue
        if can_add(S, x):
            S.add(x)
    
    # Local search to improve
    best_S = set(S)
    best_size = len(S)
    
    # Try to add more numbers
    for x in all_candidates:
        if time.time() > deadline - 5:
            break
        if x in S:
            continue
        if can_add(S, x):
            S.add(x)
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
    
    # Local search: try swapping elements
    if time.time() < deadline - 2:
        for _ in range(50):  # Limited iterations
            if time.time() > deadline - 1:
                break
            
            # Try to find a swap that improves the set
            improved = False
            S_list = list(S)
            for i, remove_x in enumerate(S_list):
                if time.time() > deadline - 1:
                    break
                S_temp = S - {remove_x}
                for add_y in all_candidates:
                    if add_y in S_temp or add_y == remove_x:
                        continue
                    if can_add(S_temp, add_y):
                        S_new = S_temp | {add_y}
                        if len(S_new) > len(S):
                            S = S_new
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    # Final validation
    if not is_valid_set(S):
        # Fallback: try to fix by removing violating elements
        S_list = list(S)
        while True:
            if time.time() > deadline - 0.5:
                break
            viol = get_violation(S_list)
            if viol is None:
                break
            a, b, c = viol
            S.remove(a)
            S_list = list(S)
    
    # Ensure we return a valid set
    if not is_valid_set(S):
        # Last resort: return primes (always valid since prime p | b*c implies p|b or p|c,
        # but if p is in the set and distinct from b,c, then p can't divide b*c unless p divides b or c,
        # which would mean b or c is a multiple of p, but since all are distinct and p is prime,
        # this can still happen... Actually primes alone might not be valid.
        # Let's use a safe fallback: numbers in (128, 256] that are prime powers
        # Actually, let's just return an empty set or a single element if needed
        # But we need nonempty. Return [2] as a safe single element.
        return [2]
    
    return list(S)
