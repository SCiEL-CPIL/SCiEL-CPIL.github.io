import time
from math import isqrt

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of distinct integers in 2..256 such that for every three 
    distinct members a, b, c, a does not divide b*c. Maximize cardinality.
    
    Strategy: 
    1. Start with all primes in [2, 256] (known valid, 54 elements).
    2. Try greedy addition of composites with efficient incremental checks.
    3. Use local search (swap) to try to improve cardinality.
    """
    deadline = time.time() + 75  # Leave 15s margin before 90s kill
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Generate all primes in [2, 256]
    primes = [p for p in range(2, 257) if is_prime(p)]
    composites = [n for n in range(2, 257) if not is_prime(n)]
    
    def can_add(c, S):
        """
        Check if c can be added to set S. S is assumed valid.
        Only need to check triples involving c.
        For each a in S (a != c): check a ∤ b*c for all b in S, b != a, b != c
        For each b in S (b != c): check c ∤ a*b for all a in S, a != b, a != c
        """
        S_list = list(S)
        for a in S_list:
            # Check: does a divide b*c for some b in S, b != a, b != c?
            for b in S_list:
                if b == a or b == c:
                    continue
                if (b * c) % a == 0:
                    return False
            # Check: does c divide a*b for some b in S, b != a, b != c?
            for b in S_list:
                if b == a or b == c:
                    continue
                if (a * b) % c == 0:
                    return False
        return True
    
    def is_valid_set(S):
        """Verify the entire set S is valid."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                b = S_list[j]
                if b == a:
                    continue
                for k in range(n):
                    c = S_list[k]
                    if c == a or c == b:
                        continue
                    if (b * c) % a == 0:
                        return False
        return True
    
    # Start with all primes
    S = set(primes)
    
    # Try to add composites greedily
    # Sort composites by number of distinct prime factors (fewer = easier to add)
    def count_distinct_prime_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                count += 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    composites_sorted = sorted(composites, key=lambda x: (count_distinct_prime_factors(x), x))
    
    for c in composites_sorted:
        if time.time() > deadline - 30:
            break
        if c not in S and can_add(c, S):
            S.add(c)
    
    # Local search: try swapping elements to increase cardinality
    # Try removing one element and adding two others
    best_S = set(S)
    best_size = len(S)
    
    # Try single swaps: remove x, add y
    all_numbers = set(range(2, 257))
    
    for _ in range(3):  # Multiple rounds of local search
        if time.time() > deadline - 15:
            break
        improved = False
        S_list = list(S)
        for x in S_list:
            if time.time() > deadline - 15:
                break
            S_without_x = S - {x}
            # Try adding each composite not in S
            for y in composites_sorted:
                if y in S_without_x:
                    continue
                if can_add(y, S_without_x):
                    new_S = S_without_x | {y}
                    if len(new_S) > best_size:
                        best_S = new_S
                        best_size = len(new_S)
                        S = new_S
                        improved = True
                        break
    
    # Try double swaps: remove x1, x2, add y1, y2
    if time.time() < deadline - 10:
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 10:
                    break
                x1, x2 = S_list[i], S_list[j]
                S_without = S - {x1, x2}
                # Try adding pairs of composites
                for y1 in composites_sorted:
                    if time.time() > deadline - 10:
                        break
                    if y1 in S_without:
                        continue
                    if not can_add(y1, S_without):
                        continue
                    temp_S = S_without | {y1}
                    for y2 in composites_sorted:
                        if time.time() > deadline - 10:
                            break
                        if y2 in temp_S or y2 == y1:
                            continue
                        if can_add(y2, temp_S):
                            new_S = temp_S | {y2}
                            if len(new_S) > best_size:
                                best_S = new_S
                                best_size = len(new_S)
                                S = new_S
    
    # Final verification
    if not is_valid_set(best_S):
        # Fallback to primes if something went wrong
        best_S = set(primes)
    
    result = [x for x in best_S]
    # Ensure all are Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
