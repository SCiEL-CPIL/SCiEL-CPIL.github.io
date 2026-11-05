import time
from math import lcm as math_lcm

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    N = 128
    # Precompute bad masks
    # bad_mask[a-1][b-1] is a bitmask of c (1..128) such that (a,b,c) is a bad triple
    bad_mask = [[0] * N for _ in range(N)]
    
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            if a == b:
                continue
            mask = 0
            # Optimization: If lcm(a,b) is fixed, we can check c faster?
            # lcm(a,b) == lcm(a,c) == lcm(b,c)
            # Let L = lcm(a,b). Then L must be a multiple of a, b, c.
            # Also lcm(a,c) = L implies c divides L and lcm(a,c)=L.
            # Actually, just iterating c is fine for N=128.
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if math_lcm(a, b) == math_lcm(a, c) == math_lcm(b, c):
                    mask |= (1 << (c - 1))
            bad_mask[a - 1][b - 1] = mask
            bad_mask[b - 1][a - 1] = mask

    def calc_conflict_mask(S):
        """Calculate the union of bad masks for all pairs in S."""
        mask = 0
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                mask |= bad_mask[a - 1][b - 1]
        return mask

    def try_add(x, S, conflict_mask):
        """
        Try to add x to S.
        Returns (success, new_conflict_mask).
        """
        # Check if x conflicts with any pair in S
        if (conflict_mask >> (x - 1)) & 1:
            return False, conflict_mask
        
        # If not, calculate new conflict mask
        # New conflicts come from pairs (x, y) for y in S
        new_mask = conflict_mask
        for y in S:
            new_mask |= bad_mask[x - 1][y - 1]
        
        return True, new_mask

    def num_distinct_primes(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    best_set = set()
    best_size = 0
    
    # Generate a few different orders
    # 1. Ascending
    # 2. Descending
    # 3. By number of distinct prime factors (desc)
    # 4. Random (fixed seed for reproducibility, or just use a few fixed permutations)
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: (-num_distinct_primes(x), x)),
        sorted(range(1, N + 1), key=lambda x: (num_distinct_primes(x), x)),
        sorted(range(1, N + 1), key=lambda x: (x % 2, -x)), # Evens then odds desc?
    ]
    
    # Add a few random-like orders using simple permutations
    # Since we can't use random without importing, we can use a simple LCG or just shuffle manually if we import random.
    # The prompt says NumPy and SciPy are available, and standard library. random is standard library.
    import random
    
    for _ in range(5):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)

    # Add more diverse orderings to explore different search landscapes
    # By divisibility chain length (numbers with many divisors first)
    def divisor_count(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    orders.append(sorted(range(1, N + 1), key=lambda x: (-divisor_count(x), x)))
    orders.append(sorted(range(1, N + 1), key=lambda x: (divisor_count(x), -x)))
    
    # By prime factorization complexity
    def prime_factor_complexity(n):
        total = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                total += 1
                n //= d
            d += 1
        if n > 1:
            total += 1
        return total

    orders.append(sorted(range(1, N + 1), key=lambda x: (-prime_factor_complexity(x), x)))
    orders.append(sorted(range(1, N + 1), key=lambda x: (prime_factor_complexity(x), x)))

    # Interleave small and large primes
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

    primes = [x for x in range(2, N + 1) if is_prime(x)]
    non_primes = [x for x in range(1, N + 1) if not is_prime(x)]
    interleaved = []
    for i in range(max(len(primes), len(non_primes))):
        if i < len(primes):
            interleaved.append(primes[i])
        if i < len(non_primes):
            interleaved.append(non_primes[i])
    orders.append(interleaved)

    for order in orders:
        if time.time() - start_time > deadline - 5:
            break
            
        S = set()
        conflict_mask = 0
        
        # Greedy Phase
        for x in order:
            ok, new_mask = try_add(x, S, conflict_mask)
            if ok:
                S.add(x)
                conflict_mask = new_mask
        
        # Local Search Phase
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline - 5:
                break
            
            # 1. Try to add any remaining element (try all, not just first)
            added_any = False
            for x in range(1, N + 1):
                if x not in S:
                    ok, new_mask = try_add(x, S, conflict_mask)
                    if ok:
                        S.add(x)
                        conflict_mask = new_mask
                        added_any = True
                        improved = True
                        # Don't break - try to add more in same iteration
                        if time.time() - start_time > deadline - 5:
                            break
            
            if added_any:
                continue
            
            # 2. Try 1-for-2 swaps
            # For each a in S, remove it, then try to add two new elements
            S_list = list(S)
            best_swap = None
            best_swap_size = len(S)
            for a in S_list:
                if time.time() - start_time > deadline - 5:
                    break
                    
                S_temp = S - {a}
                mask_temp = calc_conflict_mask(S_temp)
                
                # Try multiple candidate x values, pick the best one
                best_x_result = None
                for x in range(1, N + 1):
                    if x not in S_temp:
                        ok, new_mask = try_add(x, S_temp, mask_temp)
                        if ok:
                            S_x = S_temp | {x}
                            # Try to add as many more elements as possible
                            S_x_copy = set(S_x)
                            mask_x = new_mask
                            added_count = 0
                            for y in range(1, N + 1):
                                if y not in S_x_copy:
                                    ok2, mask_y = try_add(y, S_x_copy, mask_x)
                                    if ok2:
                                        S_x_copy.add(y)
                                        mask_x = mask_y
                                        added_count += 1
                                        if time.time() - start_time > deadline - 5:
                                            break
                            total_size = len(S_x_copy)
                            if total_size > best_swap_size:
                                best_swap_size = total_size
                                best_swap = (a, S_x_copy, mask_x)
                
                if best_swap:
                    break
                
            if best_swap:
                _, S, conflict_mask = best_swap
                improved = True
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Final verification pass: remove any elements that form bad triples
    # This ensures the returned set is valid even if search had a bug
    final_set = set(best_set)
    changed = True
    while changed:
        changed = False
        for a in list(final_set):
            for b in final_set:
                if a >= b:
                    continue
                for c in final_set:
                    if c <= b:
                        continue
                    if math_lcm(a, b) == math_lcm(a, c) == math_lcm(b, c):
                        # Remove the element that's "most redundant"
                        # Heuristic: remove the one with most conflicts
                        final_set.discard(c)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    
    return list(final_set)
