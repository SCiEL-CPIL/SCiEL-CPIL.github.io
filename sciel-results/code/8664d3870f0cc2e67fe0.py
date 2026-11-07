import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)
    
    # Precompute prime factorizations for all numbers 2..256
    # For each n, store a dict {prime: exponent}
    prime_factors = [None] * (N + 1)
    for n in range(2, N + 1):
        factors = {}
        temp = n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        prime_factors[n] = factors
    
    # Get all primes up to 256
    all_primes = sorted(set(p for pf in prime_factors[2:] if pf for p in pf))
    num_primes = len(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    
    # Precompute: for each pair (a, b), compute the "sum vector" v_p(a) + v_p(b)
    # and for each x, check if v_p(x) <= sum_vector for all p
    # conflict[x][a][b] = True if x conflicts with pair (a,b)
    # i.e., if a|x*b or b|x*a or x|a*b
    
    # More efficiently: for each x, precompute which (a,b) pairs conflict with x
    # conflict_pairs[x] = set of (a,b) tuples such that x conflicts with that pair
    
    # Actually, let's precompute for each x and each a, whether x conflicts with a in the sense that
    # for some b, the triple (x, a, b) is invalid. But we need the full triple check.
    
    # Let's precompute: for each x, a bitset (as a set of indices) of candidates y such that
    # the pair (x, y) creates a conflict with some z. But this is complex.
    
    # Better: precompute for each pair (i, j) where i < j, the set of x that conflict with that pair.
    # conflict_with_pair[i][j] = set of x indices such that adding x to {candidates[i], candidates[j]} violates constraint
    
    # Let's use a different approach: precompute for each x, the list of (a, b) pairs that conflict with x.
    # Then during local search, for a candidate x, we check if any pair in S conflicts with x.
    
    # Precompute conflict_map[x] = set of frozenset({a,b}) for all pairs (a,b) that conflict with x
    # But this uses too much memory. Instead, let's precompute:
    # For each x, for each a, whether x and a are "compatible" in the sense that no z makes them conflict.
    # This is still complex.
    
    # Let's just use the prime factorization approach for fast checking:
    # x conflicts with pair (a, b) if:
    #   (a | b*x) or (b | a*x) or (x | a*b)
    # Using prime factorizations:
    #   a | b*x iff for all p: v_p(a) <= v_p(b) + v_p(x)
    
    def divides_check(dividend_factors, divisor_factors):
        """Check if divisor_factors divides dividend_factors (i.e., for all p, v_p(divisor) <= v_p(dividend))"""
        for p, exp in divisor_factors.items():
            if dividend_factors.get(p, 0) < exp:
                return False
        return True
    
    def add_factors(f1, f2):
        """Add two factor dicts"""
        result = dict(f1)
        for p, exp in f2.items():
            result[p] = result.get(p, 0) + exp
        return result
    
    def check_valid_fast(x, S_list):
        """Check if x can be added to S_list without violating constraints using prime factorizations."""
        x_factors = prime_factors[x]
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            a_factors = prime_factors[a]
            for j in range(i + 1, n):
                b = S_list[j]
                b_factors = prime_factors[b]
                
                # Check if a | b*x
                # v_p(a) <= v_p(b) + v_p(x) for all p
                conflict = False
                for p, exp_a in a_factors.items():
                    if exp_a > b_factors.get(p, 0) + x_factors.get(p, 0):
                        conflict = True
                        break
                if conflict:
                    return False
                
                # Check if b | a*x
                conflict = False
                for p, exp_b in b_factors.items():
                    if exp_b > a_factors.get(p, 0) + x_factors.get(p, 0):
                        conflict = True
                        break
                if conflict:
                    return False
                
                # Check if x | a*b
                # v_p(x) <= v_p(a) + v_p(b) for all p
                conflict = False
                for p, exp_x in x_factors.items():
                    if exp_x > a_factors.get(p, 0) + b_factors.get(p, 0):
                        conflict = True
                        break
                if conflict:
                    return False
        
        return True
    
    def greedy_build(order, S_start=None):
        S = set(S_start) if S_start else set()
        S_list = list(S)
        for x in order:
            if time.time() > deadline:
                break
            if x in S:
                continue
            if check_valid_fast(x, S_list):
                S.add(x)
                S_list.append(x)
        return S, S_list
    
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
    
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    def num_divisors(n):
        count = 0
        for d in range(1, int(math.isqrt(n)) + 1):
            if n % d == 0:
                count += 1
                if d != n // d:
                    count += 1
        return count
    
    # Strategy 1: Descending order
    S1, _ = greedy_build(sorted(candidates, reverse=True))
    
    if time.time() > deadline:
        return sorted(S1)
    
    # Strategy 2: Primes first, then composites descending
    order2 = sorted(composites, reverse=True) + primes
    S2, _ = greedy_build(order2)
    
    if time.time() > deadline:
        return sorted(max(S1, S2, key=len))
    
    # Strategy 3: Numbers with few divisors first
    order3 = sorted(candidates, key=lambda x: (num_divisors(x), -x))
    S3, _ = greedy_build(order3)
    
    if time.time() > deadline:
        return sorted(max(S1, S2, S3, key=len))
    
    # Strategy 4: Randomized greedy with multiple trials
    random.seed(42)
    best_S = max(S1, S2, S3, key=len)
    best_S_list = sorted(best_S)
    
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        order = list(candidates)
        random.shuffle(order)
        # Weight toward larger numbers
        order.sort(key=lambda x: -x + random.randint(-30, 30))
        S_trial, _ = greedy_build(order)
        if len(S_trial) > len(best_S):
            best_S = S_trial
            best_S_list = sorted(best_S)
    
    # Local search: 2-for-1 swaps with fast validity check
    if time.time() < deadline - 5:
        S_set = set(best_S)
        S_list = sorted(best_S)
        not_in_S = sorted(set(candidates) - S_set, reverse=True)
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(S_list)):
                if time.time() > deadline - 3:
                    break
                x = S_list[i]
                # Try removing x
                S_without_x = [y for j, y in enumerate(S_list) if j != i]
                
                for y in not_in_S:
                    if time.time() > deadline - 3:
                        break
                    if y in S_set:
                        continue
                    # Check if y can be added to S_without_x
                    if not check_valid_fast(y, S_without_x):
                        continue
                    
                    S_with_y = S_without_x + [y]
                    
                    # Try to add another element z
                    for z in not_in_S:
                        if time.time() > deadline - 3:
                            break
                        if z in S_set or z == y:
                            continue
                        if check_valid_fast(z, S_with_y):
                            # Found improvement: remove x, add y and z
                            new_S = set(S_with_y) | {z}
                            if len(new_S) > len(best_S):
                                best_S = new_S
                                best_S_list = sorted(best_S)
                                S_set = set(best_S)
                                S_list = sorted(best_S)
                                not_in_S = sorted(set(candidates) - S_set, reverse=True)
                                improved = True
                                break
                    if improved:
                        break
    
    # Final: try to add any remaining elements
    if time.time() < deadline - 1:
        S_set = set(best_S)
        S_list = sorted(best_S)
        not_in_S = sorted(set(candidates) - S_set, reverse=True)
        for x in not_in_S:
            if time.time() > deadline - 1:
                break
            if check_valid_fast(x, S_list):
                best_S.add(x)
                S_list.append(x)
    
    result = sorted(best_S)
    # Ensure all are Python ints
    return [int(x) for x in result]
