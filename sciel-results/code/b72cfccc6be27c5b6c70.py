import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)
    
    # Precompute: for each pair (a,b), which x in candidates would create a conflict
    # Conflict: a | x*b, or b | x*a, or x | a*b
    # For a given set S, x is valid iff no pair (a,b) in S conflicts with x
    
    # For each x, precompute which pairs (a,b) conflict with x
    # But that's O(N^3) = ~16M, might be too slow. Let's use a different approach.
    
    # Key insight: for a set S represented as a bitset, we can check validity of x
    # by checking all pairs in S. But we can speed this up by precomputing for each x
    # the set of "forbidden pairs" - pairs (a,b) such that adding x would violate constraint.
    
    # Actually, let's use a more direct approach:
    # For each x, compute which other numbers y are "incompatible" with x in the sense that
    # for some third number z, one of the divisibility conditions fails.
    
    # Better approach: precompute conflict[x][y] = True if x and y cannot both be in S
    # because there exists a third candidate z such that the triple (x,y,z) is invalid.
    # But this is also complex. Let's just use the direct O(|S|^2) check but optimize it.
    
    # Precompute for each x, the list of pairs (a,b) from candidates such that x conflicts with that pair
    # i.e., a*b % x == 0 or x*b % a == 0 or x*a % b == 0
    
    # For efficiency, let's precompute:
    # divisors[a] = set of x such that x | a (i.e., a % x == 0)
    # multiples[a] = set of x such that a | x (i.e., x % a == 0)
    
    # Actually, the most efficient approach for local search:
    # Maintain S as a list. For each x not in S, compute "conflict_count[x]" = number of pairs (a,b) in S
    # that conflict with x. Then x can be added iff conflict_count[x] == 0.
    # When we add y to S, we update conflict_count for all x not in S.
    # When we remove y from S, we update conflict_count for all x not in S.
    
    # Let's precompute for each pair (a,b) in candidates, the set of x that conflict with that pair.
    # conflict_with_pair[a][b] = set of x such that (a*b)%x==0 or (x*b)%a==0 or (x*a)%b==0
    
    # This is O(N^3) precomputation = 255^3 ≈ 16.6M operations, which might take a few seconds.
    # Let's try a more memory-efficient approach: for each x, store a bitset of pairs that conflict.
    
    # Alternative: just do the direct check but make it fast with early termination.
    
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
    
    def check_valid(x, S_list):
        """Check if x can be added to S_list without violating constraints."""
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
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
            # Fast check
            if check_valid(x, S_list):
                S.add(x)
                S_list.append(x)
        return S, S_list
    
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
    
    for trial in range(80):
        if time.time() > deadline - 5:
            break
        order = list(candidates)
        random.shuffle(order)
        # Weight toward larger numbers
        order.sort(key=lambda x: -x + random.randint(-30, 30))
        S_trial, _ = greedy_build(order)
        if len(S_trial) > len(best_S):
            best_S = S_trial
            best_S_list = sorted(best_S)
    
    # Local search: 2-for-1 swaps with incremental validity
    # For each x in S, try removing it and adding two new elements
    # Use fast validity check
    
    if time.time() < deadline - 3:
        S_set = set(best_S)
        S_list = sorted(best_S)
        not_in_S = sorted(set(candidates) - S_set, reverse=True)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(S_list)):
                if time.time() > deadline - 2:
                    break
                x = S_list[i]
                # Try removing x
                S_without_x = [y for j, y in enumerate(S_list) if j != i]
                S_set_without_x = set(S_without_x)
                
                for y in not_in_S:
                    if time.time() > deadline - 2:
                        break
                    if y in S_set_without_x:
                        continue
                    # Check if y can be added to S_without_x
                    if not check_valid(y, S_without_x):
                        continue
                    
                    # Try adding y, then see if we can add another element
                    S_with_y = S_without_x + [y]
                    S_set_with_y = S_set_without_x | {y}
                    
                    # Try to add another element z
                    for z in not_in_S:
                        if time.time() > deadline - 2:
                            break
                        if z in S_set_with_y or z == y:
                            continue
                        if check_valid(z, S_with_y):
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
            if check_valid(x, S_list):
                best_S.add(x)
                S_list.append(x)
    
    result = sorted(best_S)
    # Ensure all are Python ints
    return [int(x) for x in result]
