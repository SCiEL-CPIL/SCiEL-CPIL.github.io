import time
import random
import bisect
from itertools import combinations

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin

    # Pre-compute primes
    primes = [x for x in range(2, 129) if all(x % i for i in range(2, int(x**0.5) + 1))]

    def is_square_free(n):
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    # Categorize numbers
    all_nums = list(range(2, 129))
    primes_set = set(primes)
    non_sqfree = [x for x in all_nums if x not in primes_set and not is_square_free(x)]
    sqfree_comp = [x for x in all_nums if x not in primes_set and is_square_free(x)]

    def count_distinct_prime_factors(n):
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

    def get_heuristic_score(x):
        if not is_square_free(x):
            return 0
        # Prefer numbers with fewer distinct prime factors (less collision risk)
        return -count_distinct_prime_factors(x)

    candidates = non_sqfree + sqfree_comp
    candidates.sort(key=lambda x: (get_heuristic_score(x), x))

    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def is_valid_insertion(S, x):
        idx = bisect.bisect_left(S, x)
        new_S = S[:idx] + [x] + S[idx:]
        return is_valid(new_S)

    # Build conflict graph: edge between i,j if adding both causes a product collision
    # For each pair, check if they can coexist in any valid sequence
    # Two numbers conflict if their inclusion in a sequence creates duplicate consecutive block products
    
    # Precompute: for each number, its "signature" in terms of prime factorization
    # Two numbers conflict if there exist positions where their products collide
    
    # Simpler approach: for each pair (a,b), check if {a,b} as a 2-element sequence is valid
    # If not, they conflict (cannot both be in any valid sequence)
    # But this is too strict - they might conflict only in certain contexts
    
    # Better: build a conflict graph where edge (i,j) means i and j cannot both be in the sequence
    # Two numbers i,j conflict if:
    # 1. i == j (not applicable since all distinct)
    # 2. Including both creates a duplicate product in some consecutive block
    
    # For two numbers a < b, they conflict if:
    # - a*b appears as a product of some other consecutive block when both are present
    # - Or some other collision occurs
    
    # Actually, the simplest sufficient condition: two numbers a,b conflict if a*b equals
    # some product of consecutive elements from a set that doesn't include a or b
    # But this depends on the full sequence...
    
    # Let's use a different approach: branch and bound with incremental validation
    
    # First, get a good initial solution using greedy + local search
    S = primes[:]
    S.sort()

    best_S = S[:]
    best_len = len(S)

    # 2. Greedy Insertion of Candidates
    for x in candidates:
        if time.time() - start_time > limit:
            break
        if is_valid_insertion(S, x):
            idx = bisect.bisect_left(S, x)
            S.insert(idx, x)
            if len(S) > best_len:
                best_S = S[:]
                best_len = len(S)

    # Local search to improve
    stuck_count = 0
    max_stuck = 30

    while time.time() - start_time < limit - 10:  # Reserve time for B&B
        improved = False

        # 1-out/1-in swap move
        sf_in_S = [x for x in S if x not in primes_set and is_square_free(x)]
        if not sf_in_S:
            sf_in_S = [x for x in S if x not in primes_set]

        if sf_in_S:
            y = random.choice(sf_in_S)
            S_without_y = [v for v in S if v != y]

            missing = [x for x in all_nums if x not in S_without_y and x != y]
            missing.sort(key=lambda x: (get_heuristic_score(x), x))

            for x in missing:
                if is_valid_insertion(S_without_y, x):
                    idx = bisect.bisect_left(S_without_y, x)
                    new_S = S_without_y[:idx] + [x] + S_without_y[idx:]
                    if len(new_S) > len(S):
                        S = new_S
                        if len(S) > best_len:
                            best_S = S[:]
                            best_len = len(S)
                        improved = True
                        break

        # 1-out/2-in move
        if not improved and sf_in_S:
            y = random.choice(sf_in_S)
            S_without_y = [v for v in S if v != y]

            missing = [x for x in all_nums if x not in S_without_y and x != y]
            missing.sort(key=lambda x: (get_heuristic_score(x), x))

            added_count = 0
            new_S = S_without_y[:]
            for x in missing:
                if added_count >= 2:
                    break
                if is_valid_insertion(new_S, x):
                    idx = bisect.bisect_left(new_S, x)
                    new_S.insert(idx, x)
                    added_count += 1

            if added_count >= 2:
                if is_valid(new_S):
                    S = new_S
                    if len(S) > best_len:
                        best_S = S[:]
                        best_len = len(S)
                    improved = True

        # Try random 1-in insertion
        if not improved:
            missing = [x for x in all_nums if x not in S]
            if missing:
                random.shuffle(missing)
                for x in missing:
                    if time.time() - start_time > limit:
                        break
                    if is_valid_insertion(S, x):
                        idx = bisect.bisect_left(S, x)
                        S.insert(idx, x)
                        if len(S) > best_len:
                            best_S = S[:]
                            best_len = len(S)
                        improved = True
                        break

        if not improved:
            stuck_count += 1
            if stuck_count >= max_stuck:
                S = best_S[:]
                non_prime_in_S = [x for x in S if x not in primes_set]
                if non_prime_in_S:
                    remove_count = min(random.randint(1, 2), len(non_prime_in_S))
                    to_remove = random.sample(non_prime_in_S, remove_count)
                    S = [x for x in S if x not in to_remove]
                stuck_count = 0
        else:
            stuck_count = 0

    # Now try branch-and-bound on the complement: which numbers to ADD
    # Start from best_S and try to add more numbers
    # Use B&B with pruning based on upper bound
    
    # For B&B: we want to find the maximum subset of all_nums that is valid
    # and contains as many elements as possible
    
    # Upper bound: len(best_S) + number of remaining numbers that could potentially be added
    # Pruning: if current_size + remaining <= best_len, prune
    
    # Sort all numbers by heuristic (prefer those likely to fit)
    all_sorted = sorted(all_nums, key=lambda x: (get_heuristic_score(x), x))
    
    # B&B state: (index into all_sorted, current_set, current_products)
    # But current_products can be large...
    
    # Alternative: use the conflict graph approach
    # Two numbers a, b conflict if they cannot both be in a valid sequence
    # We'll compute this by checking all pairs
    
    # For efficiency, precompute conflicts
    # Two numbers a < b conflict if:
    # 1. a*b equals some product of consecutive elements from a subset of {2,...,128}\{a,b}
    #    that could appear in a sequence containing both a and b
    # This is complex. Let's use a simpler necessary condition:
    # a and b conflict if a*b is not square-free in a certain sense, or if
    # there's another pair (c,d) with c*d = a*b where {c,d} is disjoint from {a,b}
    
    # Even simpler: just try to build the optimal solution using B&B with incremental validation
    # Focus on adding to the current best solution
    
    remaining = [x for x in all_sorted if x not in best_S]
    
    # Try to add numbers one by one using B&B
    # State: (pos, current_seq, current_products_set)
    # But products set can be huge...
    
    # Let's try a different B&B: decide for each number in remaining whether to include it
    # Use memoization or pruning based on upper bounds
    
    # Simpler approach: try all subsets of remaining of size k, starting from largest
    # But remaining could be ~50 elements, so 2^50 is too large
    
    # Let's stick with local search but add a more systematic approach:
    # Try to remove one element and add two, or remove two and add three, etc.
    
    # Enhanced local search with systematic removal
    for remove_count in range(1, 4):
        if time.time() - start_time > limit:
            break
        
        non_prime_in_S = [x for x in best_S if x not in primes_set]
        
        if len(non_prime_in_S) < remove_count:
            continue
        
        # Try removing each combination of remove_count elements
        # and see if we can add more
        for combo in combinations(non_prime_in_S, remove_count):
            if time.time() - start_time > limit:
                break
            
            S_without = [x for x in best_S if x not in combo]
            missing = [x for x in all_sorted if x not in S_without]
            
            added = 0
            new_S = S_without[:]
            for x in missing:
                if added >= remove_count + 1:  # Try to add at least remove_count+1
                    break
                if is_valid_insertion(new_S, x):
                    idx = bisect.bisect_left(new_S, x)
                    new_S.insert(idx, x)
                    added += 1
            
            if added > remove_count and is_valid(new_S):
                if len(new_S) > best_len:
                    best_S = new_S[:]
                    best_len = len(new_S)
    
    # Final verification
    if not is_valid(best_S):
        # Fallback to just primes
        best_S = primes[:]
        best_len = len(primes)
    
    return best_S
