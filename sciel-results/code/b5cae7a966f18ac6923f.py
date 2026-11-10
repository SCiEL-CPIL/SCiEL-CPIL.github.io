import time
import random
import bisect
from math import gcd

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin

    # Pre-compute primes
    primes = [x for x in range(2, 129) if all(x % i for i in range(2, int(x**0.5) + 1))]
    primes_set = set(primes)

    def is_square_free(n):
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    # Precompute prime factor counts for heuristic
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    all_nums = list(range(2, 129))
    non_sqfree = [x for x in all_nums if x not in primes_set and not is_square_free(x)]
    sqfree_comp = [x for x in all_nums if x not in primes_set and is_square_free(x)]

    def get_heuristic_score(x):
        # Prefer square-free composites, then fewer prime factors (more "unique" contributions)
        if not is_square_free(x):
            return (2, 0, x)  # deprioritize non-square-free
        return (1, prime_factor_count(x), x)

    candidates = non_sqfree + sqfree_comp
    candidates.sort(key=get_heuristic_score)

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
        # Check only blocks that include the new element x
        # Blocks: [i..idx-1], [idx..j], [idx..idx], etc.
        # Actually need to check all blocks in the new sequence, but we can be smart:
        # Only blocks that include position idx can potentially create new collisions
        # with existing products, but also new blocks among themselves.
        # Simplest correct approach: build new list and check all blocks involving idx
        n = len(S) + 1
        # Collect all products from blocks NOT containing idx (these are unchanged)
        # Then add blocks containing idx and check for duplicates
        products = set()
        # Products from left part (blocks ending before idx)
        for i in range(idx):
            prod = 1
            for j in range(i, idx):
                prod *= S[j]
                products.add(prod)
        # Products from right part (blocks starting after idx)
        for i in range(idx, n):
            prod = 1
            for j in range(i, n):
                val = S[j] if j < idx else (x if j == idx else S[j-1])
                prod *= val
                products.add(prod)
        # Now check blocks containing idx
        for i in range(0, idx + 1):
            prod = 1
            for j in range(i, n):
                val = S[j] if j < idx else (x if j == idx else S[j-1])
                prod *= val
                if prod in products:
                    return False
                products.add(prod)
        return True

    # 1. Initialize with Primes
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

    # 3. Local Search with multiple move types
    while time.time() - start_time < limit:
        improved = False

        # Get non-prime elements in S for potential removal
        non_prime_in_S = [x for x in S if x not in primes_set]

        # 1-out/2-in move
        if non_prime_in_S:
            y = random.choice(non_prime_in_S)
            S_without_y = [v for v in S if v != y]

            missing = [x for x in all_nums if x not in S_without_y and x != y]
            missing.sort(key=get_heuristic_score)

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

        # 2-out/3-in move
        if not improved and len(non_prime_in_S) >= 2:
            y1, y2 = random.sample(non_prime_in_S, 2)
            S_without = [v for v in S if v != y1 and v != y2]

            missing = [x for x in all_nums if x not in S_without and x != y1 and x != y2]
            missing.sort(key=get_heuristic_score)

            added_count = 0
            new_S = S_without[:]
            for x in missing:
                if added_count >= 3:
                    break
                if is_valid_insertion(new_S, x):
                    idx = bisect.bisect_left(new_S, x)
                    new_S.insert(idx, x)
                    added_count += 1

            if added_count >= 3:
                if is_valid(new_S):
                    S = new_S
                    if len(S) > best_len:
                        best_S = S[:]
                        best_len = len(S)
                    improved = True

        # 1-out/1-in swap move (NEW: escape local optima)
        if not improved and non_prime_in_S:
            y = random.choice(non_prime_in_S)
            S_without_y = [v for v in S if v != y]

            missing = [x for x in all_nums if x not in S_without_y and x != y]
            missing.sort(key=get_heuristic_score)

            new_S = S_without_y[:]
            for x in missing:
                if is_valid_insertion(new_S, x):
                    idx = bisect.bisect_left(new_S, x)
                    new_S.insert(idx, x)
                    if is_valid(new_S):
                        S = new_S
                        if len(S) > best_len:
                            best_S = S[:]
                            best_len = len(S)
                        improved = True
                    break

        # 2-out/2-in swap move (NEW: replace two elements with two different ones)
        if not improved and len(non_prime_in_S) >= 2:
            y1, y2 = random.sample(non_prime_in_S, 2)
            S_without = [v for v in S if v != y1 and v != y2]

            missing = [x for x in all_nums if x not in S_without and x != y1 and x != y2]
            missing.sort(key=get_heuristic_score)

            added_count = 0
            new_S = S_without[:]
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

    return best_S
