"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 82  # seconds, leave 8s margin before 90s kill

    def check_deadline():
        return (time.time() - start_time) > TIME_LIMIT

    N = 512

    # Prime factorization sieve
    smallest_prime = [0] * (N + 1)
    for i in range(2, N + 1):
        if smallest_prime[i] == 0:
            for j in range(i, N + 1, i):
                if smallest_prime[j] == 0:
                    smallest_prime[j] = i

    def factorize(x):
        """Return dict of prime -> exponent."""
        factors = {}
        while x > 1:
            p = smallest_prime[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            factors[p] = cnt
        return factors

    # Precompute factorizations for all numbers 1..N
    all_factors = [None] * (N + 1)
    all_factors[1] = {}
    for i in range(2, N + 1):
        all_factors[i] = factorize(i)

    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    lcm_table[1] = list(range(N + 1))
    for i in range(2, N + 1):
        fi = all_factors[i]
        row = lcm_table[i]
        for j in range(1, N + 1):
            if j == i:
                row[j] = i
            elif j == 1:
                row[j] = i
            else:
                fj = all_factors[j]
                # lcm = product of p^max(ei, ej)
                l = 1
                # iterate over union of prime keys
                for p in set(list(fi.keys()) + list(fj.keys())):
                    e1 = fi.get(p, 0)
                    e2 = fj.get(p, 0)
                    l *= p ** max(e1, e2)
                row[j] = l

    # Get divisors of m
    def get_divisors(m):
        divs = set()
        i = 1
        while i * i <= m:
            if m % i == 0:
                divs.add(i)
                if i != m // i:
                    divs.add(m // i)
            i += 1
        return divs

    # Check if x can be added to set s
    def can_add(x, s):
        row = lcm_table[x]
        for y in s:
            if row[y] > N:
                return False
        return True

    # Max lcm of x with elements of s
    def max_lcm_with(x, s):
        row = lcm_table[x]
        mx = 1
        for y in s:
            l = row[y]
            if l > N:
                return N + 1
            if l > mx:
                mx = l
        return mx

    # Greedy construction: add elements one by one, choosing the one with min max-lcm
    def greedy_build(seed_set):
        current = set(seed_set)
        current.add(1)
        candidates = [x for x in range(2, N + 1) if x not in current]

        added_any = True
        while added_any and not check_deadline():
            added_any = False
            best_c = None
            best_score = N + 1
            for x in candidates:
                if x in current:
                    continue
                s = max_lcm_with(x, current)
                if s < best_score:
                    best_score = s
                    best_c = x
            if best_c is not None and best_score <= N:
                current.add(best_c)
                candidates.remove(best_c)
                added_any = True
        return current

    # Local search: 1-for-1 swaps
    def local_search_1for1(current, max_iters=50):
        current = set(current)
        for _ in range(max_iters):
            if check_deadline():
                break
            improved = False
            elems = sorted(current)
            for el in elems:
                if el == 1:
                    continue
                if check_deadline():
                    break
                trial = set(current)
                trial.discard(el)
                # Greedily add best compatible element
                best_c = None
                best_s = N + 1
                for x in range(2, N + 1):
                    if x in trial:
                        continue
                    s = max_lcm_with(x, trial)
                    if s < best_s:
                        best_s = s
                        best_c = x
                if best_c is not None and best_s <= N:
                    trial.add(best_c)
                if len(trial) > len(current):
                    current = trial
                    improved = True
                    break
            if not improved:
                break
        return current

    # Local search: 2-for-3 swaps
    def local_search_2for3(current, max_pairs=200):
        current = set(current)
        pairs_checked = 0
        improved = True
        while improved and not check_deadline() and pairs_checked < max_pairs:
            improved = False
            elems = sorted(current)
            for i in range(len(elems)):
                if check_deadline() or pairs_checked >= max_pairs:
                    break
                for j in range(i + 1, len(elems)):
                    if check_deadline() or pairs_checked >= max_pairs:
                        break
                    el1, el2 = elems[i], elems[j]
                    if el1 == 1 or el2 == 1:
                        continue
                    pairs_checked += 1
                    trial = set(current)
                    trial.discard(el1)
                    trial.discard(el2)
                    # Greedily add up to 3 new elements
                    new_cands = [x for x in range(2, N + 1) if x not in trial]
                    added_count = 0
                    added = True
                    while added and added_count < 3 and not check_deadline():
                        added = False
                        best_c = None
                        best_s = N + 1
                        for x in new_cands:
                            if x in trial:
                                continue
                            s = max_lcm_with(x, trial)
                            if s < best_s:
                                best_s = s
                                best_c = x
                        if best_c is not None and best_s <= N:
                            trial.add(best_c)
                            new_cands.remove(best_c)
                            added = True
                            added_count += 1
                    if len(trial) > len(current):
                        current = trial
                        improved = True
                        break
                if improved:
                    break
        return current

    # Precompute divisor counts
    div_count = [0] * (N + 1)
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            div_count[m] += 1

    # Strategy 1: Start with divisors of the number with most divisors
    best_m = max(range(1, N + 1), key=lambda x: div_count[x])
    seed1 = get_divisors(best_m)
    set1 = greedy_build(seed1)
    set1 = local_search_1for1(set1)
    set1 = local_search_2for3(set1)

    best_set = set(set1)
    best_size = len(best_set)

    # Strategy 2: Try other highly-divisor-count seeds
    nums_by_divs = sorted(range(1, N + 1), key=lambda x: -div_count[x])
    tried_seeds = {best_m}
    for m in nums_by_divs[:20]:
        if check_deadline():
            break
        if m in tried_seeds:
            continue
        tried_seeds.add(m)
        seed = get_divisors(m)
        s = greedy_build(seed)
        s = local_search_1for1(s, max_iters=30)
        if len(s) > best_size:
            best_set = set(s)
            best_size = len(s)

    # Strategy 3: Random restarts
    if not check_deadline():
        random.seed(42)
        for restart in range(10):
            if check_deadline():
                break
            m = random.randint(1, N)
            seed = get_divisors(m)
            # Add a few random compatible numbers
            for _ in range(5):
                x = random.randint(1, N)
                if can_add(x, seed):
                    seed.add(x)
            s = greedy_build(seed)
            s = local_search_1for1(s, max_iters=20)
            if len(s) > best_size:
                best_set = set(s)
                best_size = len(s)

    # Final: more aggressive local search on best
    if not check_deadline():
        best_set = local_search_1for1(best_set, max_iters=50)
        best_set = local_search_2for3(best_set, max_pairs=300)

    # Final verification
    final_list = sorted(best_set)
    valid = len(final_list) > 0
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            if lcm_table[final_list[i]][final_list[j]] > N:
                valid = False
                break
        if not valid:
            break

    if not valid:
        return list(range(1, int(math.isqrt(N)) + 1))

    return final_list

# EVOLVE-BLOCK-END
