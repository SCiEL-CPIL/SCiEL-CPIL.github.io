import math
import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85  # safety margin under 90s

    N = 256

    # Sieve for primes
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    primes = sieve(N)
    prime_set = set(primes)

    # Prime powers: largest p^k <= N for each prime p
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Base solution: 1 + all prime powers
    base_set = [1] + [prime_powers[p] for p in primes]
    base_sum = sum(base_set)

    # Precompute prime factorization for all numbers 1..N
    F = [None] * (N + 1)
    F[1] = frozenset()
    for i in range(2, N + 1):
        x = i
        factors = set()
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.add(d)
                while x % d == 0:
                    x //= d
            else:
                d += 1
        if x > 1:
            factors.add(x)
        F[i] = frozenset(factors)

    # Build all valid composites from prime subsets (up to 5 primes)
    composite_by_primes = {}  # frozenset(primes) -> list of values
    for c in range(2, N + 1):
        if c in prime_set:
            continue  # skip primes
        ps = F[c]
        if len(ps) < 2:
            continue  # prime powers already handled
        if len(ps) > 5:
            continue  # too many primes
        if ps not in composite_by_primes:
            composite_by_primes[ps] = []
        composite_by_primes[ps].append(c)

    # For each prime subset, pick the best composite
    best_for_subset = {}  # frozenset(primes) -> (value, gain)
    for ps, vals in composite_by_primes.items():
        base_cost = sum(prime_powers[p] for p in ps)
        best_val = max(vals)
        gain = best_val - base_cost
        if gain > 0:
            best_for_subset[ps] = (best_val, gain)

    subsets = list(best_for_subset.keys())

    def build_solution(sel_subsets, all_primes):
        result = [1]
        used = set()
        for ps in sel_subsets:
            val, _ = best_for_subset[ps]
            result.append(val)
            used |= ps
        for p in all_primes:
            if p not in used:
                result.append(prime_powers[p])
        return result

    def sum_of_solution(sel_subsets):
        used = set()
        total = 1  # for the 1
        for ps in sel_subsets:
            val, _ = best_for_subset[ps]
            total += val
            used |= ps
        for p in primes:
            if p not in used:
                total += prime_powers[p]
        return total

    # Multi-start greedy with different orderings
    best_sum = base_sum
    best_sol = base_set[:]
    best_selected = []

    def greedy_solve(order_key):
        """Greedy: pick subsets in given order if disjoint."""
        ordered = sorted(subsets, key=order_key)
        selected = []
        used_primes = set()
        for ps in ordered:
            if ps.isdisjoint(used_primes):
                selected.append(ps)
                used_primes |= ps
        return selected

    # Multiple greedy orderings
    orderings = [
        lambda ps: -best_for_subset[ps][1],  # by gain
        lambda ps: -best_for_subset[ps][1] / len(ps),  # by gain/size
        lambda ps: -best_for_subset[ps][1] / (len(ps) ** 1.5),  # by gain/size^1.5
        lambda ps: -best_for_subset[ps][0],  # by value
    ]

    for ok in orderings:
        sel = greedy_solve(ok)
        s = sum_of_solution(sel)
        if s > best_sum:
            best_sum = s
            best_sol = build_solution(sel, primes)
            best_selected = sel[:]

    # Local search with 1-exchange, 2-exchange, and 3-exchange
    def local_search(selected, max_iter=30):
        improved = True
        iteration = 0
        cur_sum = sum_of_solution(selected)
        cur_selected = selected[:]

        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            if time.time() > deadline - 8:
                break

            # 1-exchange: remove one, try alternatives
            for idx in range(len(cur_selected)):
                if time.time() > deadline - 8:
                    break
                removed = cur_selected[idx]
                remaining = [s for i, s in enumerate(cur_selected) if i != idx]
                rem_used = set()
                for s in remaining:
                    rem_used |= s

                for ps in subsets:
                    if not ps.isdisjoint(rem_used):
                        continue
                    if not (ps & removed):
                        continue
                    new_sel = remaining + [ps]
                    new_sum = sum_of_solution(new_sel)
                    if new_sum > cur_sum:
                        cur_sum = new_sum
                        cur_selected = new_sel[:]
                        improved = True
                        break

            # 2-exchange: swap two
            if len(cur_selected) >= 2:
                for i in range(len(cur_selected)):
                    for j in range(i + 1, len(cur_selected)):
                        if time.time() > deadline - 8:
                            break
                        si = cur_selected[i]
                        sj = cur_selected[j]
                        rest = [s for k, s in enumerate(cur_selected) if k != i and k != j]
                        rest_used = set()
                        for s in rest:
                            rest_used |= s

                        for ps in subsets:
                            if not ps.isdisjoint(rest_used | sj):
                                continue
                            if ps == si:
                                continue
                            new_sel = rest + [sj, ps]
                            new_sum = sum_of_solution(new_sel)
                            if new_sum > cur_sum:
                                cur_sum = new_sum
                                cur_selected = new_sel[:]
                                improved = True
                                break

            # 3-exchange: try replacing 3 with different 3
            if len(cur_selected) >= 3:
                for i in range(len(cur_selected)):
                    for j in range(i + 1, len(cur_selected)):
                        for k in range(j + 1, len(cur_selected)):
                            if time.time() > deadline - 8:
                                break
                            si, sj, sk = cur_selected[i], cur_selected[j], cur_selected[k]
                            rest = [s for m, s in enumerate(cur_selected) if m != i and m != j and m != k]
                            rest_used = set()
                            for s in rest:
                                rest_used |= s

                            # Try replacing si with another
                            for ps in subsets:
                                if not ps.isdisjoint(rest_used | sj | sk):
                                    continue
                                if ps == si:
                                    continue
                                new_sel = rest + [sj, sk, ps]
                                new_sum = sum_of_solution(new_sel)
                                if new_sum > cur_sum:
                                    cur_sum = new_sum
                                    cur_selected = new_sel[:]
                                    improved = True
                                    break

        return cur_selected, cur_sum

    # Run local search from each greedy start
    for ok in orderings:
        if time.time() > deadline - 10:
            break
        sel = greedy_solve(ok)
        sel, s = local_search(sel, max_iter=20)
        if s > best_sum:
            best_sum = s
            best_sol = build_solution(sel, primes)
            best_selected = sel[:]

    # Simulated annealing
    if time.time() < deadline - 15:
        cur_selected = best_selected[:]
        cur_sum = best_sum
        T = 50.0
        T_min = 0.1
        cooling = 0.95

        while T > T_min and time.time() < deadline - 10:
            T *= cooling
            # Random move: add, remove, or swap
            move_type = random.random()
            if move_type < 0.33:
                # Add a subset
                candidates = [ps for ps in subsets if ps.isdisjoint(set().union(*cur_selected) if cur_selected else set())]
                if candidates:
                    ps = random.choice(candidates)
                    new_sel = cur_selected + [ps]
                    new_sum = sum_of_solution(new_sel)
                    delta = new_sum - cur_sum
                    if delta > 0 or random.random() < math.exp(delta / T):
                        cur_selected = new_sel
                        cur_sum = new_sum
            elif move_type < 0.66:
                # Remove a subset
                if cur_selected:
                    idx = random.randint(0, len(cur_selected) - 1)
                    new_sel = [s for i, s in enumerate(cur_selected) if i != idx]
                    new_sum = sum_of_solution(new_sel)
                    delta = new_sum - cur_sum
                    if delta > 0 or random.random() < math.exp(delta / T):
                        cur_selected = new_sel
                        cur_sum = new_sum
            else:
                # Swap
                if len(cur_selected) >= 1:
                    idx = random.randint(0, len(cur_selected) - 1)
                    removed = cur_selected[idx]
                    remaining = [s for i, s in enumerate(cur_selected) if i != idx]
                    rem_used = set()
                    for s in remaining:
                        rem_used |= s
                    candidates = [ps for ps in subsets if ps.isdisjoint(rem_used) and ps != removed]
                    if candidates:
                        ps = random.choice(candidates)
                        new_sel = remaining + [ps]
                        new_sum = sum_of_solution(new_sel)
                        delta = new_sum - cur_sum
                        if delta > 0 or random.random() < math.exp(delta / T):
                            cur_selected = new_sel
                            cur_sum = new_sum

            if cur_sum > best_sum:
                best_sum = cur_sum
                best_sol = build_solution(cur_selected, primes)
                best_selected = cur_selected[:]

        # Final local search from SA result
        sel, s = local_search(best_selected, max_iter=15)
        if s > best_sum:
            best_sum = s
            best_sol = build_solution(sel, primes)
            best_selected = sel[:]

    # Final polish: try adding individual composites not in current solution
    current_factors = set()
    for x in best_sol:
        current_factors |= F[x]

    for c in range(2, N + 1):
        if time.time() > deadline - 5:
            break
        if c in best_sol:
            continue
        if F[c] & current_factors:
            continue
        new_sum = best_sum + c
        if new_sum > best_sum:
            best_sum = new_sum
            best_sol = best_sol + [c]
            current_factors |= F[c]

    # Final polish: try removing one element and finding best replacement
    for i in range(len(best_sol)):
        if time.time() > deadline - 3:
            break
        if best_sol[i] == 1:
            continue
        rest = [x for j, x in enumerate(best_sol) if j != i]
        rest_factors = set()
        for x in rest:
            rest_factors |= F[x]

        best_repl = None
        best_repl_val = 0
        for c in range(2, N + 1):
            if c in rest:
                continue
            if F[c] & rest_factors:
                continue
            if c > best_repl_val:
                best_repl_val = c
                best_repl = c

        if best_repl is not None and best_repl > best_sol[i]:
            new_sum = best_sum - best_sol[i] + best_repl
            if new_sum > best_sum:
                best_sum = new_sum
                best_sol = rest + [best_repl]

    # Verify pairwise coprimality
    def verify(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    if not verify(best_sol):
        if not verify(base_set):
            # Should never happen, but fallback
            best_sol = [1]
        else:
            best_sol = base_set[:]

    # Ensure distinct and in range
    best_sol = list(set(x for x in best_sol if 1 <= x <= 256))

    # Final type check: all must be Python int
    best_sol = [int(x) for x in best_sol]

    return best_sol
