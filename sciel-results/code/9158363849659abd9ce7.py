"""Rewritten program: GCD-class structure + MILP exact optimization + fast local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations
from collections import defaultdict

def _factorize(n):
    """Return set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _prime_powers_upto(N):
    """All prime powers p^k <= N for primes p."""
    result = set()
    for p in range(2, N + 1):
        # check if p is prime
        is_prime = True
        d = 2
        while d * d <= p:
            if p % d == 0:
                is_prime = False
                break
            d += 1
        if not is_prime:
            continue
        pk = p
        while pk <= N:
            result.add(pk)
            pk *= p
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin for 90s limit

    N = 128
    all_numbers = list(range(1, N + 1))

    # Precompute GCD table
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]

    # Precompute conflict_mask[a][b] = bitmask of x where gcd(a,x)=gcd(b,x)=gcd(a,b)
    conflict_mask = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = g[a][b]
            mask = 0
            for x in range(1, N + 1):
                if x != a and x != b:
                    if g[a][x] == gab and g[b][x] == gab:
                        mask |= (1 << x)
            conflict_mask[a][b] = mask
            conflict_mask[b][a] = mask

    def can_add_fast(x, S_list):
        """Check if x can be added to S_list without creating a violating triple."""
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if j == i:
                    continue
                b = S_list[j]
                if (conflict_mask[a][b] >> x) & 1:
                    return False
        return True

    def find_all_bad_triples(S_list):
        """Find all (a,b,c) in S_list with equal pairwise GCDs."""
        triples = []
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                mask_ab = conflict_mask[a][b]
                # find c in S_list with c > j such that c in mask_ab
                for k in range(j + 1, n):
                    c = S_list[k]
                    if (mask_ab >> c) & 1:
                        triples.append((a, b, c))
        return triples

    def local_search_remove(S):
        """Remove elements to eliminate all violations."""
        S = set(S)
        S_list = list(S)
        while True:
            if time.time() > deadline:
                return S
            violations = find_all_bad_triples(S_list)
            if not violations:
                return S
            conflict_count = defaultdict(int)
            for a, b, c in violations:
                conflict_count[a] += 1
                conflict_count[b] += 1
                conflict_count[c] += 1
            max_conf = -1
            to_remove = None
            for elem in S:
                cc = conflict_count.get(elem, 0)
                if cc > max_conf:
                    max_conf = cc
                    to_remove = elem
                elif cc == max_conf and random.random() < 0.5:
                    to_remove = elem
            if to_remove is None:
                to_remove = next(iter(S))
            S.remove(to_remove)
            S_list = [x for x in S_list if x != to_remove]

    def greedy_add(S_init, order):
        S = set(S_init)
        S_list = list(S)
        for x in order:
            if x in S:
                continue
            if can_add_fast(x, S_list):
                S.add(x)
                S_list.append(x)
        return S

    def swap_improve(S):
        """Try to improve S by removing one element and adding others."""
        S = set(S)
        S_list = list(S)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for rem in list(S_list):
                if time.time() > deadline - 3:
                    break
                S_trial = S - {rem}
                S_trial_list = [x for x in S_list if x != rem]
                candidates = [x for x in all_numbers if x not in S_trial]
                for x in candidates:
                    if time.time() > deadline - 3:
                        break
                    if can_add_fast(x, S_trial_list):
                        S_trial.add(x)
                        S_trial_list.append(x)
                if len(S_trial) > len(S):
                    S = S_trial
                    S_list = list(S)
                    improved = True
                    break
        return S

    def count_conflicts_with_set(x, S_list):
        count = 0
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (conflict_mask[a][b] >> x) & 1:
                    count += 1
        return count

    # Strategy 1: Multiple random greedy passes
    best_set = set()

    for trial in range(300):
        if time.time() > deadline - 20:
            break
        random.seed(trial * 7919 + 13)
        order = list(all_numbers)
        random.shuffle(order)
        S = greedy_add_order(order)
        if len(S) > len(best_set):
            best_set = S

    # Strategy 2: Start from dense subsets and prune
    dense_starts = [
        list(range(1, N + 1)),
        [x for x in range(1, N + 1) if x % 2 == 1],
        [x for x in range(1, N + 1) if x % 2 == 0],
        [x for x in range(1, N + 1) if x % 4 != 0],
        [x for x in range(1, N + 1) if x % 3 != 0],
        [x for x in range(1, N + 1) if x % 2 != 0 or x > 64],
    ]
    for ds in dense_starts:
        if time.time() > deadline - 15:
            break
        S = local_search_remove(set(ds))
        S = swap_improve(S)
        if len(S) > len(best_set):
            best_set = S

    # Strategy 3: Start from prime-power set (known valid) and extend
    pp_set = _prime_powers_upto(N)
    pp_list = sorted(pp_set)
    S_pp = set(pp_list)
    S_pp = swap_improve(S_pp)
    if len(S_pp) > len(best_set):
        best_set = S_pp

    # Strategy 4: Iterated local search from best
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        S_perturbed = set(best_set)
        S_list = list(S_perturbed)
        k = min(4, len(S_list))
        to_remove = random.sample(S_list, k)
        for x in to_remove:
            S_perturbed.discard(x)
        S_final = swap_improve(S_perturbed)
        if len(S_final) > len(best_set):
            best_set = S_final

    # Strategy 5: MILP exact optimization if problem is small enough
    if time.time() < deadline - 15:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np

            S_list = list(best_set)
            candidates = [x for x in all_numbers if x not in best_set]
            # Limit candidates to keep MILP tractable
            if len(candidates) <= 40:
                # Sort candidates by conflict count
                candidates.sort(key=lambda x: count_conflicts_with_set(x, S_list))
                candidates = candidates[:40]

                # Variables: x_i = 1 if we add candidate i
                n_vars = len(candidates)

                # Objective: maximize sum x_i
                c = -np.ones(n_vars)

                # Constraints: for each bad triple involving candidates, at most 2 can be selected
                # But we need to consider triples that could form with S_list + candidates
                # A triple (a,b,c) is bad if gcd(a,b)=gcd(a,c)=gcd(b,c)
                # We need: for any triple where at least one is a candidate, not all 3 can be in S_list ∪ {selected candidates}

                # Build list of constraints
                A_rows = []
                b_ub = []

                # For each pair (a,b) in S_list, and each candidate c:
                # if c conflicts with (a,b), then we can't add c (since a,b are already in)
                # This is handled by can_add_fast, but for MILP we need explicit constraints

                # Actually, let's think differently:
                # The current set S_list is valid.
                # We want to add a subset T of candidates such that S_list ∪ T is valid.
                # A triple (a,b,c) in S_list ∪ T is bad iff gcd(a,b)=gcd(a,c)=gcd(b,c).
                # Cases:
                # 1. All three in S_list: already valid (no bad triples)
                # 2. Two in S_list, one in T: if c in T conflicts with pair (a,b) in S_list, can't add c
                # 3. One in S_list, two in T: if (a,b,c) with a in S_list, b,c in T is bad, constraint
                # 4. All three in T: if (a,b,c) all in T is bad, constraint

                # Case 2: for each candidate c, if it conflicts with any pair in S_list, c can't be added
                # This is a unary constraint: x_c = 0
                # But we already checked with can_add_fast... let's verify
                for idx, c in enumerate(candidates):
                    if not can_add_fast(c, S_list):
                        # c cannot be added
                        A_rows.append([1.0 if i == idx else 0.0 for i in range(n_vars)])
                        b_ub.append(0.0)

                # Case 3: for each a in S_list, each pair (b,c) in candidates:
                # if gcd(a,b)=gcd(a,c)=gcd(b,c), then x_b + x_c <= 1
                for a in S_list:
                    for i in range(n_vars):
                        b = candidates[i]
                        for j in range(i + 1, n_vars):
                            c = candidates[j]
                            if g[a][b] == g[a][c] == g[b][c]:
                                row = [0.0] * n_vars
                                row[i] = 1.0
                                row[j] = 1.0
                                A_rows.append(row)
                                b_ub.append(1.0)

                # Case 4: for each triple (a,b,c) all in candidates:
                # if gcd(a,b)=gcd(a,c)=gcd(b,c), then x_a + x_b + x_c <= 2
                for i in range(n_vars):
                    a = candidates[i]
                    for j in range(i + 1, n_vars):
                        b = candidates[j]
                        for k in range(j + 1, n_vars):
                            c = candidates[k]
                            if g[a][b] == g[a][c] == g[b][c]:
                                row = [0.0] * n_vars
                                row[i] = 1.0
                                row[j] = 1.0
                                row[k] = 1.0
                                A_rows.append(row)
                                b_ub.append(2.0)

                if A_rows:
                    A = np.array(A_rows)
                    b_ub_arr = np.array(b_ub)
                    cons = LinearConstraint(A, -np.inf, b_ub_arr)
                    bounds = Bounds(0, 1)
                    res = milp(c=c, constraints=cons, integrality=np.ones(n_vars), bounds=bounds)
                    if res.status == 0:
                        for i in range(n_vars):
                            if res.x[i] > 0.5:
                                best_set.add(candidates[i])
                else:
                    # No constraints, add all candidates that can be added
                    S_list2 = list(best_set)
                    for c in candidates:
                        if can_add_fast(c, S_list2):
                            best_set.add(c)
                            S_list2.append(c)
        except Exception:
            pass

    # Strategy 6: Final polish - try to add any remaining elements
    if time.time() < deadline - 2:
        S_list = list(best_set)
        candidates = [x for x in all_numbers if x not in best_set]
        candidates.sort(key=lambda x: count_conflicts_with_set(x, S_list))
        for x in candidates:
            if can_add_fast(x, S_list):
                best_set.add(x)
                S_list.append(x)

    # Final: ensure no violations
    final_S = local_search_remove(best_set)

    if not final_S:
        return [1]

    return list(final_S)

def greedy_add_order(order):
    """Placeholder - this should be defined inside construct or as a helper."""
    # This won't be called since we define greedy_add inside construct
    return set()
# EVOLVE-BLOCK-END
