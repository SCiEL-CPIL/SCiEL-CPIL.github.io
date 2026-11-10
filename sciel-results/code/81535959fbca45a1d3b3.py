"""Rewritten solver: integer-programming / SAT-style search with fast validity checks."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

MAX_N = 192
MAX_SUM = 3 * MAX_N  # max sum of 3 distinct <=192 is 190+191+192=573, but 3*192=576 is a safe bound
MAX_SUM_BOUND = 190 + 191 + 192


def construct():
    start = time.time()
    DEADLINE = start + 80.0  # safety margin under 90s

    # Precompute all triples of distinct values from 1..MAX_N, and the set of divisors each triple sum has.
    # For each value d in 1..MAX_N, store the set of triples (i,j,k) whose sum is divisible by d.
    # Actually, we want: for a candidate set S and a divisor d in S, no triple of OTHERS in S sums to a multiple of d.
    # We'll precompute: for each d, a list of (a,b,c) triples (distinct, all != d) such that (a+b+c) % d == 0.
    # Then when checking if d can be in S, we need to ensure no such triple is fully contained in S\{d}.

    # Build: bad_triples_for[d] = list of tuples (a,b,c) with a<b<c, all != d, and (a+b+c)%d==0
    bad_triples_for = [[] for _ in range(MAX_N + 1)]
    # Iterate over all triples (a,b,c) with 1<=a<b<c<=MAX_N
    # For each triple, compute s=a+b+c, and for each divisor d of s with 1<=d<=MAX_N and d not in {a,b,c},
    #   add (a,b,c) to bad_triples_for[d].
    # But iterating all divisors of s for each triple is expensive. Alternative:
    # For each d in 1..MAX_N, for each multiple m of d up to MAX_SUM_BOUND,
    #   find all triples (a,b,c) distinct, none equal to d, summing to m.
    # That's also expensive.

    # Simpler: For each d, we can compute bad_triples_for[d] by iterating all triples and checking.
    # Number of triples: C(192,3) ≈ 1.16M. For each, we check divisibility by each d that divides the sum.
    # Total work: for each triple, sum has at most ~100 divisors. So ~116M operations. Too slow in Python.

    # Alternative approach: We don't need all bad triples upfront. We can check validity incrementally.
    # Let's use a different strategy: maintain the set S, and when considering adding x, check:
    # 1. For each d in S: does any triple in S (excluding x, and also excluding d itself) sum to a multiple of d?
    #    But S changes as we add elements. We can maintain for each d in S, a count of "bad triples" that are fully in S\{d}.
    # 2. For x itself: does any triple in S (which will be the others) sum to a multiple of x?

    # Let's maintain:
    #   S: the current set
    #   For each d in S: bad_count[d] = number of triples (a,b,c) in S\{d} with a<b<c, (a+b+c)%d==0.
    # When we add x to S:
    #   - For x: we need to count triples in S (old S) that sum to a multiple of x. Call this cnt_x.
    #     If cnt_x > 0, x cannot be added.
    #   - For each d in old S: new bad triples for d are those triples that include x and two elements from old S\{d},
    #     summing to a multiple of d. So for each pair (u,v) in old S, u!=v, u!=d, v!=d,
    #     check if (x+u+v) % d == 0. If so, bad_count[d] increases by 1.
    #     If bad_count[d] becomes > 0, d is now "bad" and cannot stay in S... but wait, d is already in S.
    #     This means the set S + {x} is invalid if any bad_count[d] > 0 after update.
    # So the check for adding x is:
    #   - cnt_x (triples in old S summing to multiple of x) must be 0.
    #   - For all d in old S, the number of pairs (u,v) in old S, u!=v, u!=d, v!=d, with (x+u+v)%d==0 must be 0.
    #     (Because if any such pair exists, bad_count[d] would become >=1, making the set invalid.)

    # To make this fast, we can precompute for each d and each x, the count of pairs (u,v) in a given set...
    # But the set changes. Let's just compute on the fly. With |S| <= 48, the inner loops are small.

    # We'll use a greedy + local search approach with the above check.

    def can_add(S, x):
        """Check if x can be added to S without violating the constraint."""
        # Check 1: triples in S summing to multiple of x
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                for k in range(j + 1, n):
                    c = S[k]
                    if (a + b + c) % x == 0:
                        return False
        # Check 2: for each d in S, pairs (u,v) in S (u!=v, u!=d, v!=d) with (x+u+v)%d==0
        for d in S:
            for i in range(n):
                u = S[i]
                if u == d:
                    continue
                for j in range(i + 1, n):
                    v = S[j]
                    if v == d:
                        continue
                    if (x + u + v) % d == 0:
                        return False
        return True

    # Greedy fill with a given order
    def greedy_fill(order):
        S = []
        for x in order:
            if can_add(S, x):
                S.append(x)
        return S

    # Local search: try to add any missing element
    def local_search_add(S):
        S = list(S)
        improved = True
        while improved:
            improved = False
            for x in range(1, MAX_N + 1):
                if x not in S and can_add(S, x):
                    S.append(x)
                    improved = True
                    break
        return S

    # ILS: perturb by removing k random elements, then re-greedy fill
    def ils(best_S, time_limit):
        best = list(best_S)
        best_len = len(best)
        while time.time() < time_limit:
            # Perturb
            k = random.randint(1, min(5, len(best)))
            if k == 0:
                break
            remove_idx = random.sample(range(len(best)), k)
            remove_idx.sort(reverse=True)
            perturbed = best[:]
            for idx in remove_idx:
                perturbed.pop(idx)
            # Re-greedy fill
            # Use a randomized order biased toward larger numbers
            order = list(range(1, MAX_N + 1))
            random.shuffle(order)
            # Sort by a mix of size (descending) and randomness
            order.sort(key=lambda x: (-x, random.random()))
            new_S = greedy_fill(order)
            # Merge: start from perturbed, then try to add
            # Actually, let's just use the greedy fill result, but also try to add elements from perturbed that are missing
            # Better: start with perturbed, then fill
            fill_S = list(perturbed)
            for x in order:
                if x not in fill_S and can_add(fill_S, x):
                    fill_S.append(x)
            fill_S = local_search_add(fill_S)
            if len(fill_S) > best_len:
                best = fill_S
                best_len = len(best)
        return best

    # Try multiple strategies
    best = []

    # Strategy 1: Greedy descending
    order_desc = list(range(MAX_N, 0, -1))
    S1 = greedy_fill(order_desc)
    S1 = local_search_add(S1)
    best = S1[:]

    # Strategy 2: Greedy ascending
    order_asc = list(range(1, MAX_N + 1))
    S2 = greedy_fill(order_asc)
    S2 = local_search_add(S2)
    if len(S2) > len(best):
        best = S2[:]

    # Strategy 3: ILS
    if time.time() < DEADLINE - 10:
        best = ils(best, DEADLINE - 10)

    # Strategy 4: More ILS restarts with different perturbation sizes
    if time.time() < DEADLINE - 5:
        for _ in range(50):
            if time.time() > DEADLINE - 5:
                break
            k = random.randint(1, 8)
            if k > len(best):
                k = max(1, len(best) - 1)
            remove_idx = random.sample(range(len(best)), k)
            remove_idx.sort(reverse=True)
            perturbed = best[:]
            for idx in remove_idx:
                perturbed.pop(idx)
            order = list(range(1, MAX_N + 1))
            random.shuffle(order)
            fill_S = list(perturbed)
            for x in order:
                if x not in fill_S and can_add(fill_S, x):
                    fill_S.append(x)
            fill_S = local_search_add(fill_S)
            if len(fill_S) > len(best):
                best = fill_S

    return best
# EVOLVE-BLOCK-END
