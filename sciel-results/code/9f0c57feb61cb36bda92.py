"""Pairwise-coprime sumset maximizer: enumerate A, greedy set-cover for B."""
import time
from math import gcd
from itertools import combinations

# ---- precompute primes up to 256 ----
def _sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]

PRIMES = _sieve(256)

def _pfactors(n):
    fs = set()
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            fs.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        fs.add(n)
    return fs

# prime-factor set for each possible sum 2..256
PF = {}
for s in range(2, 257):
    PF[s] = _pfactors(s)

def _pairwise_coprime(vals):
    seen = set()
    for v in vals:
        for p in PF[v]:
            if p in seen:
                return False
            seen.add(p)
    return True

def _valid_set(lst):
    if len(set(lst)) != 3:
        return False
    for x in lst:
        if x < 1 or x > 128:
            return False
    return True

def _sums(A, B):
    return [a + b for a in A for b in B]

def _distinct_count(A, B):
    return len(set(_sums(A, B)))

def _is_valid(A, B):
    if not _valid_set(A) or not _valid_set(B):
        return False
    return _pairwise_coprime(set(_sums(A, B)))

def construct():
    t0 = time.time()
    deadline = t0 + 55.0   # safety margin before 60s

    best = ([1, 7, 13], [4, 10, 16])
    best_score = 0
    for A, B in (best,):
        if _is_valid(A, B):
            best_score = max(best_score, _distinct_count(A, B))

    all_vals = list(range(1, 129))
    # Precompute all valid A sets
    valid_A = []
    for combo in combinations(all_vals, 3):
        if _valid_set(list(combo)):
            valid_A.append(combo)

    found_9 = False
    for A in valid_A:
        if time.time() > deadline:
            break
        # Build candidate B values
        # For each b, compute the set of sums and check pairwise coprime
        cand_b = []
        for b in all_vals:
            B = [b]
            sums = [a + b for a in A]
            if _pairwise_coprime(sums):
                cand_b.append(b)
        if len(cand_b) < 3:
            continue

        # Greedy set-cover: pick 3 distinct b's maximizing distinct pairwise-coprime sums
        # State: (chosen_B, current_sum_set, frozenset_of_primes_used)
        # We'll do a bounded DFS with pruning
        best_for_A = 0
        best_B_for_A = None

        # Precompute for each b: its sum-list
        b_sums = {}
        for b in cand_b:
            b_sums[b] = [a + b for a in A]

        # DFS: depth up to 3
        def dfs(chosen, sum_set, used_primes, depth):
            nonlocal best_for_A, best_B_for_A
            if depth == 3:
                if len(sum_set) > best_for_A:
                    best_for_A = len(sum_set)
                    best_B_for_A = list(chosen)
                return
            # Upper bound: max possible distinct sums from remaining
            # (rough: current + 3*3)
            if len(sum_set) + (3 - depth) * 3 <= best_for_A:
                return
            # start index to avoid duplicates
            start = chosen[-1] + 1 if chosen else 0
            for i in range(start, len(cand_b)):
                b = cand_b[i]
                new_sums = b_sums[b]
                new_sum_set = set(sum_set)
                new_sum_set.update(new_sums)
                # Check pairwise coprime with existing
                ok = True
                new_used = set(used_primes)
                for s in new_sums:
                    for p in PF[s]:
                        if p in new_used:
                            ok = False
                            break
                        new_used.add(p)
                    if not ok:
                        break
                if not ok:
                    continue
                chosen.append(b)
                dfs(chosen, new_sum_set, new_used, depth + 1)
                chosen.pop()
                if best_for_A == 9:
                    return
                if time.time() > deadline:
                    return

        dfs([], set(), set(), 0)

        if best_for_A > best_score:
            best_score = best_for_A
            best = (list(A), list(best_B_for_A) if best_B_for_A else best[1])
            if best_score == 9:
                found_9 = True
                break
        elif best_for_A == 9:
            found_9 = True
            best = (list(A), list(best_B_for_A) if best_B_for_A else best[1])
            break

    if time.time() > deadline:
        return [list(best[0]), list(best[1])]

    return [list(best[0]), list(best[1])]
