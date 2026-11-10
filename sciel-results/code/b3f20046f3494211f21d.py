"""
Few prime factors across pairwise sums.

Strategy:
1. Try to find 12 distinct integers in [1, 10000] such that every pairwise sum
   is S-smooth for a small set of primes S (e.g. {2,3,5,7} or {2,3,5,7,11}).
   If such a set exists, the number of distinct primes dividing all pairwise
   sums is exactly |S| (or a subset thereof).
2. Precompute all S-smooth numbers <= 20000 (max pairwise sum).
3. Build a compatibility graph: two numbers a, b are compatible if a+b is S-smooth.
4. Use backtracking / greedy to find a clique of size 12 in this graph.
5. If no clique of size 12 is found for the smallest S, try larger S.
6. As a fallback, run a fast local search to minimize the number of distinct primes.
"""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 85.0
MAX_VAL = 10000
N = 12
MAX_SUM = 2 * MAX_VAL


def _elapsed():
    return time.time() - _START_TIME


def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS


# ---------- Precompute prime factor masks ----------
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def _build_factor_masks():
    """Precompute distinct-prime-factor bitmask for each n in [2, MAX_SUM]."""
    # First, collect all primes up to MAX_SUM via sieve
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, MAX_SUM + 1, i):
                is_prime[j] = False
    all_primes = [i for i in range(2, MAX_SUM + 1) if is_prime[i]]
    prime_to_bit = {p: i for i, p in enumerate(all_primes)}
    
    mask_table = [0] * (MAX_SUM + 1)
    for n in range(2, MAX_SUM + 1):
        x = n
        m = 0
        for p in _SMALL_PRIMES:
            if p * p > x:
                break
            if x % p == 0:
                m |= (1 << prime_to_bit[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            m |= (1 << prime_to_bit[x])
        mask_table[n] = m
    return mask_table, prime_to_bit, all_primes


_mask_table, _prime_to_bit, _all_primes = _build_factor_masks()


def _score(nums):
    """Number of distinct primes dividing pairwise sums of distinct-index pairs."""
    m = 0
    n = len(nums)
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            m |= _mask_table[ai + nums[j]]
    return bin(m).count("1")


# ---------- S-smooth generation ----------
def _generate_s_smooth(primes, limit):
    """Generate all numbers <= limit whose prime factors are all in `primes`."""
    smooth = set()
    stack = [1]
    while stack:
        x = stack.pop()
        if x > limit:
            continue
        smooth.add(x)
        for p in primes:
            if x * p <= limit:
                stack.append(x * p)
    return sorted(smooth)


def _find_clique(smooth_numbers, target_size, deadline_check):
    """
    Find a clique of size `target_size` in the compatibility graph where
    two numbers a, b are compatible if a+b is also in the smooth set.
    Uses backtracking with ordering by degree.
    """
    smooth_set = set(smooth_numbers)
    # Build adjacency: a and b are connected if a+b is in smooth_set
    # We only need numbers in [1, MAX_VAL]
    candidates = [x for x in smooth_numbers if 1 <= x <= MAX_VAL]
    if len(candidates) < target_size:
        return None
    
    # Build adjacency list
    adj = {x: set() for x in candidates}
    for i, a in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            if a + b in smooth_set:
                adj[a].add(b)
                adj[b].add(a)
    
    # Order by degree descending
    ordered = sorted(candidates, key=lambda x: len(adj[x]), reverse=True)
    pos_in_order = {x: i for i, x in enumerate(ordered)}
    
    # Backtracking to find clique of size target_size
    clique = []
    
    def backtrack(start_idx):
        if len(clique) == target_size:
            return True
        if _deadline_hit():
            return False
        for i in range(start_idx, len(ordered)):
            v = ordered[i]
            # Check if v is compatible with all in current clique
            if all(v in adj[u] for u in clique):
                clique.append(v)
                if backtrack(i + 1):
                    return True
                clique.pop()
        return False
    
    if backtrack(0):
        return sorted(clique)
    return None


def _try_smooth_solution(primes):
    """Try to find 12 numbers whose pairwise sums are all S-smooth."""
    smooth_sums = _generate_s_smooth(primes, MAX_SUM)
    smooth_set = set(smooth_sums)
    # Candidates are numbers in [1, MAX_VAL] such that for any other candidate,
    # the sum is in smooth_set. But we just need the clique search.
    result = _find_clique(smooth_sums, N, _deadline_hit)
    if result is not None and len(result) == N and len(set(result)) == N:
        return result
    return None


def _local_search(nums, budget_iters=5000):
    """Hill-climb single-element changes to reduce the raw objective."""
    current = sorted(nums)
    current_score = _score(current)
    iters = 0
    improved = True
    rng = random.Random(42)
    while improved and iters < budget_iters and not _deadline_hit():
        improved = False
        iters += 1
        n = len(current)
        for idx in range(n):
            old = current[idx]
            best_new = None
            best_new_score = current_score
            candidates = set()
            for d in range(-30, 31):
                v = old + d
                if 1 <= v <= MAX_VAL:
                    candidates.add(v)
            for k in (2, 3, 4, 5, 6, 7, 8, 10, 12):
                base = (old // k) * k
                for d in (-k, 0, k):
                    v = base + d
                    if 1 <= v <= MAX_VAL:
                        candidates.add(v)
            if old > 1:
                p2 = 1 << (old.bit_length() - 1)
                for v in (p2 // 2, p2, p2 * 2, p2 * 4):
                    if 1 <= v <= MAX_VAL:
                        candidates.add(v)
            for _ in range(10):
                v = rng.randint(1, MAX_VAL)
                candidates.add(v)

            for v in candidates:
                if v == old or v in current:
                    continue
                new_list = current[:]
                new_list[idx] = v
                ns = _score(new_list)
                if ns < best_new_score:
                    best_new_score = ns
                    best_new = (idx, v)
            if best_new is not None:
                idx, v = best_new
                current[idx] = v
                current = sorted(current)
                current_score = best_new_score
                improved = True
                break
    return current, current_score


def construct():
    rng = random.Random(987654321)
    best_result = None
    best_score = float("inf")
    
    # Phase 1: Try smooth solutions with increasing prime sets
    prime_sets = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
    ]
    
    for primes in prime_sets:
        if _deadline_hit():
            break
        result = _try_smooth_solution(primes)
        if result is not None:
            sc = _score(result)
            if sc < best_score:
                best_score = sc
                best_result = result[:]
            # If we found a solution with score <= len(primes), that's great
            if best_score <= len(primes):
                break
    
    # Phase 2: If no smooth solution found, try local search from diverse seeds
    if _deadline_hit():
        if best_result is None:
            best_result = list(range(1, N + 1))
        return best_result
    
    seeds = []
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 20, 30):
        s = []
        x = k
        while len(s) < N and x <= MAX_VAL:
            s.append(x)
            x += k
        if len(s) == N:
            seeds.append(s)
    # Powers of 2
    s = []
    x = 1
    while len(s) < N and x <= MAX_VAL:
        s.append(x)
        x *= 2
    if len(s) == N:
        seeds.append(s)
    # Random seeds
    for _ in range(30):
        seeds.append(sorted(rng.sample(range(1, MAX_VAL + 1), N)))
    
    seen = set()
    unique_seeds = []
    for s in seeds:
        t = tuple(sorted(s))
        if t not in seen:
            seen.add(t)
            unique_seeds.append(list(s))
    
    for seed in unique_seeds:
        if _deadline_hit():
            break
        if len(set(seed)) != N or not all(1 <= x <= MAX_VAL for x in seed):
            continue
        try:
            sc = _score(seed)
            if sc < best_score:
                best_score = sc
                best_result = seed[:]
            improved_list, sc2 = _local_search(sorted(seed), budget_iters=3000)
            if sc2 < best_score:
                best_score = sc2
                best_result = improved_list
        except Exception:
            continue
    
    # Final polish
    if best_result is not None and not _deadline_hit():
        cur = best_result[:]
        cur_score = _score(cur)
        if cur_score < best_score:
            best_score = cur_score
        improved_list, sc2 = _local_search(sorted(cur), budget_iters=5000)
        if sc2 < best_score:
            best_score = sc2
            best_result = improved_list
    
    # Validation
    assert best_result is not None, "no valid result found"
    assert len(set(best_result)) == N, "must have 12 distinct integers"
    assert all(1 <= x <= MAX_VAL for x in best_result), "must be <= 10000 and >= 1"
    
    return best_result
