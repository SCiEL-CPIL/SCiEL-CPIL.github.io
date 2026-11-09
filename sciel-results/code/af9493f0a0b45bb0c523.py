import time
import math
import random
import itertools

def _smooth_set(limit, primes):
    s = set()
    for n in range(2, limit + 1):
        m = n
        for p in primes:
            while m % p == 0:
                m //= p
        if m == 1:
            s.add(n)
    return s

def _factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(nums):
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            all_primes |= _factorize(nums[i] + nums[j])
    return len(all_primes)

def _find_clique(cands, smooth, size, deadline):
    n = len(cands)
    adj = [set() for _ in range(n)]
    for i in range(n):
        ai = cands[i]
        for j in range(i + 1, n):
            if (ai + cands[j]) in smooth:
                adj[i].add(j)
                adj[j].add(i)
    order = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)
    best = []
    def bt(current, pool):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) == size:
            best = list(current)
            return
        if len(pool) < size - len(current):
            return
        for k, idx in enumerate(pool):
            if time.time() > deadline:
                return
            if not all(idx in adj[c] for c in current):
                continue
            new_pool = [x for x in pool[k + 1:] if x in adj[idx]]
            current.append(idx)
            bt(current, new_pool)
            current.pop()
            if best:
                return
    bt([], order)
    return best

def _local_improve(nums, deadline, max_iters=500):
    """Try to reduce distinct prime factors by swapping elements."""
    n = len(nums)
    best_primes = _count_distinct_primes(nums)
    nums = list(nums)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(n):
                if i == j:
                    continue
                # Try replacing nums[i] with nums[j] if different
                if nums[i] == nums[j]:
                    continue
                old_val = nums[i]
                nums[i] = nums[j]
                new_primes = _count_distinct_primes(nums)
                if new_primes < best_primes:
                    best_primes = new_primes
                    improved = True
                    if best_primes <= 3:
                        return nums, best_primes
                else:
                    nums[i] = old_val
        if not improved:
            break
    return nums, best_primes

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
    ]
    best_result = None
    best_score = float('inf')
    
    for ps in prime_sets:
        if time.time() > deadline - 10.0:
            break
        # Use larger candidate pool for smaller prime sets
        if len(ps) <= 2:
            N_list = [500, 1000, 2000, 4000]
        elif len(ps) <= 3:
            N_list = [300, 600, 1200]
        else:
            N_list = [200, 400, 800]
            
        for N in N_list:
            if time.time() > deadline - 10.0:
                break
            smooth = _smooth_set(2 * N, ps)
            # Only consider numbers whose double is smooth (more likely to form smooth sums)
            cands = [x for x in range(1, N + 1) if (2 * x) in smooth]
            if len(cands) < 12:
                continue
            res = _find_clique(cands, smooth, 12, deadline - 10.0)
            if res:
                out = [cands[i] for i in res]
                if len(out) == 12 and len(set(out)) == 12:
                    score = _count_distinct_primes(out)
                    if score < best_score:
                        best_score = score
                        best_result = out
                    if best_score <= len(ps):
                        return out
    
    if best_result:
        # Try local improvement
        improved, imp_score = _local_improve(best_result, deadline - 2.0)
        if imp_score < best_score:
            best_result = improved
            best_score = imp_score
        return best_result
    
    # Fallback: try with broader candidate selection
    for ps in prime_sets[:3]:
        if time.time() > deadline - 5.0:
            break
        N = 500
        smooth = _smooth_set(2 * N, ps)
        cands = list(range(1, N + 1))
        res = _find_clique(cands, smooth, 12, deadline - 5.0)
        if res:
            out = [cands[i] for i in res]
            if len(out) == 12 and len(set(out)) == 12:
                score = _count_distinct_primes(out)
                if score < best_score:
                    best_result = out
                    best_score = score
    
    if best_result:
        return best_result
    
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
