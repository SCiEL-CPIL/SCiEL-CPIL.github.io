"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

DEADLINE = 85  # seconds, safety margin before 90s kill

def _factorize(n):
    """Return dict of prime -> exponent for n."""
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f

def _violates(a, b, c):
    """Check if a divides b*c."""
    prod = b * c
    return prod % a == 0

def _is_valid_set(S):
    """Check if set S satisfies the constraint for all triples of distinct members."""
    S = list(S)
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S[i], S[j], S[k]
                if (b * c) % a == 0:
                    return False
    return True

def _count_violations(S):
    """Count number of violating triples in S."""
    S = list(S)
    n = len(S)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S[i], S[j], S[k]
                if (b * c) % a == 0:
                    count += 1
    return count

def _greedy_construct(order, candidates):
    """Greedily add elements from order if they don't violate with existing set."""
    S = []
    for a in order:
        if a not in candidates:
            continue
        ok = True
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                b, c = S[i], S[j]
                if (b * c) % a == 0 or (a * b) % c == 0 or (a * c) % b == 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            S.append(a)
    return S

def _local_improve(S, candidates, deadline):
    """Try to improve S by adding/removing/swapping elements."""
    S = list(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each candidate
        for a in candidates:
            if a in S:
                continue
            ok = True
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    b, c = S[i], S[j]
                    if (b * c) % a == 0 or (a * b) % c == 0 or (a * c) % b == 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                S.append(a)
                improved = True
        # Try removing one and adding two (2-for-1 swap)
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                S_trial = [x for k, x in enumerate(S) if k != i and k != j]
                for a in candidates:
                    if a in S_trial:
                        continue
                    ok = True
                    for x in range(len(S_trial)):
                        for y in range(x + 1, len(S_trial)):
                            b, c = S_trial[x], S_trial[y]
                            if (b * c) % a == 0 or (a * b) % c == 0 or (a * c) % b == 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        S_trial.append(a)
                        if len(S_trial) > len(S):
                            S = S_trial
                            improved = True
                            break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    candidates = list(range(2, 257))
    
    # Pre-compute factorizations
    factors = {n: _factorize(n) for n in candidates}
    
    # Strategy 1: Deterministic greedy by various orderings
    best = []
    
    # Order by number of divisors (fewer divisors = less likely to divide products)
    def num_divisors(n):
        cnt = 0
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                cnt += 1
                if i != n // i:
                    cnt += 1
        return cnt
    
    div_counts = {n: num_divisors(n) for n in candidates}
    
    # Try multiple deterministic orderings
    orderings = [
        sorted(candidates, key=lambda x: div_counts[x]),  # fewest divisors first
        sorted(candidates, key=lambda x: -div_counts[x]), # most divisors first
        sorted(candidates, key=lambda x: x),               # ascending
        sorted(candidates, key=lambda x: -x),              # descending
        sorted(candidates, key=lambda x: (div_counts[x], x)),
        sorted(candidates, key=lambda x: (len(factors[x]), x)),  # fewest prime factors
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy_construct(order, candidates)
        S = _local_improve(S, candidates, deadline)
        if len(S) > len(best):
            best = S
    
    # Strategy 2: Multi-start randomized greedy
    rng = random.Random(42)
    num_starts = 50
    for trial in range(num_starts):
        if time.time() > deadline:
            break
        
        # Random perturbation of ordering
        order = candidates[:]
        # Weight by inverse divisor count to prefer "safe" elements
        weights = [1.0 / (div_counts[x] + 1) for x in order]
        # Random shuffle with bias
        shuffled = []
        remaining = list(range(len(order)))
        while remaining:
            total_w = sum(weights[i] for i in remaining)
            r = rng.random() * total_w
            cum = 0
            for idx, i in enumerate(remaining):
                cum += weights[i]
                if r <= cum:
                    shuffled.append(order[i])
                    remaining.remove(i)
                    break
            else:
                shuffled.append(order[remaining[-1]])
                remaining.pop()
        
        S = _greedy_construct(shuffled, candidates)
        S = _local_improve(S, candidates, deadline)
        if len(S) > len(best):
            best = S
    
    # Final local improvement on best
    best = _local_improve(best, candidates, deadline)
    
    # Verify
    assert _is_valid_set(best), "Invalid set returned"
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in best)
    assert len(set(best)) == len(best)
    assert all(2 <= x <= 256 for x in best)
    
    return list(best)
# EVOLVE-BLOCK-END
