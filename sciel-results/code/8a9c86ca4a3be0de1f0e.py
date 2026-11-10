import time
import random
import math
from itertools import combinations

_START = time.time()
_DEADLINE = 85.0  # 5s safety margin before 90s kill

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

# Precompute smallest prime factor up to 20000
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _sieve[_j] == _j:
                _sieve[_j] = _i

# Prime factor set cache
_pf_cache = [frozenset() for _ in range(_MAX_SUM + 1)]
for _n in range(2, _MAX_SUM + 1):
    _x = _n
    _f = set()
    while _x > 1:
        _p = _sieve[_x]
        _f.add(_p)
        while _x % _p == 0:
            _x //= _p
    _pf_cache[_n] = frozenset(_f)

def _pf(n):
    return _pf_cache[n]

def _score_for_prime_set(S, allowed):
    """Count distinct primes in pairwise sums that are NOT in allowed set."""
    foreign = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            s = S[i] + S[j]
            if s > _MAX_SUM:
                # factor manually
                x = s
                while x > 1:
                    p = _sieve[x] if x <= _MAX_SUM else None
                    if p is None:
                        # fallback: trial division
                        d = 2
                        while d * d <= x:
                            if x % d == 0:
                                p = d
                                break
                            d += 1
                        if p is None:
                            p = x
                    if p not in allowed:
                        foreign.add(p)
                    while x % p == 0:
                        x //= p
                    if x > 1 and x <= _MAX_SUM:
                        if x not in allowed:
                            foreign.add(x)
                    elif x > 1:
                        if x not in allowed:
                            foreign.add(x)
            else:
                for p in _pf(s):
                    if p not in allowed:
                        foreign.add(p)
    return len(foreign)

def _total_foreign_count(S, allowed):
    """Total count of foreign prime occurrences (for annealing pressure)."""
    total = 0
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            s = S[i] + S[j]
            if s <= _MAX_SUM:
                for p in _pf(s):
                    if p not in allowed:
                        total += 1
            else:
                x = s
                while x > 1:
                    d = 2
                    while d * d <= x:
                        if x % d == 0:
                            break
                        d += 1
                    if d * d > x:
                        d = x
                    if d not in allowed:
                        total += 1
                    while x % d == 0:
                        x //= d
    return total

def _try_prime_set(allowed_primes, N=12, MAX_VAL=10000):
    """
    Given a set of allowed primes, find N distinct positive integers <= MAX_VAL
    such that all pairwise sums use only primes from allowed_primes.
    Returns (cost, S) where cost = number of distinct foreign primes.
    """
    allowed = frozenset(allowed_primes)
    
    # Generate candidate values whose all prime factors (and sums with others) 
    # are "smooth" over allowed primes.
    # A number x is "allowed-smooth" if all its prime factors are in allowed.
    # But we need x + y to be allowed-smooth for all pairs.
    # Strategy: pick numbers that are themselves allowed-smooth, then check pairs.
    
    # Generate all allowed-smooth numbers up to MAX_VAL
    smooth_nums = []
    for x in range(1, MAX_VAL + 1):
        if x <= _MAX_SUM:
            ps = _pf(x)
            if ps.issubset(allowed):
                smooth_nums.append(x)
        else:
            # factor x
            y = x
            ok = True
            while y > 1:
                d = 2
                while d * d <= y:
                    if y % d == 0:
                        break
                    d += 1
                if d * d > y:
                    d = y
                if d not in allowed:
                    ok = False
                    break
                while y % d == 0:
                    y //= d
            if ok:
                smooth_nums.append(x)
    
    if len(smooth_nums) < N:
        return float('inf'), None
    
    # Build compatibility graph: two smooth numbers a, b are compatible if a+b is also allowed-smooth
    # For large sets, use greedy + local search
    
    # Greedy: pick N numbers pairwise compatible
    # Use a simple greedy with backtracking for small N
    
    def is_smooth_sum(a, b):
        s = a + b
        if s <= _MAX_SUM:
            return _pf(s).issubset(allowed)
        else:
            y = s
            while y > 1:
                d = 2
                while d * d <= y:
                    if y % d == 0:
                        break
                    d += 1
                if d * d > y:
                    d = y
                if d not in allowed:
                    return False
                while y % d == 0:
                    y //= d
            return True
    
    # Greedy construction: try to build a set of N pairwise-compatible numbers
    def greedy_build(pool):
        # Sort pool by "compatibility potential" (heuristic)
        # Just try random orders and greedy
        best_set = None
        best_score = float('inf')
        
        for trial in range(50):
            random.shuffle(pool)
            S = []
            for x in pool:
                if len(S) >= N:
                    break
                # Check compatibility with all in S
                if all(is_smooth_sum(x, y) for y in S):
                    S.append(x)
            if len(S) == N:
                return S
            # Track best partial
            if len(S) > (len(best_set) if best_set else 0):
                best_set = S
        
        if best_set and len(best_set) == N:
            return best_set
        return None
    
    # Try greedy
    result = greedy_build(smooth_nums)
    if result:
        cost = _score_for_prime_set(result, allowed)
        return cost, result
    
    # Fallback: local search starting from a random subset
    # Start with N random smooth numbers, then try to reduce foreign primes
    S = random.sample(smooth_nums, N)
    current_foreign = _total_foreign_count(S, allowed)
    best_foreign = current_foreign
    best_S = list(S)
    
    T = 5.0
    cooling = 0.9999
    steps = 0
    while not _deadline_hit() and T > 0.01:
        i = random.randint(0, N - 1)
        y = random.choice(smooth_nums)
        if y == S[i]:
            continue
        if y in S:
            continue
        
        # Compute delta foreign count
        old = S[i]
        delta = 0
        for j in range(N):
            if j == i:
                continue
            # old contribution
            s_old = old + S[j]
            if s_old <= _MAX_SUM:
                for p in _pf(s_old):
                    if p not in allowed:
                        delta -= 1
            else:
                x = s_old
                while x > 1:
                    d = 2
                    while d * d <= x:
                        if x % d == 0:
                            break
                        d += 1
                    if d * d > x:
                        d = x
                    if d not in allowed:
                        delta -= 1
                    while x % d == 0:
                        x //= d
            
            # new contribution
            s_new = y + S[j]
            if s_new <= _MAX_SUM:
                for p in _pf(s_new):
                    if p not in allowed:
                        delta += 1
            else:
                x = s_new
                while x > 1:
                    d = 2
                    while d * d <= x:
                        if x % d == 0:
                            break
                        d += 1
                    if d * d > x:
                        d = x
                    if d not in allowed:
                        delta += 1
                    while x % d == 0:
                        x //= d
        
        new_foreign = current_foreign + delta
        if new_foreign < current_foreign or random.random() < math.exp(-delta / T):
            S[i] = y
            current_foreign = new_foreign
            if current_foreign < best_foreign:
                best_foreign = current_foreign
                best_S = list(S)
        
        T *= cooling
        steps += 1
        if steps % 10000 == 0 and _deadline_hit():
            break
    
    cost = _score_for_prime_set(best_S, allowed)
    return cost, best_S

def _anneal_prime_set(allowed_primes, N=12, MAX_VAL=10000):
    """
    Anneal the allowed prime set itself: start with a small set, try to add/remove primes.
    """
    allowed = list(allowed_primes)
    best_cost = float('inf')
    best_S = None
    best_allowed = list(allowed)
    
    # First, try to find a good S for this allowed set
    cost, S = _try_prime_set(allowed, N, MAX_VAL)
    if S:
        best_cost = cost
        best_S = list(S)
        best_allowed = list(allowed)
    
    # Try expanding the allowed set with nearby primes
    all_primes = [p for p in range(2, 100) if _sieve[p] == p]
    
    T = 2.0
    cooling = 0.999
    for step in range(2000):
        if _deadline_hit():
            break
        if T < 0.01:
            break
        
        # Try adding a random prime
        if random.random() < 0.5 and len(allowed) < 15:
            candidate = random.choice(all_primes)
            if candidate not in allowed:
                new_allowed = allowed + [candidate]
                cost, S = _try_prime_set(new_allowed, N, MAX_VAL)
                if S and cost < best_cost:
                    best_cost = cost
                    best_S = list(S)
                    best_allowed = new_allowed
                    allowed = new_allowed
                elif S and (cost < best_cost or random.random() < math.exp(-(cost - best_cost) / T)):
                    allowed = new_allowed
                    if S:
                        best_S = list(S)
                        best_cost = min(best_cost, cost)
        else:
            # Try removing a prime (if > 3)
            if len(allowed) > 3:
                idx = random.randint(0, len(allowed) - 1)
                new_allowed = allowed[:idx] + allowed[idx+1:]
                cost, S = _try_prime_set(new_allowed, N, MAX_VAL)
                if S and cost < best_cost:
                    best_cost = cost
                    best_S = list(S)
                    best_allowed = new_allowed
                    allowed = new_allowed
                elif S and random.random() < math.exp(-(cost - best_cost) / T):
                    allowed = new_allowed
        
        T *= cooling
    
    return best_cost, best_S, best_allowed

def construct():
    N = 12
    MAX_VAL = 10000
    
    # Candidate prime sets to try
    prime_sets = [
        [2, 3, 5, 7],
        [2, 3, 5],
        [2, 3, 7],
        [2, 3, 5, 11],
        [2, 3, 5, 7, 11],
        [2, 3],
        [2, 3, 5, 7, 13],
        [3, 5, 7],
        [2, 3, 5, 11, 13],
        [2, 3, 5, 7, 11, 13],
    ]
    
    best_overall_cost = float('inf')
    best_overall_S = None
    
    for ps in prime_sets:
        if _deadline_hit():
            break
        
        # Direct try
        cost, S = _try_prime_set(ps, N, MAX_VAL)
        if S and cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_S = list(S)
        
        # If we found a perfect solution (cost 0), we can stop
        if best_overall_cost == 0:
            break
    
    # If we have time, try annealing the prime set
    if not _deadline_hit() and best_overall_cost > 0:
        for ps in prime_sets[:5]:
            if _deadline_hit():
                break
            cost, S, new_allowed = _anneal_prime_set(ps, N, MAX_VAL)
            if S and cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_S = list(S)
            if best_overall_cost == 0:
                break
    
    # Fallback: if no good solution found, use a heuristic set
    if best_overall_S is None:
        # Try the classic: multiples of 6 (smooth over {2,3})
        best_overall_S = [6 * (i + 1) for i in range(N)]
    
    return best_overall_S
