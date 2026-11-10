"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000.

Model-and-optimize approach:
  - Model: pick a small set S of primes; require that every pairwise sum a_i + a_j
    (i != j) has all prime factors inside S.
  - Optimize: for a given S, find 12 distinct integers via simulated annealing that
    minimizes the number of "foreign" prime factors appearing in pairwise sums.
  - Search: try many small prime sets S (favoring small primes / dense small primes),
    and anneal the 12 integers to zero foreign primes.
  - Enhanced: Precompute S-smooth numbers up to 10000, build compatibility graph,
    and use greedy + backtracking to find 12-cliques where all pairwise sums are S-smooth.
"""
import time
import math
import random
from itertools import combinations, product

_START_TIME = time.time()
_DEADLINE_SECONDS = 88.0  # safety margin under 90s wall-clock

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor sieve up to 20000 (max sum is 2*10000)
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[i] == i:
        for j in range(i * i, _MAX_SUM + 1, i):
            if _sieve[j] == j:
                _sieve[j] = i

def _prime_factors_fast(n):
    """Return set of prime factors of n using precomputed sieve."""
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

# Precompute prime factor sets for all sums up to 20000 (memoize)
_pf_cache = [None] * (_MAX_SUM + 1)
def _pf(s):
    r = _pf_cache[s]
    if r is None:
        r = _prime_factors_fast(s)
        _pf_cache[s] = r
    return r

N = 12
MAX_V = 10000

def _score_for_prime_set(nums, S):
    """Count distinct foreign primes (not in S) across all pairwise sums.
    Also return the set of foreign primes."""
    foreign = set()
    for i in range(N):
        for j in range(i + 1, N):
            s = nums[i] + nums[j]
            for p in _pf(s):
                if p not in S:
                    foreign.add(p)
    return len(foreign), foreign

def _is_S_smooth(n, S):
    """Check if all prime factors of n are in S."""
    for p in _pf(n):
        if p not in S:
            return False
    return True

def _find_S_smooth_clique(S, max_val=MAX_V, target_size=N):
    """Find target_size distinct numbers <= max_val such that all pairwise sums are S-smooth.
    Uses greedy construction with backtracking."""
    if _deadline_hit():
        return None
    
    # Generate all S-smooth numbers up to max_val
    smooth_nums = []
    for x in range(1, max_val + 1):
        if _is_S_smooth(x, S):
            smooth_nums.append(x)
    
    if len(smooth_nums) < target_size:
        return None
    
    # Build compatibility: two numbers are compatible if their sum is S-smooth
    # We want a clique of size target_size in this graph.
    # Use greedy: start with a random seed, then add compatible numbers.
    
    rng = random.Random(hash(frozenset(S)) % 10000)
    
    # Try multiple seeds
    for attempt in range(50):
        if _deadline_hit():
            return None
        
        # Pick a random starting number
        start_idx = rng.randint(0, len(smooth_nums) - 1)
        clique = [smooth_nums[start_idx]]
        remaining = [x for x in smooth_nums if x != smooth_nums[start_idx]]
        
        # Greedily add numbers that are compatible with all in clique
        while len(clique) < target_size and remaining:
            if _deadline_hit():
                return None
            compatible = []
            for x in remaining:
                if all(_is_S_smooth(x + y, S) for y in clique):
                    compatible.append(x)
            if not compatible:
                break
            # Pick one (prefer smaller to leave room, or random)
            if rng.random() < 0.5:
                chosen = compatible[0]
            else:
                chosen = rng.choice(compatible)
            clique.append(chosen)
            remaining.remove(chosen)
        
        if len(clique) == target_size:
            return sorted(clique)
    
    return None

def _find_S_smooth_clique_enhanced(S, max_val=MAX_V, target_size=N):
    """Enhanced clique finder using a more systematic approach."""
    if _deadline_hit():
        return None
    
    # Generate all S-smooth numbers up to max_val
    smooth_nums = []
    for x in range(1, max_val + 1):
        if _is_S_smooth(x, S):
            smooth_nums.append(x)
    
    if len(smooth_nums) < target_size:
        return None
    
    rng = random.Random(hash(frozenset(S)) % 10000)
    
    # Try all pairs as seeds, then extend
    n_smooth = len(smooth_nums)
    tried_pairs = set()
    
    for _ in range(200):
        if _deadline_hit():
            return None
        
        # Pick two random numbers as seed
        i = rng.randint(0, n_smooth - 1)
        j = rng.randint(0, n_smooth - 1)
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        if pair in tried_pairs:
            continue
        tried_pairs.add(pair)
        
        a, b = smooth_nums[i], smooth_nums[j]
        if not _is_S_smooth(a + b, S):
            continue
        
        clique = [a, b]
        remaining = [x for x in smooth_nums if x not in (a, b)]
        
        # Greedily add
        while len(clique) < target_size and remaining:
            if _deadline_hit():
                return None
            compatible = []
            for x in remaining:
                if all(_is_S_smooth(x + y, S) for y in clique):
                    compatible.append(x)
            if not compatible:
                break
            chosen = rng.choice(compatible)
            clique.append(chosen)
            remaining.remove(chosen)
        
        if len(clique) == target_size:
            return sorted(clique)
    
    return None

def _total_foreign_count(nums, S):
    """Total count of foreign prime appearances (with multiplicity across pairs)."""
    cnt = 0
    for i in range(N):
        for j in range(i + 1, N):
            s = nums[i] + nums[j]
            for p in _pf(s):
                if p not in S:
                    cnt += 1
    return cnt

def _anneal(S, start, iters, T0, T1, rng, smooth_nums=None):
    """Simulated annealing to minimize foreign prime count for prime set S.
    Returns best (score, nums) found.
    If smooth_nums is provided, bias moves toward S-smooth values."""
    best = start[:]
    best_score, _ = _score_for_prime_set(best, S)
    cur = start[:]
    cur_score, _ = _score_for_prime_set(cur, S)
    
    def _eval_move(idx, new_val):
        old_val = cur[idx]
        old_cnt = 0
        for j in range(N):
            if j == idx:
                continue
            for p in _pf(old_val + cur[j]):
                if p not in S:
                    old_cnt += 1
        new_cnt = 0
        for j in range(N):
            if j == idx:
                continue
            for p in _pf(new_val + cur[j]):
                if p not in S:
                    new_cnt += 1
        return new_cnt - old_cnt
    
    cur_cnt = _total_foreign_count(cur, S)
    best_cnt = cur_cnt
    best = cur[:]
    
    for it in range(iters):
        if _deadline_hit():
            break
        T = T0 * (T1 / T0) ** (it / max(1, iters))
        idx = rng.randint(0, N - 1)
        # Propose new value: bias toward S-smooth numbers if available
        if smooth_nums and rng.random() < 0.4:
            # Pick a random S-smooth number
            new_val = rng.choice(smooth_nums)
        elif rng.random() < 0.7:
            delta = rng.randint(-200, 200)
            new_val = cur[idx] + delta
        else:
            new_val = rng.randint(1, MAX_V)
        new_val = max(1, min(MAX_V, new_val))
        if new_val == cur[idx]:
            continue
        if any(cur[j] == new_val for j in range(N) if j != idx):
            continue
        delta_cnt = _eval_move(idx, new_val)
        if delta_cnt < 0 or rng.random() < math.exp(-delta_cnt / max(T, 1e-9)):
            cur[idx] = new_val
            cur_cnt += delta_cnt
            if cur_cnt < best_cnt:
                best_cnt = cur_cnt
                best = cur[:]
                if best_cnt == 0:
                    break
    true_score, _ = _score_for_prime_set(best, S)
    return true_score, best

def construct():
    rng = random.Random(12345)
    best_score = 10 ** 9
    best_nums = list(range(1, 13))
    
    # Strategy A: hill-climb on a fixed "all small primes" set to get a good baseline
    # then extract the actual prime set used.
    # We first do a quick anneal with a large S (all primes up to 20000) to minimize
    # total foreign (none), i.e., just minimize distinct primes globally via count of
    # distinct primes in sums.
    
    # Actually, simpler: do global anneal minimizing number of DISTINCT primes in sums
    # (the true objective) using a count-based proxy won't work directly.
    # Instead, use the model: pick S as the set of primes, and we want 0 foreign.
    # We'll try many S.
    
    # Build candidate prime sets:
    # 1) First k smallest primes
    # 2) Random small primes
    # 3) Primes from a good baseline solution
    
    # First, get a baseline by annealing with S = all primes (minimizes nothing useful)
    # Instead, hill-climb directly on the objective: number of distinct primes in sums.
    def _objective(nums):
        primes = set()
        for i in range(N):
            for j in range(i + 1, N):
                primes |= _pf(nums[i] + nums[j])
        return len(primes)
    
    # Direct simulated annealing on the true objective (distinct prime count)
    # using a count-based delta for speed: count of distinct primes is hard to delta,
    # so use total prime-appearance count as proxy, then verify.
    
    def _total_appearances(nums):
        cnt = 0
        for i in range(N):
            for j in range(i + 1, N):
                cnt += len(_pf(nums[i] + nums[j]))
        return cnt
    
    def _anneal_objective(start, iters, T0, T1):
        cur = start[:]
        best = start[:]
        best_obj = _objective(best)
        cur_cnt = _total_appearances(cur)
        best_cnt = cur_cnt
        for it in range(iters):
            if _deadline_hit():
                break
            T = T0 * (T1 / T0) ** (it / max(1, iters))
            idx = rng.randint(0, N - 1)
            if rng.random() < 0.7:
                delta = rng.randint(-300, 300)
                new_val = cur[idx] + delta
            else:
                new_val = rng.randint(1, MAX_V)
            new_val = max(1, min(MAX_V, new_val))
            if new_val == cur[idx]:
                continue
            if any(cur[j] == new_val for j in range(N) if j != idx):
                continue
            # delta in total appearances
            old_val = cur[idx]
            old_cnt = 0
            for j in range(N):
                if j == idx:
                    continue
                old_cnt += len(_pf(old_val + cur[j]))
            new_cnt = 0
            for j in range(N):
                if j == idx:
                    continue
                new_cnt += len(_pf(new_val + cur[j]))
            delta_cnt = new_cnt - old_cnt
            if delta_cnt < 0 or rng.random() < math.exp(-delta_cnt / max(T, 1e-9)):
                cur[idx] = new_val
                cur_cnt += delta_cnt
                if cur_cnt < best_cnt:
                    best_cnt = cur_cnt
                    obj = _objective(cur)
                    if obj < best_obj:
                        best_obj = obj
                        best = cur[:]
        return best_obj, best
    
    # Initial candidates
    inits = [
        list(range(1, 13)),
        list(range(2, 25, 2)),
        list(range(6, 78, 6)),
        list(range(12, 156, 12)),
        [2 ** i for i in range(1, 13)],
    ]
    for _ in range(8):
        inits.append(sorted(rng.sample(range(1, MAX_V + 1), N)))
    
    # Phase 1: direct objective annealing from several starts
    for start in inits:
        if _deadline_hit():
            break
        if len(set(start)) != N:
            continue
        if not all(1 <= x <= MAX_V for x in start):
            continue
        obj, nums = _anneal_objective(start, 30000, 50.0, 0.01)
        if obj < best_score:
            best_score = obj
            best_nums = nums[:]
    
    # Phase 2: model-and-optimize with extracted prime sets
    def _extract_S(nums):
        primes = set()
        for i in range(N):
            for j in range(i + 1, N):
                primes |= _pf(nums[i] + nums[j])
        return primes
    
    S_best = _extract_S(best_nums)
    
    # Candidate S sets to try
    S_candidates = [S_best]
    small_primes = []
    m = 2
    while len(small_primes) < 30:
        if _sieve[m] == m:
            small_primes.append(m)
        m += 1
    for k in [3, 4, 5, 6, 7, 8, 10]:
        if k <= len(small_primes):
            S_candidates.append(set(small_primes[:k]))
    extra = set(small_primes) - S_best
    S_candidates.append(S_best | set(list(extra)[:3]))
    S_candidates.append(S_best | set(list(extra)[:6]))
    if len(S_best) > 5:
        S_candidates.append(set(sorted(S_best, reverse=True)[:max(5, len(S_best) - 2)]))
    
    # Phase 2a: Try S-smooth clique finding for small S
    for S in S_candidates:
        if _deadline_hit():
            break
        if len(S) > 10:
            continue  # Only try clique finding for small S
        
        # Generate S-smooth numbers
        smooth_nums = [x for x in range(1, MAX_V + 1) if _is_S_smooth(x, S)]
        
        # Try direct clique finding
        clique = _find_S_smooth_clique(S, MAX_V, N)
        if clique:
            obj = _objective(clique)
            if obj < best_score:
                best_score = obj
                best_nums = clique[:]
        
        # Also try enhanced clique finder
        if not _deadline_hit():
            clique2 = _find_S_smooth_clique_enhanced(S, MAX_V, N)
            if clique2:
                obj = _objective(clique2)
                if obj < best_score:
                    best_score = obj
                    best_nums = clique[:]
        
        # Anneal with smooth number bias
        if not _deadline_hit():
            for start in [best_nums[:], list(range(1, 13))]:
                if len(set(start)) != N:
                    continue
                if not all(1 <= x <= MAX_V for x in start):
                    continue
                score, nums = _anneal(S, start, 30000, 40.0, 0.005, rng, smooth_nums)
                if score == 0:
                    obj = _objective(nums)
                    if obj < best_score:
                        best_score = obj
                        best_nums = nums[:]
                    S2 = _extract_S(nums)
                    if S2 != S and len(S2) <= 10:
                        S_candidates.append(S2)
                else:
                    obj = _objective(nums)
                    if obj < best_score:
                        best_score = obj
                        best_nums = nums[:]
    
    # Phase 3: final local refinement on best_nums
    improved = True
    while improved and not _deadline_hit():
        improved = False
        cur = best_nums[:]
        cur_score = best_score
        for idx in range(N):
            if _deadline_hit():
                break
            old_val = cur[idx]
            tries = set()
            for d in range(-200, 201, 7):
                v = old_val + d
                if 1 <= v <= MAX_V and v != old_val:
                    tries.add(v)
            for _ in range(30):
                tries.add(rng.randint(1, MAX_V))
            for nv in tries:
                if nv in cur:
                    continue
                trial = cur[:]
                trial[idx] = nv
                if len(set(trial)) != N:
                    continue
                obj = _objective(trial)
                if obj < cur_score:
                    cur = trial
                    cur_score = obj
                    improved = True
                    break
            if improved:
                break
        best_nums = cur
        best_score = cur_score
    
    # Phase 4: Try to reduce prime set by dropping primes and re-optimizing
    if not _deadline_hit():
        S_final = _extract_S(best_nums)
        # Try subsets of S_final
        for drop in range(1, min(3, len(S_final))):
            for subset in combinations(sorted(S_final), len(S_final) - drop):
                S_sub = set(subset)
                if len(S_sub) < 3:
                    continue
                smooth_nums = [x for x in range(1, MAX_V + 1) if _is_S_smooth(x, S_sub)]
                clique = _find_S_smooth_clique(S_sub, MAX_V, N)
                if clique:
                    obj = _objective(clique)
                    if obj < best_score:
                        best_score = obj
                        best_nums = clique[:]
    
    result = best_nums
    assert len(set(result)) == N
    assert all(1 <= x <= MAX_V for x in result)
    return result

# EVOLVE-BLOCK-END
