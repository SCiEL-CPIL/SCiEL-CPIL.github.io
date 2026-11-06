"""
CSP-style approach: find 12 distinct positive integers <=10000 such that
all 66 pairwise sums have prime factors only from a small set S.
Search for cliques in a compatibility graph where edge (a,b) exists iff
prime_factors(a+b) ⊆ S.

Strategy:
  1. Try small prime sets S = {2,3}, {2,3,5}, {2,3,5,7}, ...
  2. For each S, build the set of valid integers whose pairwise sums
     are S-smooth, then find a clique of size 12.
  3. Use greedy + backtracking for clique finding.
  4. Set-cover guided local search: identify which primes appear in
     the fewest sums, then try to eliminate them by adjusting elements.
  5. Fallback: local search on the objective (distinct prime count) using
     the precomputed SPF table.
"""
import time
import random
from itertools import combinations

# ---------- Precomputation ----------
_MAX_SUM = 20000

# Smallest prime factor table
_spf = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _spf[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _spf[_j] == _j:
                _spf[_j] = _i

# Prime factor set cache
_pf_cache = [None] * (_MAX_SUM + 1)
_pf_cache[1] = frozenset()
for _n in range(2, _MAX_SUM + 1):
    _factors = set()
    _x = _n
    while _x > 1:
        _p = _spf[_x]
        _factors.add(_p)
        while _x % _p == 0:
            _x //= _p
    _pf_cache[_n] = frozenset(_factors)


def _objective(arr):
    """Number of distinct primes dividing any sum of two distinct elements."""
    primes_set = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            primes_set.update(_pf_cache[arr[i] + arr[j]])
    return len(primes_set)


def _is_smooth(n, prime_set):
    """Check if all prime factors of n are in prime_set."""
    if n <= 1:
        return True
    for p in _pf_cache[n]:
        if p not in prime_set:
            return False
    return True


def _build_valid_numbers(prime_set, max_val=10000):
    """
    Build list of numbers in [1, max_val] such that for every pair (a,b) in the list,
    prime_factors(a+b) ⊆ prime_set.
    We do this by first filtering individual numbers, then using a greedy clique approach.
    """
    # First pass: numbers whose double (a+a) is smooth — necessary but not sufficient
    # Actually, we need pairwise sums. Let's build a graph.
    # For efficiency, limit to numbers where a+b for b in [1, max_val] can be checked.
    
    # Strategy: generate candidate numbers, then find a clique of size 12.
    # A number a is "compatible" with b if _is_smooth(a+b, prime_set).
    
    # To make this tractable, restrict to numbers that are multiples of the product
    # of primes in prime_set (this guarantees a+b is divisible by each prime in S,
    # but doesn't guarantee no other primes). Actually that's not right either.
    
    # Better: just enumerate all numbers 1..max_val and build adjacency.
    # max_val = 10000, so 10000 nodes. Edge check is O(1) with cache.
    # Clique of size 12 in a 10000-node graph: use greedy + backtracking.
    
    candidates = list(range(1, max_val + 1))
    return candidates


def _find_clique(prime_set, size=12, max_val=10000, time_limit=5.0):
    """
    Find a clique of given size in the compatibility graph.
    Edge (a,b) exists if _is_smooth(a+b, prime_set).
    Uses greedy coloring + backtracking (Tomita-style).
    """
    start = time.time()
    
    # Precompute for each candidate, which other candidates it's compatible with.
    # For memory, we'll use a set of neighbors for each node.
    # 10000 nodes, each with up to 9999 neighbors. That's ~50M entries worst case.
    # Too much. Instead, compute on the fly during backtracking.
    
    candidates = list(range(1, max_val + 1))
    
    # Greedy: sort by degree (number of compatible partners) descending.
    # But computing all degrees is O(n^2) = 100M checks. Too slow for 5s.
    
    # Alternative: random sampling + local search.
    # Start with a random valid set, then try to fix it.
    
    # Actually, let's use a smarter approach:
    # For small prime sets like {2,3}, the numbers must be such that all pairwise sums
    # are of the form 2^a * 3^b. These are "3-smooth" numbers (also called Hamming numbers).
    # The pairwise sums of two numbers from our set must be 3-smooth.
    
    # Let's precompute all 3-smooth numbers up to 20000.
    # 2^a * 3^b <= 20000
    smooth_numbers = set()
    for a in range(15):
        val = 1
        for b in range(12):
            val *= 3
            if val > _MAX_SUM:
                break
            smooth_numbers.add(val)
    
    # A number a is "valid" if for all b in our chosen set, a+b is smooth.
    # This is hard to precompute. Let's use a different approach.
    
    # Approach: enumerate small sets of numbers and check.
    # For {2,3}: try numbers that are multiples of 6, since then a+b is multiple of 6,
    # and we need (a+b)/6 to be 3-smooth (i.e., of form 2^a * 3^b / 6... no).
    # If a = 6k, b = 6m, then a+b = 6(k+m). We need 6(k+m) to be 2^x * 3^y.
    # 6(k+m) = 2 * 3 * (k+m). So k+m must be 2^(x-1) * 3^(y-1).
    # So k+m must be a "reduced" smooth number.
    
    # Let's just do a direct search with pruning.
    
    # Generate candidate pool: numbers where 2*number is smooth (necessary condition for a+a)
    pool = [n for n in range(1, max_val + 1) if _is_smooth(2 * n, prime_set)]
    
    if len(pool) < size:
        return None
    
    # Random restarts with greedy construction
    best_clique = None
    for restart in range(200):
        if time.time() - start > time_limit:
            break
        
        # Shuffle pool and greedily build clique
        random.shuffle(pool)
        clique = []
        for num in pool:
            if len(clique) >= size:
                break
            # Check if num is compatible with all in clique
            compatible = True
            for c in clique:
                if not _is_smooth(num + c, prime_set):
                    compatible = False
                    break
            if compatible:
                clique.append(num)
        
        if len(clique) == size:
            return clique
    
    # Backtracking search on a smaller pool
    if time.time() - start < time_limit - 1:
        # Take top 200 candidates by some heuristic
        # Heuristic: numbers with more smooth partners
        pool200 = pool[:200]
        clique = []
        
        def backtrack(start_idx, current_clique):
            if len(current_clique) == size:
                return current_clique
            if time.time() - start > time_limit:
                return None
            
            for i in range(start_idx, len(pool200)):
                num = pool200[i]
                # Check compatibility
                ok = True
                for c in current_clique:
                    if not _is_smooth(num + c, prime_set):
                        ok = False
                        break
                if ok:
                    result = backtrack(i + 1, current_clique + [num])
                    if result is not None:
                        return result
            return None
        
        result = backtrack(0, [])
        if result is not None:
            return result
    
    return None


def _local_search(arr, current_score, deadline, best_score, best_arr):
    """Local search to minimize distinct prime count."""
    improved = True
    no_improve = 0
    while improved and time.time() < deadline:
        improved = False
        no_improve += 1
        if no_improve > 20:
            break
        
        for i in range(12):
            if time.time() > deadline:
                break
            
            # Try several replacement values
            for _ in range(30):
                if time.time() > deadline:
                    break
                
                # Biased towards small numbers and multiples
                r = random.random()
                if r < 0.3:
                    new_val = random.randint(1, 100)
                elif r < 0.5:
                    new_val = 2 * random.randint(1, 200)
                elif r < 0.7:
                    new_val = 6 * random.randint(1, 200)
                else:
                    new_val = random.randint(1, 500)
                
                if new_val in arr and new_val != arr[i]:
                    continue
                
                cand = list(arr)
                cand[i] = new_val
                if len(set(cand)) != 12 or max(cand) > 10000:
                    continue
                
                score = _objective(cand)
                if score < current_score:
                    arr = cand
                    current_score = score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_arr = list(arr)
                    break
                elif score == current_score and random.random() < 0.1:
                    arr = cand
                    improved = True
    
    return best_score, best_arr


def _prime_frequency(arr):
    """Count how many pairwise sums each prime divides."""
    freq = {}
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            for p in _pf_cache[arr[i] + arr[j]]:
                freq[p] = freq.get(p, 0) + 1
    return freq


def _set_cover_local_search(arr, current_score, deadline, best_score, best_arr):
    """
    Set-cover guided local search: identify rare primes (appearing in few sums)
    and try to eliminate them by adjusting elements that contribute to those sums.
    """
    improved = True
    no_improve = 0
    while improved and time.time() < deadline:
        improved = False
        no_improve += 1
        if no_improve > 15:
            break
        
        freq = _prime_frequency(arr)
        if not freq:
            break
        
        # Sort primes by frequency (ascending) - rare primes first
        rare_primes = sorted(freq.items(), key=lambda x: x[1])
        
        # Try to eliminate the rarest primes
        for prime, count in rare_primes[:5]:
            if time.time() > deadline:
                break
            
            # Find which pairs have this prime in their sum
            affected_pairs = []
            n = len(arr)
            for i in range(n):
                for j in range(i + 1, n):
                    if prime in _pf_cache[arr[i] + arr[j]]:
                        affected_pairs.append((i, j))
            
            if not affected_pairs:
                continue
            
            # For each affected pair, try to adjust one element to remove this prime
            for i, j in affected_pairs[:10]:
                if time.time() > deadline:
                    break
                
                # Try adjusting element i
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    
                    # Generate candidate values that might avoid this prime
                    # If arr[i] + arr[j] ≡ 0 mod p, try values where sum ≢ 0 mod p
                    base = arr[j]
                    candidates = []
                    for offset in range(1, 20):
                        for sign in [1, -1]:
                            cand_val = arr[i] + sign * offset
                            if 1 <= cand_val <= 10000 and cand_val != arr[j]:
                                # Check if this avoids the prime
                                if (cand_val + base) % prime != 0:
                                    candidates.append(cand_val)
                    
                    if not candidates:
                        continue
                    
                    new_val = random.choice(candidates)
                    if new_val in arr and new_val != arr[i]:
                        continue
                    
                    cand_arr = list(arr)
                    cand_arr[i] = new_val
                    if len(set(cand_arr)) != 12 or max(cand_arr) > 10000:
                        continue
                    
                    score = _objective(cand_arr)
                    if score < current_score:
                        arr = cand_arr
                        current_score = score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_arr = list(arr)
                        break
                    elif score == current_score and random.random() < 0.05:
                        arr = cand_arr
                        improved = True
        
        # Also try general improvements
        if time.time() < deadline:
            for i in range(12):
                if time.time() > deadline:
                    break
                
                for _ in range(15):
                    if time.time() > deadline:
                        break
                    
                    r = random.random()
                    if r < 0.3:
                        new_val = random.randint(1, 100)
                    elif r < 0.5:
                        new_val = 2 * random.randint(1, 200)
                    elif r < 0.7:
                        new_val = 6 * random.randint(1, 200)
                    else:
                        new_val = random.randint(1, 500)
                    
                    if new_val in arr and new_val != arr[i]:
                        continue
                    
                    cand = list(arr)
                    cand[i] = new_val
                    if len(set(cand)) != 12 or max(cand) > 10000:
                        continue
                    
                    score = _objective(cand)
                    if score < current_score:
                        arr = cand
                        current_score = score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_arr = list(arr)
                        break
    
    return best_score, best_arr


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    best_arr = None
    best_score = float('inf')
    
    # --- Phase 1: CSP-style clique search for small prime sets ---
    prime_sets_to_try = [
        frozenset({2, 3}),
        frozenset({2, 3, 5}),
        frozenset({2, 3, 5, 7}),
        frozenset({2, 3, 5, 7, 11}),
    ]
    
    for ps in prime_sets_to_try:
        if time.time() > deadline - 30:
            break
        
        clique = _find_clique(ps, size=12, max_val=10000, time_limit=5.0)
        if clique is not None:
            score = _objective(clique)
            if score < best_score:
                best_score = score
                best_arr = list(clique)
                if best_score <= len(ps):
                    # Found optimal for this prime set
                    break
    
    # --- Phase 2: Structured candidates as fallback ---
    if time.time() < deadline - 20:
        structured = []
        
        # Multiples of 6
        for start in range(1, 100):
            arr = [6 * (start + i) for i in range(12)]
            if max(arr) <= 10000:
                structured.append(arr)
        
        # Multiples of 2
        for start in range(1, 100):
            arr = [2 * (start + i) for i in range(12)]
            if max(arr) <= 10000:
                structured.append(arr)
        
        # Random multiples of 6
        for _ in range(200):
            mults = random.sample(range(1, 1667), 12)
            arr = [6 * m for m in mults]
            if len(set(arr)) == 12 and max(arr) <= 10000:
                structured.append(arr)
        
        # Random small numbers
        for _ in range(200):
            arr = random.sample(range(1, 200), 12)
            structured.append(arr)
        
        random.shuffle(structured)
        
        for arr in structured:
            if time.time() > deadline - 20:
                break
            arr = list(dict.fromkeys(arr))
            if len(arr) != 12:
                continue
            if any(x < 1 or x > 10000 for x in arr):
                continue
            score = _objective(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
                if best_score <= 4:
                    break
    
    if best_arr is None:
        best_arr = list(range(1, 13))
        best_score = _objective(best_arr)
    
    # --- Phase 3: Set-cover guided local search refinement ---
    if time.time() < deadline - 5:
        current = list(best_arr)
        current_score = best_score
        local_deadline = min(deadline - 3, time.time() + 25)
        best_score, best_arr = _set_cover_local_search(current, current_score, local_deadline, best_score, best_arr)
    
    # --- Phase 4: Random restarts with set-cover local search ---
    if time.time() < deadline - 3:
        restarts = 0
        while time.time() < deadline - 2 and restarts < 150:
            restarts += 1
            
            r = random.random()
            if r < 0.2:
                arr = random.sample(range(1, 100), 12)
            elif r < 0.4:
                arr = [6 * m for m in random.sample(range(1, 1667), 12)]
            elif r < 0.6:
                arr = [2 * m for m in random.sample(range(1, 5000), 12)]
            elif r < 0.8:
                candidates = list(set(list(range(2, 200, 2)) + list(range(3, 200, 3))))
                arr = random.sample(candidates, 12)
            else:
                arr = random.sample(range(1, 1000), 12)
            
            if len(set(arr)) != 12 or any(x < 1 or x > 10000 for x in arr):
                continue
            
            score = _objective(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
            
            # Short set-cover local search
            if time.time() < deadline - 2:
                cur = list(arr)
                cur_score = score
                search_deadline = min(deadline - 2, time.time() + 1.5)
                best_score, best_arr = _set_cover_local_search(cur, cur_score, search_deadline, best_score, best_arr)
    
    return best_arr
