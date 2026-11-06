"""Optimized program: search for 12 distinct positive integers <=10000 minimizing distinct primes in pair sums."""
import time
import random
from itertools import combinations

# Precompute smallest-prime-factor table up to 20000 (max sum = 10000+9999=19999)
_MAX_SUM = 20000
_spf = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM**0.5) + 1):
    if _spf[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _spf[_j] == _j:
                _spf[_j] = _i

# Precompute prime factors for each number up to _MAX_SUM
_pf_table = [None] * (_MAX_SUM + 1)
for _n in range(2, _MAX_SUM + 1):
    factors = set()
    temp = _n
    while temp > 1:
        p = _spf[temp]
        factors.add(p)
        while temp % p == 0:
            temp //= p
    _pf_table[_n] = factors

def _objective(arr):
    """Count distinct primes dividing any sum of two distinct elements."""
    primes_set = set()
    n = len(arr)
    table = _pf_table
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            primes_set.update(table[ai + arr[j]])
    return len(primes_set)

def _find_clique_in_compatibility_graph(allowed_primes, deadline, max_candidates=500):
    """
    Find 12 distinct positive integers <=10000 such that all pairwise sums
    have prime factors only from allowed_primes.
    Uses a compatibility graph: vertex = integer, edge if sum has no prime outside allowed_primes.
    Greedy + backtracking to find a clique of size 12.
    """
    if time.time() > deadline:
        return None
    
    # Generate candidate integers that are "compatible" with themselves
    # (i.e., 2*x has prime factors only in allowed_primes)
    candidates = []
    for x in range(1, 10001):
        s = 2 * x
        if s > _MAX_SUM:
            break
        factors = _pf_table[s]
        if factors.issubset(allowed_primes):
            candidates.append(x)
    
    if len(candidates) < 12:
        return None
    
    # Limit candidates to speed up search
    if len(candidates) > max_candidates:
        # Prefer smaller numbers (more likely to have sums within allowed primes)
        random.shuffle(candidates)
        candidates = candidates[:max_candidates]
    
    n_cand = len(candidates)
    
    # Build compatibility: compatible[i][j] = True if candidates[i]+candidates[j] has allowed primes only
    # Use a set of sets for adjacency
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        ci = candidates[i]
        for j in range(i + 1, n_cand):
            cj = candidates[j]
            s = ci + cj
            if s <= _MAX_SUM:
                factors = _pf_table[s]
                if factors.issubset(allowed_primes):
                    adj[i].add(j)
                    adj[j].add(i)
    
    # Greedy clique finding with backtracking
    # Start with the candidate with highest degree
    degrees = [len(adj[i]) for i in range(n_cand)]
    
    # Sort candidates by degree (descending) to guide search
    order = sorted(range(n_cand), key=lambda i: -degrees[i])
    
    def backtrack(clique, candidates_remaining):
        if time.time() > deadline:
            return None
        if len(clique) == 12:
            return clique[:]
        
        # Try each candidate in candidates_remaining
        for idx in candidates_remaining:
            # Check if idx is compatible with all in clique
            compatible = True
            for c in clique:
                if idx not in adj[c]:
                    compatible = False
                    break
            if not compatible:
                continue
            
            new_clique = clique + [idx]
            # Remaining candidates must be compatible with idx
            new_remaining = [c for c in candidates_remaining if c in adj[idx] and c not in new_clique]
            
            result = backtrack(new_clique, new_remaining)
            if result is not None:
                return result
        
        return None
    
    # Initial candidates for the first element: all high-degree candidates
    initial_remaining = order[:]
    result = backtrack([], initial_remaining)
    if result is not None:
        return [candidates[i] for i in result]
    
    return None

def _local_search(current, current_score, deadline, best_score, best_arr):
    """Lightweight local search. Returns (best_score, best_arr)."""
    no_improve_count = 0
    while time.time() < deadline:
        no_improve_count += 1
        if no_improve_count > 20:
            break
        
        improved = False
        # Try a few random single-element replacements
        for _ in range(5):
            if time.time() > deadline:
                break
            i = random.randint(0, 11)
            # Bias towards small numbers
            if random.random() < 0.8:
                new_val = random.randint(1, 200)
            else:
                new_val = random.randint(1, 10000)
            
            if new_val in current:
                continue
                
            cand = list(current)
            cand[i] = new_val
            if len(set(cand)) != 12 or max(cand) > 10000:
                continue
                
            score = _objective(cand)
            if score < current_score:
                current = cand
                current_score = score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_arr = list(current)
                break
            elif score == current_score and random.random() < 0.1:
                current = cand
                improved = True
                break
        
        if not improved:
            # Try a swap
            if random.random() < 0.5:
                i, j = random.sample(range(12), 2)
                new_vi = random.randint(1, 200)
                new_vj = random.randint(1, 200)
                if new_vi != new_vj and new_vi not in current and new_vj not in current:
                    cand = list(current)
                    cand[i] = new_vi
                    cand[j] = new_vj
                    if len(set(cand)) == 12 and max(cand) <= 10000:
                        score = _objective(cand)
                        if score <= current_score:
                            current = cand
                            current_score = score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_arr = list(current)

        if improved:
            no_improve_count = 0
        else:
            # Perturb
            if random.random() < 0.2:
                i = random.randint(0, 11)
                cand = list(current)
                cand[i] = random.randint(1, 500)
                if len(set(cand)) == 12 and max(cand) <= 10000:
                    current = cand
                    current_score = _objective(current)

    return best_score, best_arr

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: must finish within 60s wall-clock
    
    best_arr = None
    best_score = float('inf')
    
    # --- Phase 1: Constraint Satisfaction with Small Prime Sets ---
    # Try increasing sets of allowed primes
    prime_sets_to_try = [
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {3, 5},
        {3, 5, 7},
    ]
    
    for primes in prime_sets_to_try:
        if time.time() > deadline - 10:
            break
        
        result = _find_clique_in_compatibility_graph(primes, deadline - 10, max_candidates=300)
        if result is not None:
            score = _objective(result)
            if score < best_score:
                best_score = score
                best_arr = list(result)
                # If we found a solution with few primes, keep searching for better
                if best_score <= len(primes):
                    # Found a very good solution, but try to improve with local search
                    break
    
    # --- Phase 2: Fallback structured candidates if CSP didn't find good solution ---
    if best_arr is None or best_score > 6:
        # Generate structured candidates
        structured = []
        
        # Arithmetic progressions
        for d in range(1, 50):
            for a in range(1, 100):
                arr = [a + i * d for i in range(12)]
                if max(arr) <= 10000 and len(set(arr)) == 12:
                    structured.append(arr)
        
        # Multiples of small bases
        for base in [2, 3, 6, 10, 30]:
            max_mult = 10000 // base
            if max_mult < 12:
                continue
            for start in range(1, 30):
                if start + 11 > max_mult:
                    break
                arr = [base * (start + i) for i in range(12)]
                if len(set(arr)) == 12:
                    structured.append(arr)
        
        # Small random integers
        for _ in range(200):
            arr = random.sample(range(1, 100), 12)
            structured.append(arr)
        
        for _ in range(100):
            arr = random.sample(range(2, 200, 2), 12)
            structured.append(arr)
        
        random.shuffle(structured)
        
        for arr in structured:
            if time.time() > deadline - 10:
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
    
    # --- Phase 3: Local Search from Best ---
    if time.time() < deadline - 5:
        current = list(best_arr)
        current_score = best_score
        local_deadline = min(deadline - 3, start_time + 25)
        best_score, best_arr = _local_search(current, current_score, local_deadline, best_score, best_arr)
    
    # --- Phase 4: Random Restarts with Local Search ---
    if time.time() < deadline - 3:
        restarts_done = 0
        while time.time() < deadline - 2 and restarts_done < 30:
            restarts_done += 1
            
            r = random.random()
            if r < 0.3:
                arr = random.sample(range(1, 100), 12)
            elif r < 0.6:
                arr = random.sample(range(2, 300, 2), 12)
            else:
                base = random.choice([2, 3, 6, 10])
                max_m = 10000 // base
                mults = random.sample(range(1, max_m + 1), 12)
                arr = [base * m for m in mults]
            
            if len(set(arr)) != 12 or any(x < 1 or x > 10000 for x in arr):
                continue
            
            score = _objective(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
            
            if time.time() < deadline - 2:
                cur = list(arr)
                cur_score = score
                search_deadline = min(deadline - 2, time.time() + 0.5)
                best_score, best_arr = _local_search(cur, cur_score, search_deadline, best_score, best_arr)

    return best_arr
