import time
import math
import random

_START = time.time()
_HARD_DEADLINE = _START + 80.0

def _init_spf(limit):
    spf = list(range(limit + 1))
    if limit >= 1:
        spf[0] = 0
    if limit >= 2:
        for i in range(2, int(limit**0.5) + 1):
            if spf[i] == i:
                for j in range(i*i, limit + 1, i):
                    if spf[j] == j:
                        spf[j] = i
    return spf

SPF_LIMIT = 20000
_spf = _init_spf(SPF_LIMIT)

def _prime_factors(num):
    if num <= 1:
        return frozenset()
    out = set()
    x = num
    while x > 1:
        p = _spf[x]
        out.add(p)
        while x % p == 0:
            x //= p
    return frozenset(out)

def _is_smooth(num, allowed_primes):
    if num <= 1:
        return True
    for p in allowed_primes:
        while num % p == 0:
            num //= p
    return num == 1

def construct():
    deadline = time.time() + 75.0
    n = 12
    max_val = 10000
    
    best_overall = list(range(1, 13))
    best_score_overall = 999
    
    def calc_score(lst):
        union = set()
        for i in range(n):
            for j in range(i+1, n):
                union |= _prime_factors(lst[i] + lst[j])
        return len(union)
    
    cur_score = calc_score(best_overall)
    if cur_score < best_score_overall:
        best_score_overall = cur_score

    # Generate 3-smooth numbers up to max_val
    smooth_3 = []
    p = 1
    while p <= max_val:
        q = p
        while q <= max_val:
            r = q
            while r <= max_val:
                smooth_3.append(r)
                r *= 3
            q *= 2
        p *= 2
    smooth_3 = sorted(set(smooth_3))
    
    # Generate 5-smooth numbers up to max_val (primes 2, 3, 5)
    smooth_5 = []
    p = 1
    while p <= max_val:
        q = p
        while q <= max_val:
            r = q
            while r <= max_val:
                s = r
                while s <= max_val:
                    smooth_5.append(s)
                    s *= 5
                r *= 3
            q *= 2
        p *= 3
    smooth_5 = sorted(set(smooth_5))

    def find_clique(max_k, n_target, allowed_primes, pool, time_limit):
        # Filter pool to those <= max_k
        candidates = [k for k in pool if k <= max_k]
        if len(candidates) < n_target:
            return None
        
        # Precompute compatibility?
        # candidates might be up to a few hundred.
        # O(N^2) precomputation is fast.
        N = len(candidates)
        adj = [set() for _ in range(N)]
        for i in range(N):
            ci = candidates[i]
            for j in range(i+1, N):
                cj = candidates[j]
                if _is_smooth(ci + cj, allowed_primes):
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Find max clique size n_target
        # Use a simple branch and bound / backtracking
        # Order candidates by degree (descending) to improve pruning
        degs = [(len(adj[i]), i) for i in range(N)]
        degs.sort(reverse=True)
        sorted_indices = [idx for _, idx in degs]
        
        # Map old index to new index for easier adjacency lookup?
        # Or just use the list of indices.
        # Let's stick to indices into candidates.
        
        found = None
        
        def backtrack(current_clique, available):
            nonlocal found
            if found is not None:
                return
            if len(current_clique) == n_target:
                found = current_clique[:]
                return
            
            # Pruning: if len(current_clique) + len(available) < n_target:
            #    return
            if len(current_clique) + len(available) < n_target:
                return
                
            for i in range(len(available)):
                # Try adding available[i]
                v = available[i]
                # Check if v is connected to all in current_clique
                ok = True
                for u in current_clique:
                    if v not in adj[u]:
                        ok = False
                        break
                if not ok:
                    continue
                
                current_clique.append(v)
                # Update available: remove v and anything not connected to v
                new_available = [x for x in available[i+1:] if x in adj[v]]
                backtrack(current_clique, new_available)
                current_clique.pop()
                if found is not None:
                    return

        # Initial available: all candidates sorted by degree
        # Actually, standard maximum clique algorithms work on the graph directly.
        # Let's just call backtrack with all nodes.
        # But we need to be careful about order.
        # Let's try a few random shuffles of the sorted order if the first fails?
        # Or just run it once with the degree sorted order.
        
        backtrack([], sorted_indices)
        
        if found:
            return [candidates[v] for v in found]
            
        return None

    # Strategy 1: 2-prime solution {2, 3}
    # Base must be 3-smooth.
    # k_i must be 3-smooth and k_i + k_j must be 3-smooth.
    # Note: k_i themselves don't need to be 3-smooth for the SUM to be 3-smooth?
    # Wait. If a_i = B * k_i, then a_i + a_j = B * (k_i + k_j).
    # For a_i + a_j to have only primes in P_B U P_sum, 
    # if B is P_B-smooth, then k_i + k_j must be P_sum-smooth.
    # The values of k_i themselves don't restrict the primes of k_i+k_j directly, 
    # but k_i + k_j being 3-smooth is a very strong constraint on k_i.
    # Actually, if k_i + k_j is 3-smooth, does k_i have to be? No.
    # Example: 1 + 5 = 6 (3-smooth). 5 is not 3-smooth.
    # So we should NOT restrict candidates to 3-smooth k_i.
    # We should restrict k_i to any integer in [1, max_k].
    # But the search space [1, max_k] is huge (up to 10000).
    # However, if k_i + k_j is 3-smooth, then k_i and k_j are "close" to 3-smooth numbers?
    # Or rather, the graph of integers where edges connect sums that are 3-smooth.
    # Let's check the density.
    # Number of 3-smooth numbers up to 20000 is small.
    # Pairs (x, y) such that x+y is 3-smooth.
    # If max_k is small, we can search all integers.
    # If max_k is large, we need a smarter way.
    
    # Let's try to find a clique in the graph of integers [1, max_k]
    # Edge if sum is 3-smooth.
    
    def find_int_clique(max_k, n_target, allowed_primes, time_limit):
        candidates = list(range(1, max_k + 1))
        if len(candidates) < n_target:
            return None
        
        N = len(candidates)
        # This might be too big for O(N^2) precomputation if max_k is large.
        # 10000^2 = 100M operations. Too slow in Python.
        # We need to restrict candidates.
        # Which candidates are likely to form a clique?
        # If k_i + k_j is 3-smooth, then k_i is likely "small" or "structured".
        # Actually, let's look at the structure of 3-smooth sums.
        # 3-smooth numbers: 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, ...
        # If we pick k_i from a set of 3-smooth numbers, is k_i+k_j likely 3-smooth?
        # 2+2=4 (yes), 2+3=5 (no).
        # So even 3-smooth k_i don't guarantee 3-smooth sums.
        
        # Let's try a heuristic:
        # Sample a subset of candidates.
        # Candidates that are 3-smooth, and their neighbors?
        
        # Alternative:
        # Try specific small values of max_k (large B).
        # If B is large, max_k is small.
        # If B=1000, max_k=10. Trivial.
        # If B=100, max_k=100.
        # Let's iterate B in descending order of size (ascending max_k).
        
        return None # Placeholder, logic moved to loop below

    # Iterate bases B
    # We want to try bases that allow finding a clique.
    # Small max_k is easier to search exhaustively or with simple heuristics.
    # Let's try B from 10000 down to 1.
    # But B must be 3-smooth for the 2-prime case.
    
    bases_3_smooth = [b for b in smooth_3 if b <= max_val]
    # Sort by B descending (max_k ascending)
    bases_3_smooth.sort(reverse=True)
    
    found_2_prime = False
    
    for B in bases_3_smooth:
        if time.time() > deadline - 10.0:
            break
            
        max_k = max_val // B
        if max_k < n:
            continue
            
        # Search for clique of size n in [1, max_k]
        # Edge if x+y is 3-smooth.
        
        candidates = list(range(1, max_k + 1))
        N = len(candidates)
        
        if N > 500:
            # Too many to do full pairwise check quickly?
            # 500^2 = 250k checks. Fast.
            # 1000^2 = 1M checks. Fast.
            # Let's try.
            pass
        
        adj = [set() for _ in range(N)]
        # Map value to index
        # value v is at index v-1
        
        for i in range(N):
            vi = candidates[i] + 1 # 1-indexed value
            for j in range(i+1, N):
                vj = candidates[j] + 1
                s = vi + vj
                if _is_smooth(s, [2, 3]):
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Find clique
        found = None
        
        def backtrack_clique(current, available):
            nonlocal found
            if found is not None:
                return
            if len(current) == n:
                found = current[:]
                return
            if len(current) + len(available) < n:
                return
            
            # Pruning heuristic: sort available by degree in available?
            # For now, simple order.
            for i in range(len(available)):
                v = available[i]
                ok = True
                for u in current:
                    if v not in adj[u]:
                        ok = False
                        break
                if not ok:
                    continue
                
                current.append(v)
                new_available = [x for x in available[i+1:] if x in adj[v]]
                backtrack_clique(current, new_available)
                current.pop()
                if found is not None:
                    return

        # Initial available: all nodes
        # To help backtracking, order by degree (descending)
        degs = [(len(adj[i]), i) for i in range(N)]
        degs.sort(reverse=True)
        init_available = [idx for _, idx in degs]
        
        backtrack_clique([], init_available)
        
        if found:
            result = [B * (idx + 1) for idx in found]
            # Verify
            if len(set(result)) == n and all(x <= max_val for x in result):
                score = calc_score(result)
                if score < best_score_overall:
                    best_score_overall = score
                    best_overall = result
                if score <= 2:
                    return result
                found_2_prime = True
                break # Found a 2-prime solution, no need to try other B for 2-prime
    
    if not found_2_prime:
        # Try 3-prime solution {2, 3, 5}
        # B must be 5-smooth.
        # x+y must be 5-smooth.
        
        bases_5_smooth = [b for b in smooth_5 if b <= max_val]
        bases_5_smooth.sort(reverse=True)
        
        found_3_prime = False
        
        for B in bases_5_smooth:
            if time.time() > deadline - 5.0:
                break
                
            max_k = max_val // B
            if max_k < n:
                continue
                
            candidates = list(range(1, max_k + 1))
            N = len(candidates)
            
            adj = [set() for _ in range(N)]
            for i in range(N):
                vi = candidates[i] + 1
                for j in range(i+1, N):
                    vj = candidates[j] + 1
                    s = vi + vj
                    if _is_smooth(s, [2, 3, 5]):
                        adj[i].add(j)
                        adj[j].add(i)
                        
            found = None
            
            def backtrack_clique_3(current, available):
                nonlocal found
                if found is not None:
                    return
                if len(current) == n:
                    found = current[:]
                    return
                if len(current) + len(available) < n:
                    return
                
                for i in range(len(available)):
                    v = available[i]
                    ok = True
                    for u in current:
                        if v not in adj[u]:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    current.append(v)
                    new_available = [x for x in available[i+1:] if x in adj[v]]
                    backtrack_clique_3(current, new_available)
                    current.pop()
                    if found is not None:
                        return

            degs = [(len(adj[i]), i) for i in range(N)]
            degs.sort(reverse=True)
            init_available = [idx for _, idx in degs]
            
            backtrack_clique_3([], init_available)
            
            if found:
                result = [B * (idx + 1) for idx in found]
                if len(set(result)) == n and all(x <= max_val for x in result):
                    score = calc_score(result)
                    if score < best_score_overall:
                        best_score_overall = score
                        best_overall = result
                    if score <= 3:
                        return result
                    found_3_prime = True
                    break
    
    # If we didn't find a clean 2 or 3 prime solution via this method,
    # we rely on the initial best_overall (which might be improved by local search if we add it)
    
    # Optional: Local search to improve best_overall
    if time.time() < deadline - 10.0:
        cur = list(best_overall)
        cur_score = calc_score(cur)
        
        # Pool: 5-smooth numbers
        pool = [x for x in smooth_5 if x <= max_val]
        random.shuffle(pool)
        
        improved = True
        iters = 0
        while improved and iters < 500 and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            # Pick a random index to replace
            idx = random.randint(0, n-1)
            old_val = cur[idx]
            
            # Try a few replacements
            for _ in range(20):
                new_val = random.choice(pool)
                if new_val == old_val or new_val in cur:
                    continue
                cur[idx] = new_val
                new_score = calc_score(cur)
                if new_score < cur_score:
                    cur_score = new_score
                    improved = True
                    if cur_score < best_score_overall:
                        best_score_overall = cur_score
                        best_overall = cur[:]
                    break
                cur[idx] = old_val
                if improved:
                    break
            
    return best_overall
