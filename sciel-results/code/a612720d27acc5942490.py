"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _spf_sieve(limit):
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _pf_set(n, spf):
    s = set()
    while n > 1:
        p = spf[n]
        s.add(p)
        while n % p == 0:
            n //= p
    return s

def _score_full(arr, spf):
    """Count distinct prime factors of all pairwise sums of arr."""
    ps = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s = ai + arr[j]
            if s > 1:
                ps |= _pf_set(s, spf)
    return len(ps)

def construct():
    t0 = time.time()
    deadline = t0 + 85.0

    MAX_VAL = 10000
    MAX_SUM = 20000
    spf = _spf_sieve(MAX_SUM)

    # Best fallback
    best_set = list(range(1, 13))
    best_score = _score_full(best_set, spf)

    def try_update(arr):
        nonlocal best_set, best_score
        s = _score_full(arr, spf)
        if s < best_score:
            best_score = s
            best_set = arr[:]
        return s

    # === Phase 1: S-smooth clique search ===
    # For a fixed prime set S, find 12 distinct values in [1,10000]
    # such that every pairwise sum is S-smooth.
    # This means the sum's prime factors are all in S.

    # Precompute: for each prime set S, which values v in [1,10000]
    # can participate? v must be such that there exist 11 other values
    # where all pairwise sums are S-smooth.
    # 
    # Key insight: if all values are multiples of some base, or if we
    # restrict to S-smooth numbers themselves...
    # Actually, the constraint is on SUMS, not on the values themselves.
    #
    # Strategy: For prime set S = {p1, p2, ..., pk}, precompute the set
    # of all S-smooth numbers <= 20000. Then two values a, b are
    # "compatible" if a+b is S-smooth. We need a clique of size 12.

    # Try various prime sets, starting with the smallest
    prime_sets_to_try = [
        {2, 3},
        {2, 5},
        {3, 5},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
        {2, 3, 11},
        {2, 3, 13},
        {2, 5, 11},
        {2, 5, 13},
        {2, 7, 11},
        {2, 3, 5, 7},
        {2, 3, 5, 11},
        {2, 3, 5, 13},
        {2, 3, 7, 11},
        {2, 5, 7, 11},
        {3, 5, 7, 11},
        {2, 3, 5, 7, 11},
    ]

    def is_smooth(n, prime_set):
        """Check if n is smooth over prime_set."""
        if n <= 1:
            return True
        for p in sorted(prime_set):
            while n % p == 0:
                n //= p
        return n == 1

    def generate_smooth_numbers(limit, prime_set):
        """Generate all prime_set-smooth numbers <= limit."""
        result = set()
        primes = sorted(prime_set)
        
        def gen(idx, current):
            if current > limit:
                return
            if current > 1:
                result.add(current)
            if idx >= len(primes):
                return
            val = current
            while val <= limit:
                result.add(val)
                gen(idx + 1, val)
                if val > limit // primes[idx]:
                    break
                val *= primes[idx]
        
        gen(0, 1)
        return result

    # For each prime set, find a clique of 12 in the compatibility graph
    for prime_set in prime_sets_to_try:
        if time.time() > deadline - 5:
            break
        
        # Quick check: how many prime_set-smooth numbers <= 20000 exist?
        smooth_sums = generate_smooth_numbers(MAX_SUM, prime_set)
        if len(smooth_sums) < 66:  # Need at least 66 pairwise sums to be smooth
            continue
        
        # Now, which values a in [1, 10000] are "viable"?
        # a is viable if there exist at least 11 other values b such that
        # a+b is in smooth_sums. But that's expensive to check directly.
        #
        # Alternative: Build the compatibility graph on a subset of candidate values.
        # Candidate values: all a in [1, 10000] such that a + 1 is smooth OR
        # more practically, just try a structured set.
        #
        # Better approach: The values themselves don't need to be smooth,
        # only their pairwise sums. So let's think about what values work.
        #
        # If all values are congruent to some residue r mod m, and m divides
        # all pairwise sums... no, that's not quite right.
        #
        # Let's just build the graph on a reasonable candidate set.
        # Candidate values: all numbers in [1, 10000] where the number
        # of "compatible partners" is high.
        
        # Precompute for each a in [1, 10000]: how many b in [1, 10000]
        # have a+b smooth? This is O(10000 * 10000) which is too slow.
        #
        # Instead: for each a, the compatible b's are those where a+b is
        # in smooth_sums, i.e., b in {s - a : s in smooth_sums} ∩ [1, 10000].
        
        # Let's be smarter. For a given a, compatible b's are:
        # b such that a+b ∈ smooth_sums, 1 ≤ b ≤ 10000, b ≠ a
        # i.e., b ∈ {s - a : s ∈ smooth_sums, 1 ≤ s-a ≤ 10000, s-a ≠ a}
        
        # Precompute smooth_sums as a sorted list for faster lookup
        smooth_sums_list = sorted(smooth_sums)
        
        # For each candidate a, compute degree in compatibility graph
        # We'll sample candidate a's rather than checking all 10000
        
        # Actually, let's think about it differently.
        # If we pick all values from a specific arithmetic structure...
        
        # Try: values that are themselves smooth (over the same prime set)
        # Then a+b where a,b are smooth may or may not be smooth.
        # Not guaranteed.
        
        # Alternative structured approach:
        # If all values are of the form c * s where s is smooth,
        # then a+b = c*(s1+s2). For this to be smooth, c*(s1+s2) must be smooth.
        # If c is smooth, we need s1+s2 to be smooth.
        
        # Let's just do a direct clique search on a filtered candidate set.
        
        # Filter: keep a in [1, 10000] if at least 11 other values b
        # exist with a+b smooth. Approximate by checking if there are
        # at least 11 smooth numbers in [a+1, a+10000].
        
        # For speed, let's use a different tactic:
        # Generate candidate values as: for each smooth number s <= 20000,
        # and each split s = a + b with 1 <= a,b <= 10000, add a and b.
        # This gives us values that have at least one compatible partner.
        
        candidates = set()
        for s in smooth_sums_list:
            if s < 2:
                continue
            # s = a + b, 1 <= a <= b-1, a+b=s, a,b <= 10000
            lo = max(1, s - 10000)
            hi = min(10000, s - 1)
            if lo < hi:
                # Add some samples from this range
                # For efficiency, add endpoints and a few middle values
                candidates.add(lo)
                candidates.add(hi - 1)
                if hi - lo > 20:
                    candidates.add((lo + hi - 1) // 2)
        
        # Also add small values and smooth numbers
        for v in range(1, 50):
            candidates.add(v)
        
        candidates = sorted(candidates)
        if len(candidates) < 12:
            continue
        
        # Build adjacency: for each pair of candidates, check if sum is smooth
        # Use a set for O(1) lookup
        smooth_set = set(smooth_sums_list)
        
        n_cand = len(candidates)
        if n_cand > 500:
            # Too many, subsample
            # Keep candidates with highest "smoothness potential"
            # Approximate: candidates that are themselves smooth
            cand_smooth = [c for c in candidates if is_smooth(c, prime_set)]
            if len(cand_smooth) >= 12:
                candidates = cand_smooth[:500]
            else:
                candidates = candidates[:500]
            n_cand = len(candidates)
        
        # Build adjacency list
        adj = [set() for _ in range(n_cand)]
        for i in range(n_cand):
            ai = candidates[i]
            for j in range(i + 1, n_cand):
                aj = candidates[j]
                if ai + aj in smooth_set:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Find clique of size 12 using backtracking
        # Order candidates by degree (descending) for better pruning
        deg_order = sorted(range(n_cand), key=lambda i: -len(adj[i]))
        
        found_clique = None
        
        def find_clique(candidates_list, target_size):
            """Find a clique of target_size in the subgraph induced by candidates_list."""
            if len(candidates_list) < target_size:
                return None
            if target_size == 1:
                return [candidates_list[0]]
            if target_size == 0:
                return []
            
            # Try each vertex as the first in the clique
            for idx in range(len(candidates_list)):
                v = candidates_list[idx]
                # Remaining candidates must be neighbors of v
                remaining = [u for u in candidates_list[idx+1:] if u in adj[v]]
                if len(remaining) < target_size - 1:
                    continue
                sub = find_clique(remaining, target_size - 1)
                if sub is not None:
                    return [v] + sub
            return None
        
        # This recursive approach may be too slow for large candidate sets.
        # Use a greedy + local search approach instead.
        
        # Greedy: pick vertex with max degree, then restrict to neighbors, repeat
        clique = []
        available = list(range(n_cand))
        for _ in range(12):
            if not available:
                break
            # Pick vertex with max degree within available
            best_v = max(available, key=lambda v: len(adj[v] & set(available)))
            clique.append(best_v)
            # Restrict to common neighbors
            common = set(adj[best_v])
            available = [v for v in available if v != best_v and v in common]
        
        if len(clique) == 12:
            found_clique = [candidates[v] for v in clique]
        
        # If greedy didn't find a full clique, try backtracking on a smaller set
        if found_clique is None and n_cand <= 200:
            # Use Bron-Kerbosch variant for max clique, but we just need size 12
            # Try a simpler backtracking with the degree-ordered vertices
            def bt_clique(picked, candidates_remaining):
                if len(picked) == 12:
                    return picked[:]
                if not candidates_remaining:
                    return None
                for idx in range(len(candidates_remaining)):
                    v = candidates_remaining[idx]
                    # Check v is adjacent to all in picked
                    ok = all(v in adj[p] for p in picked)
                    if not ok:
                        continue
                    new_picked = picked + [v]
                    new_remaining = [u for u in candidates_remaining[idx+1:] 
                                     if all(u in adj[p] for p in new_picked)]
                    if len(new_remaining) < 12 - len(new_picked):
                        continue
                    result = bt_clique(new_picked, new_remaining)
                    if result is not None:
                        return result
                return None
            
            found_clique = bt_clique([], deg_order)
        
        if found_clique is not None:
            # Verify
            if len(set(found_clique)) == 12 and all(1 <= v <= 10000 for v in found_clique):
                arr = sorted(found_clique)
                try_update(arr)
                if best_score <= len(prime_set):
                    # Can't do better than the number of primes in S
                    return best_set
        
        # If we found a clique with score = len(prime_set), that's optimal for this S
        # Check if we can do even better with a smaller prime set (already tried)
    
    # === Phase 2: Local search to improve ===
    if time.time() < deadline - 5:
        random.seed(42)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                old = best_set[i]
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    new_val = random.randint(1, 10000)
                    if new_val == old:
                        continue
                    if new_val in best_set:
                        continue
                    test = best_set[:]
                    test[i] = new_val
                    test.sort()
                    ns = _score_full(test, spf)
                    if ns < best_score:
                        best_score = ns
                        best_set = test
                        improved = True
                        break
    
    # === Phase 3: Neighborhood polish ===
    if time.time() < deadline - 3:
        for i in range(12):
            if time.time() > deadline - 3:
                break
            old = best_set[i]
            for delta in range(-50, 51):
                if delta == 0:
                    continue
                new_val = old + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in best_set:
                    continue
                test = best_set[:]
                test[i] = new_val
                test.sort()
                ns = _score_full(test, spf)
                if ns < best_score:
                    best_score = ns
                    best_set = test
    
    return best_set
# EVOLVE-BLOCK-END
