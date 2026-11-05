import time
import math
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

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

def _count_primes(nums):
    primes = set()
    for a, b in combinations(nums, 2):
        primes |= _factorize(a + b)
    return len(primes)

def _max_clique_size(adj, n):
    """Branch and bound maximum clique using bitmasks (n up to ~30)."""
    best = 0
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda v: -adj[v].bit_count())
    remap = {order[i]: i for i in range(n)}
    # Build remapped adjacency
    new_adj = [0] * n
    for i in range(n):
        v = order[i]
        for j in range(n):
            if adj[v] & (1 << j):
                new_adj[i] |= (1 << remap[j])
    
    def expand(candidates, current_size):
        nonlocal best
        if current_size + candidates.bit_count() <= best:
            return
        if current_size > best:
            best = current_size
        if candidates == 0:
            return
        while candidates:
            # Pick vertex with max degree in candidates
            v = -1
            max_deg = -1
            c = candidates
            while c:
                lb = c & (-c)
                idx = lb.bit_length() - 1
                deg = (new_adj[idx] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                c ^= lb
            candidates &= ~(1 << v)
            # Try including v
            if current_size + 1 + (new_adj[v] & candidates).bit_count() > best:
                expand(new_adj[v] & candidates, current_size + 1)
            if current_size + candidates.bit_count() <= best:
                return
    
    expand((1 << n) - 1, 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    target_size = 12
    max_val = 10000
    best_result = list(range(1, 13))
    best_count = _count_primes(best_result)
    
    # Strategy: try increasing sets of primes, for each find 12 numbers
    # whose pairwise sums use only those primes
    # Primes to consider in increasing order of frequency
    candidate_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for k in range(2, len(candidate_primes) + 1):
        if time.time() > deadline - 5:
            break
        # Try all subsets of size k? Too many. Instead try specific combinations.
        # For k=2,3: try all combinations
        # For larger k: use greedy/heuristic
        
        if k <= 4:
            prime_combos = list(combinations(candidate_primes[:15], k))
        else:
            # For larger k, focus on the smallest primes
            prime_combos = [tuple(candidate_primes[:k])]
            # Also try some variations
            for i in range(k, min(k+3, len(candidate_primes))):
                combo = tuple(candidate_primes[:i] + candidate_primes[i+1:i+2]) if i+2 <= len(candidate_primes) else None
                if combo:
                    prime_combos.append(combo)
        
        for pcombo in prime_combos:
            if time.time() > deadline - 5:
                break
            if len(pcombo) >= best_count:
                continue
            
            S = set(pcombo)
            P = 1
            for p in pcombo:
                P *= p
            
            # Find numbers <= max_val such that when paired with other chosen numbers,
            # the sum is S-smooth (all prime factors in S)
            # Precompute valid numbers: a number x is "S-smooth-sum-compatible" if
            # there exist at least 11 other numbers y such that x+y is S-smooth
            # Simpler: find all pairs (a,b) with a<b, a+b S-smooth, then find max clique of size 12
            
            # Build graph: nodes are numbers 1..max_val, edge if a+b is S-smooth
            # This is too large. Instead, restrict to numbers where x mod P has properties.
            
            # Two numbers a,b have a+b divisible by all primes in S iff:
            # For each p in S: a+b ≡ 0 (mod p)
            # This means b ≡ -a (mod p) for all p in S, i.e., b ≡ -a (mod P)
            # So a+b is divisible by P. But a+b could also have other factors...
            # We need a+b to be S-SMOOTH, meaning all prime factors are in S.
            
            # Let's precompute S-smooth numbers up to 2*max_val
            smooth_set = set()
            for s in range(2, 2 * max_val + 1):
                t = s
                for p in S:
                    while t % p == 0:
                        t //= p
                if t == 1:
                    smooth_set.add(s)
            
            # Now find numbers that can form many pairs
            # For each x in 1..max_val, count how many y in 1..max_val, y!=x, have x+y in smooth_set
            # Then pick top candidates and find clique
            
            # More efficient: for each smooth sum s, the pairs are (1,s-1), (2,s-2), ..., (s/2, s/2)
            # Filter to 1<=a<b<=max_val
            # Build adjacency list for candidate numbers
            
            # Get candidate numbers: those appearing in at least 11 valid pairs
            degree = {}
            for s in smooth_set:
                # Pairs (a, s-a) where 1<=a<s-a<=max_val, i.e., a < s/2 and s-a <= max_val and a >= 1
                a_min = max(1, s - max_val)
                a_max = min(s - 1, max_val)
                # Need a < s - a, so a < s/2
                a_max = min(a_max, (s - 1) // 2)
                for a in range(a_min, a_max + 1):
                    b = s - a
                    if b > max_val or b <= a:
                        continue
                    degree[a] = degree.get(a, 0) + 1
                    degree[b] = degree.get(b, 0) + 1
            
            # Pick top candidates by degree
            candidates = sorted(degree.items(), key=lambda x: -x[1])
            # Need at least 12 candidates with high degree
            top = [x for x, d in candidates if d >= 11][:50]
            
            if len(top) < 12:
                continue
            
            # Build graph on top candidates
            n_top = len(top)
            idx_map = {x: i for i, x in enumerate(top)}
            adj = [0] * n_top
            for i in range(n_top):
                a = top[i]
                for j in range(i + 1, n_top):
                    b = top[j]
                    if a + b in smooth_set:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            if n_top > 60:
                # Too large for bit-based clique, try greedy
                # Greedy: repeatedly pick vertex with max degree in remaining
                chosen = []
                remaining = set(range(n_top))
                while len(chosen) < 12 and remaining:
                    best_v = max(remaining, key=lambda v: (adj[v] & (sum((1<<u for u in remaining), 0))).bit_count())
                    deg = (adj[best_v] & sum((1<<u for u in remaining if u != best_v), 0)).bit_count()
                    if deg < 12 - len(chosen) - 1:
                        break
                    chosen.append(best_v)
                    remaining.discard(best_v)
                if len(chosen) >= 12:
                    subset = [top[i] for i in chosen[:12]]
                    cnt = _count_primes(subset)
                    if cnt < best_count:
                        best_count = cnt
                        best_result = subset
            else:
                clique_size = _max_clique_size(adj, n_top)
                if clique_size >= 12:
                    # Found a clique, extract it
                    # Re-run to extract the actual clique
                    def extract_clique(adj, n, target_size):
                        order = sorted(range(n), key=lambda v: -adj[v].bit_count())
                        remap = {order[i]: i for i in range(n)}
                        new_adj = [0] * n
                        for i in range(n):
                            v = order[i]
                            for j in range(n):
                                if adj[v] & (1 << j):
                                    new_adj[i] |= (1 << remap[j])
                        
                        best_clique = set()
                        def expand(candidates, current):
                            nonlocal best_clique
                            if len(current) >= target_size:
                                if len(best_clique) < target_size or len(current) > len(best_clique):
                                    best_clique = set(current)
                                return
                            if len(current) + candidates.bit_count() < target_size:
                                return
                            while candidates:
                                v = -1
                                max_deg = -1
                                c = candidates
                                while c:
                                    lb = c & (-c)
                                    idx = lb.bit_length() - 1
                                    deg = (new_adj[idx] & candidates).bit_count()
                                    if deg > max_deg:
                                        max_deg = deg
                                        v = idx
                                    c ^= lb
                                candidates &= ~(1 << v)
                                expand(new_adj[v] & candidates, current | (1 << v))
                        
                        expand((1 << n) - 1, 0)
                        return [order[i] for i in range(n) if best_clique & (1 << i)]
                    
                    clique = extract_clique(adj, n_top, 12)
                    if len(clique) >= 12:
                        subset = [top[i] for i in clique[:12]]
                        cnt = _count_primes(subset)
                        if cnt < best_count:
                            best_count = cnt
                            best_result = subset
    
    return best_result
