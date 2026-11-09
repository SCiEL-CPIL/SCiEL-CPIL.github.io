"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import random
from functools import lru_cache

def _is_smooth(n, primes):
    """Check if n is composed only of the given primes."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def _get_prime_factors(n):
    """Return the set of prime factors of n."""
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

def _precompute_smooth_sums(max_val, primes):
    """
    Precompute a boolean array where smooth_sums[s] is True if s is smooth
    with respect to the given primes. s ranges from 2 to 2*max_val.
    """
    smooth_sums = [False] * (2 * max_val + 1)
    for s in range(2, 2 * max_val + 1):
        smooth_sums[s] = _is_smooth(s, primes)
    return smooth_sums

def _find_clique_backtrack(adj, n_nodes, target_size, max_time=30):
    """
    Find a clique of target_size using backtracking with forward checking.
    adj is a list of sets (adjacency lists).
    Returns a list of node indices or None.
    """
    import time
    start_time = time.time()
    
    # Order by degree descending
    degrees = [len(adj[i]) for i in range(n_nodes)]
    order = sorted(range(n_nodes), key=lambda x: degrees[x], reverse=True)
    
    # Remap to use the ordered indices for better pruning
    # We'll work directly with the original indices but iterate in order
    
    best_clique = []
    
    def backtrack(clique, available, depth):
        nonlocal best_clique
        if time.time() - start_time > max_time:
            return False
        if len(clique) == target_size:
            best_clique = clique[:]
            return True
        if not available:
            return False
        # If not enough candidates left
        if len(available) < target_size - len(clique):
            return False
        
        for v in available:
            # Check if v is connected to all in clique
            if all(v in adj[u] for u in clique):
                clique.append(v)
                # Forward checking: intersect available with neighbors of v
                new_available = set()
                for c in available:
                    if c != v and c in adj[v]:
                        new_available.add(c)
                if backtrack(clique, new_available, depth + 1):
                    return True
                clique.pop()
        return False
    
    initial_available = set(order)
    if backtrack([], initial_available, 0):
        return best_clique
    return None

def construct():
    """
    Find 12 distinct positive integers <= 10000 such that the union of
    prime factors of all pairwise sums (distinct indices) is minimized.
    
    Strategy: Try to find a set where all pairwise sums are k-smooth for
    small k. Start with k=4 (primes 2,3,5,7), then try k=3, k=2, etc.
    For each k, build a compatibility graph and find a clique of size 12.
    """
    max_val = 10000
    n_needed = 12
    
    # Try different prime sets, starting with smaller ones (fewer primes = better)
    # But we need enough candidates to form a clique of size 12
    
    prime_sets_to_try = [
        [2, 3],           # 3-smooth
        [2, 3, 5],        # 5-smooth
        [2, 3, 5, 7],     # 7-smooth
        [2, 3, 5, 7, 11], # 11-smooth
        [2, 3, 5, 7, 11, 13],  # 13-smooth
    ]
    
    best_result = None
    best_prime_count = float('inf')
    
    for primes in prime_sets_to_try:
        if len(primes) >= best_prime_count:
            break  # No point trying sets with more primes
        
        # Precompute smooth sums
        smooth_sums = _precompute_smooth_sums(max_val, primes)
        
        # Get candidate numbers: those that are themselves smooth
        candidates = []
        for i in range(1, max_val + 1):
            if _is_smooth(i, primes):
                candidates.append(i)
        
        if len(candidates) < n_needed:
            continue
        
        # Build compatibility graph
        n_cand = len(candidates)
        adj = [set() for _ in range(n_cand)]
        
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                s = candidates[i] + candidates[j]
                if s <= 2 * max_val and smooth_sums[s]:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Find clique
        clique = _find_clique_backtrack(adj, n_cand, n_needed, max_time=15)
        
        if clique is not None:
            result = [candidates[i] for i in clique]
            # Verify and count distinct primes
            all_primes = set()
            for a, b in itertools.combinations(result, 2):
                all_primes |= _get_prime_factors(a + b)
            if len(all_primes) < best_prime_count:
                best_prime_count = len(all_primes)
                best_result = result
            # If we found a solution with the minimum possible primes for this set, we can stop
            if len(all_primes) <= len(primes):
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: use a heuristic approach
    # Try to find numbers where pairwise sums have few prime factors
    # Use a greedy approach: start with a random number, then add compatible ones
    
    # Generate candidates that are smooth with respect to {2,3,5,7,11,13,17}
    fallback_primes = [2, 3, 5, 7, 11, 13, 17]
    candidates = []
    for i in range(1, max_val + 1):
        if _is_smooth(i, fallback_primes):
            candidates.append(i)
    
    if len(candidates) < n_needed:
        return list(range(1, 13))
    
    # Greedy: pick numbers that maximize compatibility
    random.seed(42)
    selected = []
    remaining = list(range(len(candidates)))
    random.shuffle(remaining)
    
    # Start with a random candidate
    first = remaining[0]
    selected.append(first)
    remaining.remove(first)
    
    while len(selected) < n_needed and remaining:
        # For each remaining candidate, count how many of the selected it's compatible with
        best_idx = -1
        best_score = -1
        for idx in remaining:
            score = 0
            for sel_idx in selected:
                s = candidates[idx] + candidates[sel_idx]
                if _is_smooth(s, fallback_primes):
                    score += 1
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)
    
    if len(selected) == n_needed:
        result = [candidates[i] for i in selected]
        all_primes = set()
        for a, b in itertools.combinations(result, 2):
            all_primes |= _get_prime_factors(a + b)
        return result
    
    # Ultimate fallback
    return list(range(1, 13))

# EVOLVE-BLOCK-END
