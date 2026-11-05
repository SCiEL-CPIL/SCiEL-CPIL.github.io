"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

N = 5000
_DEADLINE = 300.0  # seconds wall-clock limit

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_list):
    n = len(points)
    if n == 0:
        return True
    for p in primes_list:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _greedy_optimized(deadline, seed=None, prime_order="min_class"):
    """
    Optimized greedy vertex cover: maintain residue counts per prime,
    track violated primes, remove least-populated residue class of selected prime.
    """
    if seed is not None:
        random.seed(seed)
    
    all_primes = _primes(N + 1)
    
    remaining = bytearray(N + 1)
    for x in range(N + 1):
        remaining[x] = 1
    
    prime_counts = {}
    violated = set()
    
    for p in all_primes:
        if p > N + 1:
            break
        counts = [0] * p
        for x in range(N + 1):
            counts[x % p] += 1
        prime_counts[p] = counts
        if min(counts) > 0:
            violated.add(p)
    
    removed_count = 0
    
    while violated:
        if time.time() > deadline:
            break
        
        n_remaining = N + 1 - removed_count
        violated = {p for p in violated if p <= n_remaining}
        if not violated:
            break
        
        if prime_order == "min_class":
            best_p = None
            best_min = float('inf')
            for p in violated:
                m = min(prime_counts[p])
                if m < best_min:
                    best_min = m
                    best_p = p
            p = best_p
        elif prime_order == "increasing":
            p = min(violated)
        elif prime_order == "decreasing":
            p = max(violated)
        else:
            p = random.choice(list(violated))
        
        counts = prime_counts[p]
        min_val = min(counts)
        min_residues = [r for r in range(p) if counts[r] == min_val]
        r = random.choice(min_residues)
        
        for x in range(r, N + 1, p):
            if remaining[x]:
                remaining[x] = 0
                removed_count += 1
                for p2 in all_primes:
                    if p2 > N + 1:
                        break
                    prime_counts[p2][x % p2] -= 1
        
        for p2 in list(violated):
            if min(prime_counts[p2]) == 0:
                violated.discard(p2)
    
    return [x for x in range(N + 1) if remaining[x]]


def _try_add_back(points, primes_list, deadline):
    points_set = set(points)
    n = len(points)
    candidates = [x for x in range(N + 1) if x not in points_set]
    random.shuffle(candidates)
    max_try = 200
    for x in candidates[:max_try]:
        if time.time() > deadline:
            break
        new_n = n + 1
        ok = True
        for p in primes_list:
            if p > new_n:
                break
            residues = set(v % p for v in points)
            residues.add(x % p)
            if len(residues) == p:
                ok = False
                break
        if ok:
            points_set.add(x)
            n += 1
    return list(points_set)


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - 15
    all_primes = _primes(N + 1)
    best_result = []
    best_len = 0
    
    orders = ["min_class", "increasing", "decreasing", "random"]
    num_restarts = 3
    
    for order in orders:
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
            seed = (hash((order, restart)) & 0x7FFFFFFF)
            result = _greedy_optimized(deadline - 20, seed=seed, prime_order=order)
            if len(result) > best_len:
                best_len = len(result)
                best_result = result
            if time.time() < deadline - 10 and len(result) >= best_len - 50:
                improved = _try_add_back(result, all_primes, deadline - 10)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_result = improved
    
    if best_result:
        n = len(best_result)
        ok = True
        for p in all_primes:
            if p > n:
                break
            residues = set(v % p for v in best_result)
            if len(residues) == p:
                ok = False
                break
        if ok:
            return best_result
    
    # Fallback
    points = list(range(N + 1))
    while True:
        if time.time() > deadline:
            break
        n = len(points)
        if n == 0:
            break
        relevant = [p for p in all_primes if p <= n]
        found = False
        for p in relevant:
            classes = defaultdict(list)
            for v in points:
                classes[v % p].append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                to_remove = set(smallest)
                points = [v for v in points if v not in to_remove]
                found = True
                break
        if not found:
            break
    return points
# EVOLVE-BLOCK-END
