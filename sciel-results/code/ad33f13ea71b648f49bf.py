"""
Structural rewrite: simulated-annealing / local-search over residue assignments,
with fast prefix-score evaluation via vectorized coverage counts.

Key structural change from the seed:
  - Instead of forward-greedy + backward-refine (which gets stuck in a local
    optimum defined by the processing order), we treat the residue vector as a
    168-dim discrete search space and perform iterated local search with
    simulated annealing. Each move changes one or a few residues; the objective
    is the exact doubly-covered prefix length L, evaluated in O(P * L/p) ~
    O(P log L) using NumPy vectorized range updates.
  - We also add a "pair-swap" neighborhood (swap residues of two primes) which
    is a different move type from single-coordinate ascent.
  - Random restarts with perturbation from the best solution found.
"""
import time
import numpy as np
import random

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _prefix_L(residues, primes, scan_limit=12000):
    """
    Compute the largest L such that every integer 1..L is covered at least twice
    by the congruences x ≡ r_p (mod p).
    Returns (L, counts_array) where counts_array has length scan_limit+1.
    """
    n = len(primes)
    counts = np.zeros(scan_limit + 1, dtype=np.int32)
    for i in range(n):
        r = residues[i]
        p = primes[i]
        if r == 0:
            r_eff = p
        else:
            r_eff = r
        if r_eff <= scan_limit:
            counts[r_eff::p] += 1
    # Find first position where count < 2
    deficient = np.where(counts[1:] < 2)[0]
    if len(deficient) == 0:
        L = scan_limit
    else:
        L = int(deficient[0])  # positions are 1-indexed, so deficient[0] is the 0-based index into counts[1:]
    return L, counts

def _score_full(residues, primes):
    """Exact score capped at 10^5."""
    # First do a quick scan to find approximate L
    L, _ = _prefix_L(residues, primes, scan_limit=12000)
    if L < 12000:
        return L
    # Extend scan
    L, _ = _prefix_L(residues, primes, scan_limit=50000)
    if L < 50000:
        return L
    L, _ = _prefix_L(residues, primes, scan_limit=100000)
    return min(L, 100000)

def _quick_score(residues, primes, scan_limit=8000):
    """Fast approximate score for inner-loop optimization."""
    L, _ = _prefix_L(residues, primes, scan_limit=scan_limit)
    return L

def _random_residues(primes):
    n = len(primes)
    return [random.randint(0, p - 1) for p in primes]

def _perturb(residues, primes, strength=0.3):
    """Create a perturbed copy of residues by changing a fraction of coordinates."""
    n = len(primes)
    new_res = list(residues)
    num_changes = max(1, int(strength * n))
    indices = random.sample(range(n), num_changes)
    for idx in indices:
        new_res[idx] = random.randint(0, primes[idx] - 1)
    return new_res

def _greedy_seed(primes, scan_limit=5000):
    """
    A smarter greedy: process primes in order of increasing p (already sorted),
    but for each prime pick the residue that maximizes the prefix-L in a
    small local window. We use a simple heuristic: try all residues for small p,
    and a subset for large p.
    """
    n = len(primes)
    residues = [0] * n
    
    # Start with a baseline: set r_p = 1 for all (covers 1, p+1, 2p+1, ...)
    # Actually, let's use a smarter init: r_p = p-1 (covers p-1, 2p-1, ...)
    # or r_p = 0 (covers p, 2p, 3p, ...)
    # Let's try multiple inits and pick the best
    
    best_res = None
    best_L = -1
    for init_mode in range(4):
        res = [0] * n
        for i, p in enumerate(primes):
            if init_mode == 0:
                res[i] = 0
            elif init_mode == 1:
                res[i] = 1
            elif init_mode == 2:
                res[i] = p - 1
            else:
                res[i] = random.randint(0, p - 1)
        L, _ = _prefix_L(res, primes, scan_limit)
        if L > best_L:
            best_L = L
            best_res = res[:]
    
    # Now do a simple forward pass: for each prime, try all residues (if p is small)
    # or a random sample (if p is large), and keep the one that gives best L.
    counts_cache = None  # We'll recompute each time for simplicity
    
    for idx in range(n):
        p = primes[idx]
        best_r = best_res[idx]
        best_L_local = -1
        
        # Number of candidates to try
        if p <= 50:
            candidates = list(range(p))
        elif p <= 200:
            candidates = list(range(0, p, max(1, p // 20)))
            if best_r not in candidates:
                candidates.append(best_r)
        else:
            candidates = [best_r] + [random.randint(0, p - 1) for _ in range(15)]
        
        for r in candidates:
            test_res = best_res[:]
            test_res[idx] = r
            L, _ = _prefix_L(test_res, primes, scan_limit)
            if L > best_L_local:
                best_L_local = L
                best_r = r
        
        best_res[idx] = best_r
    
    return best_res

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    random.seed(42)
    np.random.seed(42)
    
    primes = _get_primes(1000)
    n = len(primes)
    
    # Phase 1: Get a good initial solution via greedy
    best_res = _greedy_seed(primes, scan_limit=6000)
    best_L, _ = _prefix_L(best_res, primes, 12000)
    
    # Phase 2: Iterated local search with simulated annealing
    scan_limit_inner = 8000
    temperature = 500.0
    cooling = 0.995
    min_temp = 0.5
    
    stagnation_count = 0
    max_stagnation = 500
    
    # Main loop
    while time.time() < deadline - 10.0:
        # Random move: change one or two residues
        if random.random() < 0.7:
            # Single-coordinate move
            idx = random.randint(0, n - 1)
            p = primes[idx]
            old_r = best_res[idx]
            if p <= 100:
                new_r = random.randint(0, p - 1)
            else:
                # For large p, try a few options: old_r, 0, 1, p-1, random
                options = [old_r, 0, 1, p - 1, random.randint(0, p - 1)]
                new_r = random.choice(options)
            
            if new_r == old_r:
                continue
            
            best_res[idx] = new_r
            new_L, _ = _prefix_L(best_res, primes, scan_limit_inner)
            
            if new_L >= best_L:
                best_L = new_L
                stagnation_count = 0
            else:
                # Accept with probability exp(-(best_L - new_L) / temperature)
                delta = best_L - new_L
                if delta > 0 and delta < temperature:
                    prob = np.exp(-delta / temperature)
                    if random.random() < prob:
                        best_L = new_L
                    else:
                        best_res[idx] = old_r
                else:
                    best_res[idx] = old_r
                stagnation_count += 1
        else:
            # Pair-swap move: swap residues of two random primes
            i1 = random.randint(0, n - 1)
            i2 = random.randint(0, n - 1)
            if i1 == i2:
                continue
            p1, p2 = primes[i1], primes[i2]
            r1, r2 = best_res[i1], best_res[i2]
            
            # Swap (if residues are compatible, i.e., r1 < p2 and r2 < p1)
            if r1 < p2 and r2 < p1:
                best_res[i1] = r1 % p2  # Wait, this isn't a swap, this is assignment
                # Actually let's do a proper swap: assign r1 to prime i2 and r2 to prime i1
                # But r1 might be >= p2, so we need r1 % p2
                new_r1 = r1 % p2  # residue for prime i2
                new_r2 = r2 % p1  # residue for prime i1
                
                best_res[i1] = new_r2
                best_res[i2] = new_r1
                new_L, _ = _prefix_L(best_res, primes, scan_limit_inner)
                
                if new_L >= best_L:
                    best_L = new_L
                    stagnation_count = 0
                else:
                    delta = best_L - new_L
                    if delta > 0 and delta < temperature:
                        prob = np.exp(-delta / temperature)
                        if random.random() < prob:
                            best_L = new_L
                        else:
                            best_res[i1] = r1
                            best_res[i2] = r2
                    else:
                        best_res[i1] = r1
                        best_res[i2] = r2
                    stagnation_count += 1
        
        # Cool down
        temperature *= cooling
        if temperature < min_temp:
            temperature = min_temp
        
        # Stagnation handling: if stuck, do a random restart from perturbed best
        if stagnation_count > max_stagnation:
            perturbed = _perturb(best_res, primes, strength=0.2)
            new_L, _ = _prefix_L(perturbed, primes, scan_limit_inner)
            if new_L > best_L:
                best_res = perturbed
                best_L = new_L
            temperature = max(temperature, 200.0)
            stagnation_count = 0
    
    # Phase 3: Final polish - do a few passes of single-coordinate ascent with full scan
    for _ in range(3):
        if time.time() > deadline - 5.0:
            break
        improved = True
        while improved:
            improved = False
            for idx in range(n):
                if time.time() > deadline - 3.0:
                    break
                p = primes[idx]
                old_r = best_res[idx]
                best_r_local = old_r
                best_L_local = best_L
                
                # Try all residues if p is small, otherwise a sample
                if p <= 80:
                    candidates = list(range(p))
                else:
                    candidates = [old_r, 0, 1, p - 1] + [random.randint(0, p - 1) for _ in range(10)]
                
                for r in candidates:
                    if r == old_r:
                        continue
                    best_res[idx] = r
                    L, _ = _prefix_L(best_res, primes, 12000)
                    if L > best_L_local:
                        best_L_local = L
                        best_r_local = r
                    elif L == best_L_local and r < best_r_local:
                        best_r_local = r
                best_res[idx] = best_r_local
                if best_r_local != old_r:
                    best_L = best_L_local
                    improved = True
    
    # Final exact score
    final_score = _score_full(best_res, primes)
    
    return best_res
