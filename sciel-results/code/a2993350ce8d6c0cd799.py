# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

def _build_counts(residues, primes, cap):
    """Build count array for 1..cap."""
    counts = np.zeros(cap + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        if start <= cap:
            counts[start::p] += 1
    return counts

def _find_L(counts, cap):
    """Find largest L such that all 1..L have count >= 2."""
    below = np.where(counts[1:cap+1] < 2)[0]
    if len(below) == 0:
        return cap
    return int(below[0])

def _score(residues, primes, cap=100000):
    counts = _build_counts(residues, primes, cap)
    return _find_L(counts, cap)

def _greedy_first_bad_point(residues, primes, cap, max_iter=100):
    """Greedy repair: fix first bad point iteratively."""
    n = len(primes)
    for _ in range(max_iter):
        counts = _build_counts(residues, primes, cap)
        L = _find_L(counts, cap)
        if L >= cap:
            break
        # Find first bad point
        below = np.where(counts[1:cap+1] < 2)[0]
        if len(below) == 0:
            break
        bad_x = int(below[0]) + 1
        # Try to fix by changing one residue
        fixed = False
        # Prefer larger primes first (less disruption)
        order = list(range(n))
        random.shuffle(order)
        # Sort by prime size descending for less disruption
        order.sort(key=lambda i: -primes[i])
        for idx in order:
            old_r = residues[idx]
            new_r = bad_x % primes[idx]
            if old_r == new_r:
                continue
            residues[idx] = new_r
            new_counts = _build_counts(residues, primes, cap)
            new_L = _find_L(new_counts, cap)
            if new_L >= L:
                fixed = True
                break
            else:
                residues[idx] = old_r
        if not fixed:
            # Try smaller primes
            order2 = list(range(n))
            random.shuffle(order2)
            order2.sort(key=lambda i: primes[i])
            for idx in order2:
                old_r = residues[idx]
                new_r = bad_x % primes[idx]
                if old_r == new_r:
                    continue
                residues[idx] = new_r
                new_counts = _build_counts(residues, primes, cap)
                new_L = _find_L(new_counts, cap)
                if new_L >= L:
                    fixed = True
                    break
                else:
                    residues[idx] = old_r
        if not fixed:
            break
    return residues

def _simulated_annealing(residues, primes, cap, deadline, iters=2000):
    """Simulated annealing on residues."""
    n = len(primes)
    best = residues[:]
    best_sc = _score(best, primes, cap)
    current = residues[:]
    current_sc = best_sc
    T = 50.0
    cooling = 0.999
    
    for it in range(iters):
        if time.time() > deadline:
            break
        # Random perturbation
        idx = random.randint(0, n - 1)
        old_r = current[idx]
        current[idx] = random.randint(0, primes[idx] - 1)
        
        new_sc = _score(current, primes, cap)
        delta = new_sc - current_sc
        
        if delta >= 0 or random.random() < np.exp(delta / max(T, 0.1)):
            current_sc = new_sc
            if new_sc > best_sc:
                best_sc = new_sc
                best = current[:]
        else:
            current[idx] = old_r
        
        T *= cooling
    
    return best, best_sc

def _targeted_local_search(residues, primes, cap, deadline, max_iter=500):
    """Targeted local search: change residues to cover bad points."""
    n = len(primes)
    best = residues[:]
    best_sc = _score(best, primes, cap)
    current = residues[:]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        counts = _build_counts(current, primes, cap)
        L = _find_L(counts, cap)
        if L >= cap:
            break
        below = np.where(counts[1:cap+1] < 2)[0]
        if len(below) == 0:
            break
        bad_x = int(below[0]) + 1
        
        # Try changing 1-2 residues to cover bad_x
        improved = False
        # Single change
        candidates = list(range(n))
        random.shuffle(candidates)
        for idx in candidates:
            old_r = current[idx]
            new_r = bad_x % primes[idx]
            if old_r == new_r:
                continue
            current[idx] = new_r
            new_sc = _score(current, primes, cap)
            if new_sc > best_sc:
                best_sc = new_sc
                best = current[:]
                improved = True
                break
            else:
                current[idx] = old_r
        if improved:
            continue
        
        # Double change
        c1 = random.randint(0, n - 1)
        c2 = random.randint(0, n - 1)
        if c1 == c2:
            continue
        old_r1 = current[c1]
        old_r2 = current[c2]
        current[c1] = bad_x % primes[c1]
        current[c2] = bad_x % primes[c2]
        new_sc = _score(current, primes, cap)
        if new_sc > best_sc:
            best_sc = new_sc
            best = current[:]
        else:
            current[c1] = old_r1
            current[c2] = old_r2
    
    return best, best_sc

def _greedy_init_small_primes(primes):
    """Initialize by assigning small primes to cover small integers densely."""
    n = len(primes)
    residues = [0] * n
    # For each prime, assign residue to cover a target integer
    # Use a pattern that distributes coverage
    for i, p in enumerate(primes):
        # Target integer: spread across 1..n
        target = (i % 50) + 1
        residues[i] = target % p
    return residues

def _random_init(primes):
    """Random initialization."""
    n = len(primes)
    return [random.randint(0, p - 1) for p in primes]

def _structured_init(primes):
    """Structured initialization based on prime indices."""
    n = len(primes)
    residues = [0] * n
    for i, p in enumerate(primes):
        # Alternate between covering odd and even positions
        target = 1 + (i % 30)
        residues[i] = target % p
    return residues

def construct():
    deadline = time.time() + 140  # safety margin
    
    primes = _primes_upto(1000)
    n = len(primes)  # 168
    
    best_residues = None
    best_score = 0
    
    # Multi-restart strategy: 3 independent initializations
    inits = [
        _greedy_init_small_primes(primes),
        _random_init(primes),
        _structured_init(primes)
    ]
    
    for init_idx, init_res in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        residues = init_res[:]
        
        # Phase 1: Greedy first-bad-point repair
        residues = _greedy_first_bad_point(residues, primes, cap=5000, max_iter=50)
        sc = _score(residues, primes, cap=5000)
        if sc > best_score:
            best_score = sc
            best_residues = residues[:]
        
        # Phase 2: Simulated annealing
        if time.time() > deadline - 20:
            break
        sa_res, sa_sc = _simulated_annealing(residues, primes, cap=5000, 
                                              deadline=deadline - 15, iters=1000)
        if sa_sc > best_score:
            best_score = sa_sc
            best_residues = sa_res[:]
        
        # Phase 3: Targeted local search
        if time.time() > deadline - 10:
            break
        tl_res, tl_sc = _targeted_local_search(sa_res, primes, cap=10000,
                                                deadline=deadline - 5, max_iter=300)
        if tl_sc > best_score:
            best_score = tl_sc
            best_residues = tl_res[:]
    
    # Final refinement with larger cap
    if best_residues is not None:
        if time.time() < deadline - 5:
            final_res = _greedy_first_bad_point(best_residues, primes, cap=20000, max_iter=30)
            final_sc = _score(final_res, primes, cap=20000)
            if final_sc > best_score:
                best_score = final_sc
                best_residues = final_res[:]
    
    if best_residues is not None:
        return best_residues
    # Fallback
    return [((i + 2) // 2) % p for i, p in enumerate(primes)]
# EVOLVE-BLOCK-END
