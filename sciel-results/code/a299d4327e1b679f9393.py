"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_primes():
    """Return list of primes <= 1000."""
    primes = []
    for p in range(2, 1001):
        if all(p % q for q in range(2, int(p ** 0.5) + 1)):
            primes.append(p)
    return primes

def _compute_L(residues, primes, max_L=100000):
    """Compute the largest L such that every integer 1..L is covered by at least 2 congruences."""
    L = min(max_L, 10000)
    # Vectorized: for each prime i, compute which integers in 1..L satisfy x ≡ residues[i] (mod primes[i])
    x = np.arange(1, L + 1)
    coverage = np.zeros(L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        # Integers x in [1, L] with x ≡ r (mod p)
        # x = r + k*p, where k >= 0 and x <= L
        # First valid x: if r == 0, x = p (since x >= 1 and x ≡ 0 mod p means x = p, 2p, ...)
        # if r > 0, x = r, r+p, r+2p, ...
        if r == 0:
            start = p
        else:
            start = r
        if start > L:
            continue
        idx = (np.arange(start, L + 1, p)) - 1  # 0-based index into x
        coverage[idx + 1] += 1
    # Find largest L such that coverage[1..L] >= 2
    for L in range(1, min(L, 100000) + 1):
        if coverage[L] < 2:
            return L - 1
    return L

def _compute_L_fast(residues, primes, current_L):
    """Faster check: only verify up to current_L + some margin."""
    L = min(current_L + 200, 100000)
    x = np.arange(1, L + 1)
    coverage = np.zeros(L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start > L:
            continue
        idx = (np.arange(start, L + 1, p)) - 1
        coverage[idx + 1] += 1
    for L in range(1, min(L, 100000) + 1):
        if coverage[L] < 2:
            return L - 1
    return L

def _greedy_construct(primes):
    """Greedy: assign each prime to cover a specific integer, aiming for double coverage."""
    n = len(primes)
    residues = [0] * n
    # Target: cover integers 1..n/2 twice
    target_count = n // 2
    # Assign primes to targets: prime i covers target (i % target_count) + 1
    for i, p in enumerate(primes):
        target = (i % target_count) + 1
        residues[i] = target % p
    return residues

def _local_search(residues, primes, max_time=120.0):
    """Local search with simulated annealing to improve L."""
    start_time = time.time()
    deadline = start_time + max_time - 10.0  # 10s safety margin
    
    best_L = _compute_L(residues, primes, 10000)
    best_residues = residues[:]
    
    current_residues = residues[:]
    current_L = best_L
    
    T = 100.0  # initial temperature
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        T *= cooling
        if T < T_min:
            break
        
        # Pick a random prime to change
        i = np.random.randint(0, len(primes))
        old_r = current_residues[i]
        new_r = np.random.randint(0, primes[i])
        current_residues[i] = new_r
        
        new_L = _compute_L_fast(current_residues, primes, current_L)
        
        if new_L >= current_L:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = current_residues[:]
        else:
            # Accept with probability exp(-(current_L - new_L) / T)
            delta = current_L - new_L
            if np.random.rand() < np.exp(-delta / T):
                current_L = new_L
            else:
                current_residues[i] = old_r
    
    # Final check with full computation
    final_L = _compute_L(best_residues, primes, 100000)
    if final_L > best_L:
        best_L = final_L
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    primes = _get_primes()
    
    # Start with greedy construction
    residues = _greedy_construct(primes)
    L = _compute_L(residues, primes, 10000)
    
    # Local search / simulated annealing
    if time.time() < deadline - 5.0:
        residues, L = _local_search(residues, primes, max_time=deadline - start_time)
    
    return residues
# EVOLVE-BLOCK-END
