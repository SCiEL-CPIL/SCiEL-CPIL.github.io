"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 88.0  # seconds, safety margin before 90s kill

def _max_abs_ap_sum(signs):
    """
    Compute max |sum| over all arithmetic progressions in [1, n].
    Optimized using vectorized cumsum for each step d.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    
    # Singletons give 1. Pairs give up to 2.
    # We can start with 2 to skip trivial checks if we want, 
    # but let's just compute properly.
    max_disc = 1
    
    # We only need to check steps d where the AP length is >= 2 to potentially exceed 1.
    # For d > n//2, max length is 2. Sum is +/- 2 or 0.
    # So max possible for d > n//2 is 2.
    # We can set max_disc = max(max_disc, 2) initially if n >= 2.
    if n >= 2:
        max_disc = 2

    for d in range(1, n):
        # If d is large, the sequences are short.
        # If d > n//2, max length is 2. We already covered max 2.
        # If d > n//3, max length is 3. Max sum 3.
        # We can skip d if the theoretical max sum for this d is <= current max_disc.
        # Theoretical max sum for step d is floor((n-1)/d) + 1.
        max_len = (n - 1) // d + 1
        if max_len <= max_disc:
            continue
        
        # For this d, process all residues r in 0..d-1
        # We can vectorize this by creating a matrix.
        # However, creating a matrix for each d might be overhead.
        # Let's try iterating residues but using numpy for the cumsum.
        
        # Optimization: If d is small, we have many long sequences.
        # If d is large, we have few short sequences.
        
        for r in range(d):
            seq = signs[r::d]
            if len(seq) <= max_disc:
                continue
            csum = np.cumsum(seq)
            m = np.max(np.abs(csum))
            if m > max_disc:
                max_disc = m
                # Early exit if we hit a theoretical limit? No, global max is unknown.
                # But if max_disc gets large, we can prune more.
                
    return max_disc

def _legendre_init():
    """
    Initialize using Legendre symbol modulo a prime near 256.
    """
    n = _N
    p = 257
    
    signs = np.zeros(n, dtype=np.int32)
    for i in range(n):
        pos = i + 1
        if pos % p == 0:
            signs[i] = 1
        else:
            ls = pow(pos, (p - 1) // 2, p)
            if ls == 1:
                signs[i] = 1
            elif ls == p - 1:
                signs[i] = -1
            else:
                signs[i] = 1
    
    return signs.tolist()

def _local_search(signs_list, max_time, rng, T_start=10.0, T_end=0.01, cooling=0.999):
    """
    Simulated annealing / Hill climbing hybrid.
    """
    n = _N
    signs = np.array(signs_list, dtype=np.int32)
    current_disc = _max_abs_ap_sum(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    start_time = time.time()
    T = T_start
    no_improve_count = 0
    max_no_improve = 500  # Reset or restart if stuck
    
    while time.time() - start_time < max_time:
        # Pick a random position to flip
        i = rng.randint(0, n - 1)
        old_sign = signs[i]
        signs[i] = -old_sign
        
        new_disc = _max_abs_ap_sum(signs)
        dE = new_disc - current_disc
        
        if dE <= 0:
            # Accept improvement
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Metropolis criterion
            if T > T_end and rng.random() < np.exp(-dE / T):
                current_disc = new_disc
                no_improve_count += 1
            else:
                # Revert
                signs[i] = old_sign
                no_improve_count += 1
        
        # Cool down
        T = max(T * cooling, T_end)
        
        # If stuck for a long time, try a random restart or perturbation
        if no_improve_count > max_no_improve:
            # Perturb: flip 5 random signs
            for _ in range(5):
                j = rng.randint(0, n - 1)
                signs[j] = -signs[j]
            current_disc = _max_abs_ap_sum(signs)
            no_improve_count = 0
            T = max(T * 1.5, 1.0) # Reheat slightly

    return best_signs.tolist(), best_disc

def construct():
    """
    Construct 256 signs minimizing max |sum| over all arithmetic progressions.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_signs = None
    best_disc = float('inf')
    
    # Phase 1: Algebraic initialization
    legendre_signs = _legendre_init()
    legendre_disc = _max_abs_ap_sum(legendre_signs)
    if legendre_disc < best_disc:
        best_disc = legendre_disc
        best_signs = legendre_signs
    
    # Phase 2: Local search from Legendre
    remaining_time = _DEADLINE - (time.time() - start_time)
    if remaining_time > 2:
        sa_signs, sa_disc = _local_search(legendre_signs, min(remaining_time - 1, 30), rng)
        if sa_disc < best_disc:
            best_disc = sa_disc
            best_signs = sa_signs
    
    # Phase 3: Multi-start with random inits
    # We have remaining time for more restarts
    while time.time() - start_time < _DEADLINE - 5:
        random_signs = [rng.choice((-1, 1)) for _ in range(_N)]
        
        # Short local search
        time_left = _DEADLINE - 5 - (time.time() - start_time)
        if time_left < 1:
            break
        
        # Allocate a portion of the remaining time to this restart
        # e.g., 50% of remaining time, capped at 10s per restart
        search_time = min(10.0, time_left * 0.5)
        
        sa_signs, sa_disc = _local_search(random_signs, search_time, rng, T_start=5.0, T_end=0.01)
        
        if sa_disc < best_disc:
            best_disc = sa_disc
            best_signs = sa_signs
    
    return best_signs
# EVOLVE-BLOCK-END
