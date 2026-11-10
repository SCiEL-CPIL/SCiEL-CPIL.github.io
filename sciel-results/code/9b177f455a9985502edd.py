import time
import random
import numpy as np

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)

def _build_count_np(residues, L_MAX):
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    # Find largest L such that count[1..L] all >= 2
    # Use numpy for speed
    if L_MAX < 1:
        return 0
    # Find first index where count < 2
    below = np.where(count[1:L_MAX+1] < 2)[0]
    if len(below) == 0:
        return L_MAX
    return int(below[0])  # Cast to Python int

def _apply_move(count, residues, i, new_r, L_LIMIT):
    """Apply a residue change, updating count array in-place."""
    p = _PRIMES[i]
    old_r = residues[i]
    if old_r == new_r:
        return
    start_old = p if old_r == 0 else old_r
    start_new = p if new_r == 0 else new_r
    count[start_old::p] -= 1
    count[start_new::p] += 1
    residues[i] = new_r

def _revert_move(count, residues, i, old_r, L_LIMIT):
    """Revert a residue change."""
    p = _PRIMES[i]
    new_r = residues[i]
    start_old = p if old_r == 0 else old_r
    start_new = p if new_r == 0 else new_r
    count[start_new::p] -= 1
    count[start_old::p] += 1
    residues[i] = old_r

def _greedy_init_small_primes(L_SMALL=500):
    """
    Fix residues for small primes to maximize coverage density in [1, L_SMALL].
    Strategy: For each small prime, choose r_p to maximize the number of x in [1, L_SMALL]
    that would have count >= 2 after adding this prime's coverage.
    """
    residues = [0] * _NPRIMES
    count = np.zeros(L_SMALL + 1, dtype=np.int32)
    
    # Process primes in increasing order (small primes first for density)
    for i, p in enumerate(_PRIMES):
        if p > L_SMALL:
            break  # Large primes don't affect [1, L_SMALL] much
            
        best_r = 0
        best_score = -1
        
        for r in range(p):
            # Simulate adding this residue
            start = p if r == 0 else r
            # Count how many x in [1, L_SMALL] with x % p == r would go from 1 to 2
            # (i.e., currently count[x] == 1, would become 2)
            # And how many would go from 0 to 1 (neutral)
            # We want to maximize the number that reach >= 2
            
            # Get current counts at positions r, r+p, r+2p, ...
            positions = np.arange(start, L_SMALL + 1, p)
            if len(positions) == 0:
                continue
            current_counts = count[positions]
            # Number that would become >= 2 (currently 1)
            would_reach_2 = np.sum(current_counts == 1)
            # Number that would stay >= 2 (currently >= 2)
            stays_2 = np.sum(current_counts >= 2)
            
            # Score: prioritize making more positions reach 2
            score = would_reach_2 * 10 + stays_2
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[i] = best_r
        # Apply
        start = p if best_r == 0 else best_r
        count[start::p] += 1
    
    # For remaining primes (p > L_SMALL), initialize with r=0 for now
    # They'll be optimized later
    for i, p in enumerate(_PRIMES):
        if p > L_SMALL:
            residues[i] = 0
    
    return residues

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    random.seed(42)
    
    # Phase 1: Greedy initialization focusing on small primes
    L_SMALL = 500
    residues = _greedy_init_small_primes(L_SMALL)
    
    # Phase 2: Build count and find initial L
    L_LIMIT = 5000
    count = _build_count_np(residues, L_LIMIT)
    current_L = _find_prefix_L_np(count, L_LIMIT)
    
    best_L = current_L
    best_residues = list(residues)
    
    # Phase 3: Greedy repair with targeted local search
    stagnation_count = 0
    max_stagnation = 30
    
    while time.time() < deadline - 15:
        # Extend window if needed
        if current_L > L_LIMIT - 500:
            new_limit = min(L_LIMIT * 2, 100000)
            count = _build_count_np(residues, new_limit)
            L_LIMIT = new_limit
            current_L = _find_prefix_L_np(count, L_LIMIT)
        
        gap = current_L + 1
        
        if gap > L_LIMIT:
            new_limit = min(L_LIMIT * 2, 100000)
            count = _build_count_np(residues, new_limit)
            L_LIMIT = new_limit
            current_L = _find_prefix_L_np(count, L_LIMIT)
            continue
        
        # Check if gap is already covered
        if count[gap] >= 2:
            # Extend L
            while gap <= L_LIMIT and count[gap] >= 2:
                current_L = gap
                gap += 1
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
            stagnation_count = 0
            continue
        
        # Gap not covered. Find best move to fix it.
        # Try all primes, compute cost (number of x in [1, current_L] that drop below 2)
        best_move = None
        min_cost = float('inf')
        
        # Sort primes: try smaller primes first (higher frequency, more impact)
        prime_order = list(range(_NPRIMES))
        prime_order.sort(key=lambda i: _PRIMES[i])
        
        for i in prime_order:
            p = _PRIMES[i]
            old_r = residues[i]
            new_r = gap % p
            
            if old_r == new_r:
                continue
            
            # Compute cost: number of x in [1, current_L] with x % p == old_r and count[x] == 2
            start_x = p if old_r == 0 else old_r
            if start_x > current_L:
                cost = 0
            else:
                positions = np.arange(start_x, current_L + 1, p)
                cost = np.sum(count[positions] == 2)
            
            # Also consider benefit: does this move help cover more gaps after current_L?
            # For now, just minimize cost
            
            if cost < min_cost:
                min_cost = cost
                best_move = (i, new_r)
                if cost == 0:
                    break  # Can't do better than 0
        
        if best_move is not None:
            i, new_r = best_move
            old_r = residues[i]
            p = _PRIMES[i]
            
            # Apply move
            _apply_move(count, residues, i, new_r, L_LIMIT)
            
            new_L = _find_prefix_L_np(count, L_LIMIT)
            
            if new_L > current_L:
                current_L = new_L
                stagnation_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = list(residues)
            else:
                # Revert
                _revert_move(count, residues, i, old_r, L_LIMIT)
                stagnation_count += 1
        else:
            stagnation_count += 1
        
        # If stuck, do targeted SA on bottleneck primes
        if stagnation_count > max_stagnation:
            # Identify primes that cover the gap and nearby positions
            gap = current_L + 1
            bottleneck_primes = []
            for i, p in enumerate(_PRIMES):
                # If changing this prime's residue could cover the gap
                if (gap % p) != residues[i]:
                    bottleneck_primes.append(i)
            
            # SA on these primes
            T = 3.0
            cooling = 0.95
            min_T = 0.3
            sa_start = time.time()
            
            while time.time() < sa_start + 3.0 and T > min_T:
                if not bottleneck_primes:
                    i = random.randint(0, _NPRIMES - 1)
                else:
                    i = random.choice(bottleneck_primes)
                
                p = _PRIMES[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue
                
                _apply_move(count, residues, i, new_r, L_LIMIT)
                new_L = _find_prefix_L_np(count, L_LIMIT)
                delta = new_L - current_L
                
                if delta > 0:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = list(residues)
                        stagnation_count = 0
                else:
                    if delta == 0:
                        pass  # Keep
                    else:
                        prob = np.exp(delta / T)
                        if random.random() > prob:
                            _revert_move(count, residues, i, old_r, L_LIMIT)
                        else:
                            current_L = new_L
                
                T *= cooling
            
            stagnation_count = 0

    # Final: verify and return best
    return [int(r) for r in best_residues]
