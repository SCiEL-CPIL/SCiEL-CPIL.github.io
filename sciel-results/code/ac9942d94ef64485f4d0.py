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
        r = int(residues[i])
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    if L_MAX < 1:
        return 0
    # Find first index where count < 2
    below = np.where(count[1:L_MAX+1] < 2)[0]
    if len(below) == 0:
        return L_MAX
    return int(below[0])

def _evaluate_move(count, L_current, i, new_r):
    """
    Evaluate the effect of changing residue of prime i to new_r.
    Returns the new prefix length if the move is applied.
    Assumes count is the current count array.
    """
    p = _PRIMES[i]
    old_r = int(count.__dict__.get('residues', [0]*_NPRIMES)[i]) # This is inefficient, pass residues
    
    # We need the old residue to calculate the delta.
    # It's better to pass the residues array or store it.
    pass

def _apply_move(count, i, old_r, new_r, L_MAX):
    p = _PRIMES[i]
    if old_r == 0:
        start_old = p
    else:
        start_old = old_r
    if new_r == 0:
        start_new = p
    else:
        start_new = new_r
        
    if start_old <= L_MAX:
        count[start_old::p] -= 1
    if start_new <= L_MAX:
        count[start_new::p] += 1

def _revert_move(count, i, old_r, new_r, L_MAX):
    _apply_move(count, i, new_r, old_r, L_MAX)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_L_global = 0
    best_residues_global = [0] * _NPRIMES
    
    # We will run multiple restarts if time allows
    # Each restart does a greedy init + local search + kicks
    
    L_LIMIT = 50000
    
    for restart in range(50): # Max 50 restarts, will stop earlier if time runs out
        if time.time() > deadline - 5:
            break
            
        random.seed(42 + restart)
        residues = [0] * _NPRIMES
        
        # 1. Smart Initialization
        # Greedy for first 30 primes to maximize early coverage
        L_INIT = 2000
        count_init = np.zeros(L_INIT + 1, dtype=np.int32)
        
        for i, p in enumerate(_PRIMES[:30]):
            best_r = 0
            best_score = -1
            for r in range(p):
                start = p if r == 0 else r
                if start <= L_INIT:
                    positions = np.arange(start, L_INIT + 1, p)
                    # Score: number of x with count==1 (potential to become 2) 
                    # minus number of x with count>=2 (potential to become 3, less critical)
                    # Actually, we just want to maximize the number of x that become covered by 2.
                    # If count[x] == 1, adding this makes it 2 (good).
                    # If count[x] == 0, adding this makes it 1 (neutral/less good).
                    # If count[x] >= 2, adding this makes it >=3 (neutral).
                    score = np.sum(count_init[positions] == 1)
                else:
                    score = 0
                if score > best_score:
                    best_score = score
                    best_r = r
            residues[i] = best_r
            r = best_r
            start = p if r == 0 else r
            if start <= L_INIT:
                count_init[start::p] += 1

        # Random for the rest
        for i in range(30, _NPRIMES):
            p = _PRIMES[i]
            residues[i] = random.randint(0, p - 1)

        # Build full count
        count = _build_count_np(residues, L_LIMIT)
        current_L = _find_prefix_L_np(count, L_LIMIT)
        
        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = [int(r) for r in residues]

        # 2. Local Search with Kicks
        stagnation = 0
        max_stagnation = 50
        kick_count = 0
        
        while time.time() < deadline - 5:
            # Check if we need to extend L_LIMIT
            if current_L > L_LIMIT - 1000:
                new_limit = L_LIMIT * 2
                count = _build_count_np(residues, new_limit)
                L_LIMIT = new_limit
                current_L = _find_prefix_L_np(count, L_LIMIT)
                if current_L > best_L_global:
                    best_L_global = current_L
                    best_residues_global = [int(r) for r in residues]
                continue

            gap = current_L + 1
            
            if gap > L_LIMIT:
                new_limit = L_LIMIT * 2
                count = _build_count_np(residues, new_limit)
                L_LIMIT = new_limit
                current_L = _find_prefix_L_np(count, L_LIMIT)
                continue

            # If gap is already covered (shouldn't happen if current_L is correct, but safe check)
            if count[gap] >= 2:
                # This means our prefix L calculation was wrong or count changed
                # Recalculate
                current_L = _find_prefix_L_np(count, L_LIMIT)
                if current_L > best_L_global:
                    best_L_global = current_L
                    best_residues_global = [int(r) for r in residues]
                continue

            # Find best move to cover 'gap'
            best_move = None
            min_cost = float('inf')
            
            # Candidate primes: those where gap % p is different from current residue
            # We prioritize smaller primes as they cover more numbers in the prefix
            
            # To speed up, we can limit the search to primes where the change is "cheap"
            # Cost = number of x in [1, current_L] with count[x]==2 and x%p==old_r
            # We want to minimize this cost.
            
            for i, p in enumerate(_PRIMES):
                old_r = int(residues[i])
                new_r = gap % p
                
                if old_r == new_r:
                    continue
                
                # Calculate cost
                if old_r == 0:
                    start_x = p
                else:
                    start_x = old_r
                
                if start_x > current_L:
                    cost = 0
                else:
                    positions = np.arange(start_x, current_L + 1, p)
                    cost = int(np.sum(count[positions] == 2))
                
                if cost < min_cost:
                    min_cost = cost
                    best_move = (i, new_r)
                    if cost == 0:
                        break # Found a free move, best possible
            
            if best_move is not None:
                i, new_r = best_move
                p = _PRIMES[i]
                old_r = int(residues[i])
                
                # Apply move
                _apply_move(count, i, old_r, new_r, L_LIMIT)
                residues[i] = new_r
                
                new_L = _find_prefix_L_np(count, L_LIMIT)
                
                if new_L > current_L:
                    current_L = new_L
                    stagnation = 0
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = [int(r) for r in residues]
                else:
                    # Revert
                    _revert_move(count, i, old_r, new_r, L_LIMIT)
                    residues[i] = old_r
                    stagnation += 1
            else:
                stagnation += 1
                
            # Kick if stagnant
            if stagnation > max_stagnation:
                # Perform a kick: change a few random residues
                num_kicks = random.randint(3, 10)
                for _ in range(num_kicks):
                    i = random.randint(0, _NPRIMES - 1)
                    p = _PRIMES[i]
                    old_r = int(residues[i])
                    new_r = random.randint(0, p - 1)
                    if new_r != old_r:
                        _apply_move(count, i, old_r, new_r, L_LIMIT)
                        residues[i] = new_r
                        
                # Recalculate L
                current_L = _find_prefix_L_np(count, L_LIMIT)
                if current_L > best_L_global:
                    best_L_global = current_L
                    best_residues_global = [int(r) for r in residues]
                
                stagnation = 0
                max_stagnation = random.randint(50, 100) # Increase patience after kick
                kick_count += 1
                
                # If we've kicked too many times without improvement, restart
                if kick_count > 20:
                    break

    # Final verification with a larger limit just in case
    L_MAX_FINAL = 100000
    count_final = _build_count_np(best_residues_global, L_MAX_FINAL)
    final_L = _find_prefix_L_np(count_final, L_MAX_FINAL)
    
    if final_L > best_L_global:
        best_L_global = final_L

    return [int(r) for r in best_residues_global]
