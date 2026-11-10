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
        if r == 0:
            start = p
        else:
            start = r
        if start <= L_MAX:
            count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    slice_view = count[1:L_MAX+1]
    if slice_view.min() >= 2:
        return L_MAX
    first_bad = np.argmin(slice_view < 2)
    return first_bad

def _score_full(residues):
    L_MAX = 100000
    count = _build_count_np(residues, L_MAX)
    return _find_prefix_L_np(count, L_MAX)

def _coverage_deficit(residues, L_LIMIT):
    """Compute coverage deficit for each position in [1, L_LIMIT]."""
    count = _build_count_np(residues, L_LIMIT)
    deficit = np.zeros(L_LIMIT + 1, dtype=np.int32)
    for x in range(1, L_LIMIT + 1):
        if count[x] < 2:
            deficit[x] = 2 - count[x]
    return deficit

def _greedy_cs_assign(residues, L_LIMIT):
    """
    Constraint satisfaction approach: fix small primes to maximize density,
    then use coverage deficit to assign larger primes.
    """
    # Phase 1: Fix small primes (p <= 50) to maximize initial coverage density
    small_primes = [p for p in _PRIMES if p <= 50]
    small_indices = [i for i, p in enumerate(_PRIMES) if p <= 50]
    
    # Try all combinations for small primes? No, too many. Use greedy.
    # For small primes, try to cover [1, 200] densely.
    for idx in small_indices:
        p = _PRIMES[idx]
        # Try all residues for this prime, pick the one that maximizes coverage in [1, 200]
        best_r = 0
        best_score = -1
        for r in range(p):
            # Temporarily set residue
            old_r = residues[idx]
            residues[idx] = r
            
            # Quick score: count how many positions in [1, 200] have count >= 2
            # This is expensive, so use a proxy: sum of counts in [1, 200]
            count = _build_count_np(residues, 200)
            score = np.sum(count[1:201] >= 2)
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[idx] = best_r
    
    # Phase 2: Use coverage deficit to assign larger primes
    # Process primes in increasing order, but use deficit to guide assignment
    for idx in range(len(_PRIMES)):
        p = _PRIMES[idx]
        
        # Compute current deficit
        count = _build_count_np(residues, L_LIMIT)
        deficit = 2 - count
        deficit = np.maximum(deficit, 0)
        
        # Find the position with highest deficit that can be covered by this prime
        # For each possible residue r, compute how much deficit it would reduce
        best_r = 0
        best_deficit_reduction = -1
        
        for r in range(p):
            # Compute how many positions with deficit > 0 would be covered by residue r
            if r == 0:
                start = p
            else:
                start = r
            
            if start > L_LIMIT:
                continue
            
            # Positions: start, start+p, start+2p, ...
            positions = np.arange(start, L_LIMIT + 1, p)
            # Sum of deficits at these positions
            reduction = np.sum(deficit[positions])
            
            if reduction > best_deficit_reduction:
                best_deficit_reduction = reduction
                best_r = r
        
        residues[idx] = best_r
    
    return residues

def _local_search_bottleneck(residues, L_LIMIT, max_iters=5000):
    """
    Targeted local search: focus on primes that cover the current bottleneck.
    """
    count = _build_count_np(residues, L_LIMIT)
    current_L = _find_prefix_L_np(count, L_LIMIT)
    best_L = current_L
    best_residues = list(residues)
    
    for it in range(max_iters):
        # Find the bottleneck position
        if current_L >= L_LIMIT:
            break
        bottleneck = current_L + 1
        if count[bottleneck] >= 2:
            # Extend L
            new_L = _find_prefix_L_np(count, L_LIMIT)
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = list(residues)
            continue
        
        # Find primes that could cover the bottleneck
        # A prime p can cover bottleneck if we set r_p = bottleneck % p
        # Identify which primes currently cover the bottleneck (count[bottleneck] < 2)
        
        # Strategy: try to move a prime to cover the bottleneck
        # Cost: how many positions in [1, current_L] with count == 2 would lose coverage
        
        best_move = None
        min_cost = float('inf')
        
        # Only consider primes that don't already cover the bottleneck
        for i, p in enumerate(_PRIMES):
            new_r = bottleneck % p
            old_r = residues[i]
            
            if old_r == new_r:
                continue
            
            # Calculate cost: positions in [1, current_L] with count == 2 that lose this prime's coverage
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            
            if start_old > current_L:
                cost = 0
            else:
                positions = np.arange(start_old, current_L + 1, p)
                cost = np.sum(count[positions] == 2)
            
            if cost < min_cost:
                min_cost = cost
                best_move = (i, new_r, old_r)
                if cost == 0:
                    break
        
        if best_move is not None:
            i, new_r, old_r = best_move
            p = _PRIMES[i]
            
            # Apply move
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r
            
            count[start_old:L_LIMIT+1:p] -= 1
            count[start_new:L_LIMIT+1:p] += 1
            residues[i] = new_r
            
            new_L = _find_prefix_L_np(count, L_LIMIT)
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = list(residues)
            else:
                # Revert
                count[start_new:L_LIMIT+1:p] -= 1
                count[start_old:L_LIMIT+1:p] += 1
                residues[i] = old_r
        else:
            # No safe move, try random perturbation
            i = random.randint(0, _NPRIMES - 1)
            p = _PRIMES[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r
            
            count[start_old:L_LIMIT+1:p] -= 1
            count[start_new:L_LIMIT+1:p] += 1
            residues[i] = new_r
            
            new_L = _find_prefix_L_np(count, L_LIMIT)
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = list(residues)
            else:
                # Revert
                count[start_new:L_LIMIT+1:p] -= 1
                count[start_old:L_LIMIT+1:p] += 1
                residues[i] = old_r
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Phase 1: Greedy constraint satisfaction assignment
    L_LIMIT = 5000
    residues = _greedy_cs_assign([0] * _NPRIMES, L_LIMIT)
    
    # Phase 2: Targeted local search on bottleneck
    residues, current_L = _local_search_bottleneck(residues, L_LIMIT, max_iters=10000)
    
    # Phase 3: Expand window and continue search if time allows
    while time.time() < deadline - 15:
        # Expand window if close to limit
        if current_L > L_LIMIT - 200:
            L_LIMIT = L_LIMIT * 2
            count = _build_count_np(residues, L_LIMIT)
            current_L = _find_prefix_L_np(count, L_LIMIT)
        
        # Continue local search
        old_L = current_L
        residues, current_L = _local_search_bottleneck(residues, L_LIMIT, max_iters=5000)
        
        if current_L == old_L:
            # Stagnation, try different L_LIMIT or random restart
            if time.time() < deadline - 30:
                # Random perturbation and retry
                for _ in range(100):
                    i = random.randint(0, _NPRIMES - 1)
                    p = _PRIMES[i]
                    residues[i] = random.randint(0, p - 1)
                
                count = _build_count_np(residues, L_LIMIT)
                current_L = _find_prefix_L_np(count, L_LIMIT)
                
                if current_L > old_L:
                    residues, current_L = _local_search_bottleneck(residues, L_LIMIT, max_iters=5000)
    
    # Final verification with full range
    L_MAX_FINAL = 100000
    count_final = _build_count_np(residues, L_MAX_FINAL)
    final_L = _find_prefix_L_np(count_final, L_MAX_FINAL)
    
    return residues
