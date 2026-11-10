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
# Precompute prime indices for fast access
_PRIMES_ARR = np.array(_PRIMES, dtype=np.int64)

def _build_count_np(residues, L_MAX):
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    if L_MAX < 1:
        return 0
    below = np.where(count[1:L_MAX+1] < 2)[0]
    if len(below) == 0:
        return L_MAX
    return int(below[0])

def _apply_move(count, residues, i, new_r, L_LIMIT):
    p = _PRIMES[i]
    old_r = residues[i]
    if old_r == new_r:
        return
    start_old = p if old_r == 0 else old_r
    start_new = p if new_r == 0 else new_r
    if start_old <= L_LIMIT:
        count[start_old::p] -= 1
    if start_new <= L_LIMIT:
        count[start_new::p] += 1
    residues[i] = new_r

def _revert_move(count, residues, i, old_r, L_LIMIT):
    p = _PRIMES[i]
    new_r = residues[i]
    start_old = p if old_r == 0 else old_r
    start_new = p if new_r == 0 else new_r
    if start_new <= L_LIMIT:
        count[start_new::p] -= 1
    if start_old <= L_LIMIT:
        count[start_old::p] += 1
    residues[i] = old_r

def _greedy_init(order_seed, L_WORK=3000):
    """Greedy initialization: process primes in given order, pick best residue."""
    rng = random.Random(order_seed)
    residues = [0] * _NPRIMES
    count = np.zeros(L_WORK + 1, dtype=np.int32)
    
    # Process primes in the given order
    for i in order_seed:
        p = _PRIMES[i]
        best_r = 0
        best_score = -1
        
        if p <= L_WORK:
            for r in range(p):
                start = p if r == 0 else r
                if start > L_WORK:
                    continue
                positions = np.arange(start, L_WORK + 1, p)
                if len(positions) == 0:
                    continue
                current_counts = count[positions]
                would_reach_2 = np.sum(current_counts == 1)
                stays_2 = np.sum(current_counts >= 2)
                # Prefer moves that help close gaps (count==1 -> 2)
                score = would_reach_2 * 100 + stays_2
                if score > best_score:
                    best_score = score
                    best_r = r
        else:
            best_r = rng.randint(0, p - 1)
        
        residues[i] = best_r
        start = p if best_r == 0 else best_r
        if start <= L_WORK:
            count[start::p] += 1
    
    return residues

def _gap_repair(count, residues, current_L, L_LIMIT, time_limit):
    """Targeted gap repair: find first gap, fix it with minimal cost move."""
    end_time = time.time() + time_limit
    improved = True
    
    while improved and time.time() < end_time:
        improved = False
        
        if current_L >= L_LIMIT:
            break
        
        gap = current_L + 1
        if gap > L_LIMIT:
            break
        
        if count[gap] >= 2:
            # Already covered, advance
            while gap <= L_LIMIT and count[gap] >= 2:
                current_L = gap
                gap += 1
            continue
        
        # Find best move to fix gap: try each prime, compute cost
        best_move = None
        min_cost = float('inf')
        
        # Try primes in order of increasing p (smaller primes have more positions, higher cost)
        # But we want low cost, so try larger primes first (fewer positions affected)
        prime_indices = list(range(_NPRIMES))
        prime_indices.sort(key=lambda i: -_PRIMES[i])  # larger primes first
        
        for i in prime_indices:
            p = _PRIMES[i]
            old_r = residues[i]
            new_r = gap % p
            
            if old_r == new_r:
                continue
            
            # Compute cost: number of positions in prefix that would drop below 2
            start_old = p if old_r == 0 else old_r
            if start_old <= current_L:
                positions = np.arange(start_old, current_L + 1, p)
                if len(positions) > 0:
                    cost = np.sum(count[positions] == 2)
                else:
                    cost = 0
            else:
                cost = 0
            
            if cost < min_cost:
                min_cost = cost
                best_move = (i, new_r)
                if cost == 0:
                    break
        
        if best_move is not None:
            i, new_r = best_move
            old_r = residues[i]
            p = _PRIMES[i]
            
            _apply_move(count, residues, i, new_r, L_LIMIT)
            new_L = _find_prefix_L_np(count, L_LIMIT)
            
            if new_L > current_L:
                current_L = new_L
                improved = True
            else:
                _revert_move(count, residues, i, old_r, L_LIMIT)
                # Try random moves to escape
                for _ in range(50):
                    if time.time() > end_time:
                        break
                    j = random.randint(0, _NPRIMES - 1)
                    pj = _PRIMES[j]
                    old_rj = residues[j]
                    new_rj = random.randint(0, pj - 1)
                    if new_rj == old_rj:
                        continue
                    _apply_move(count, residues, j, new_rj, L_LIMIT)
                    new_L = _find_prefix_L_np(count, L_LIMIT)
                    if new_L > current_L:
                        current_L = new_L
                        improved = True
                        break
                    else:
                        _revert_move(count, residues, j, old_rj, L_LIMIT)
    
    return current_L

def _sa_search(count, residues, current_L, L_LIMIT, time_limit, T_init=50.0):
    """Simulated annealing search with incremental updates."""
    T = T_init
    best_res = list(residues)
    best_L = current_L
    no_improve = 0
    no_improve_limit = 5000
    
    end_time = time.time() + time_limit
    
    while time.time() < end_time:
        # Pick a prime to modify
        idx = random.randint(0, _NPRIMES - 1)
        p = _PRIMES[idx]
        old_r = residues[idx]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Compute loss: positions in prefix that would drop below 2
        start_old = p if old_r == 0 else old_r
        loss = 0
        if start_old <= current_L:
            positions = np.arange(start_old, current_L + 1, p)
            if len(positions) > 0:
                loss = int(np.sum(count[positions] == 2))
        
        if loss > 0:
            # Accept with probability based on temperature
            if T < 1.0 or random.random() > (0.001 / (1 + loss * 0.1)):
                no_improve += 1
                T *= 0.999
                if T < 0.05:
                    T = 0.05
                continue
        else:
            # Apply move
            _apply_move(count, residues, idx, new_r, L_LIMIT)
            new_L = _find_prefix_L_np(count, L_LIMIT)
            
            if new_L >= current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_res = list(residues)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                _revert_move(count, residues, idx, old_r, L_LIMIT)
                no_improve += 1
        
        if no_improve > no_improve_limit:
            T = min(T * 3.0, T_init)
            no_improve = 0
        
        T *= 0.999
        if T < 0.05:
            T = 0.05
    
    return best_res, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_overall_L = 0
    best_overall_residues = None
    
    # Multi-restart strategy
    # Time allocation: multiple restarts with greedy init + SA + gap repair
    num_restarts = 4
    time_per_restart = (deadline - start_time - 10.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Different prime ordering for diversity
        prime_order = list(range(_NPRIMES))
        random.seed(42 + restart * 137)
        random.shuffle(prime_order)
        
        # Phase 1: Greedy init
        residues = _greedy_init(prime_order, L_WORK=3000)
        
        # Phase 2: Build count and find initial L
        L_LIMIT = 5000
        count = _build_count_np(residues, L_LIMIT)
        current_L = _find_prefix_L_np(count, L_LIMIT)
        
        # Phase 3: Gap repair
        time_for_repair = time_per_restart * 0.3
        current_L = _gap_repair(count, residues, current_L, L_LIMIT, time_for_repair)
        
        # Phase 4: SA search
        time_for_sa = time_per_restart * 0.5
        best_res, best_L = _sa_search(count, residues, current_L, L_LIMIT, time_for_sa)
        
        if best_L > best_overall_L:
            best_overall_L = best_L
            best_overall_residues = list(best_res)
        
        # Phase 5: Final gap repair on best
        if time.time() < deadline - 5.0:
            count = _build_count_np(best_res, L_LIMIT)
            current_L = _find_prefix_L_np(count, L_LIMIT)
            current_L = _gap_repair(count, best_res, current_L, L_LIMIT, min(5.0, deadline - time.time() - 3.0))
            if current_L > best_overall_L:
                best_overall_L = current_L
                best_overall_residues = list(best_res)
    
    # Final verification
    if best_overall_residues is None:
        best_overall_residues = [0] * _NPRIMES
    
    # Ensure we return valid residues
    return [int(r) for r in best_overall_residues]
