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
_L_LIMIT = 10000

def _build_count(residues, limit=None):
    if limit is None:
        limit = _L_LIMIT
    count = np.zeros(limit + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        if start <= limit:
            count[start::p] += 1
    return count

def _find_L(count, limit=None):
    if limit is None:
        limit = len(count) - 1
    below = np.where(count[1:limit+1] < 2)[0]
    if len(below) == 0:
        return limit
    return int(below[0])

def _coverage_deficit_greedy(deadline):
    """
    Structural change: Model as constraint satisfaction.
    1. Fix small primes to maximize coverage density in [1, 2000].
    2. Use 'coverage deficit' heuristic for larger primes to repair the first uncovered integer.
    3. Iterate until budget exhausted.
    """
    residues = [0] * _NPRIMES
    limit = 2000
    count = _build_count(residues, limit)
    
    # Phase 1: Initialize small primes to cover [1, 2000] densely
    # We want to ensure that for any x in [1, 2000], it is covered by many primes.
    # A simple strategy: for each prime p, choose r_p such that the sequence r_p, r_p+p, ...
    # hits as many "weak" points as possible.
    
    # Let's start with a randomized restart strategy within this function
    # to find a good configuration.
    
    best_L = 0
    best_res = None
    
    n_restarts = 5
    time_per_restart = 10.0
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        residues = [0] * _NPRIMES
        # Random initialization for diversity
        for i, p in enumerate(_PRIMES):
            residues[i] = random.randint(0, p - 1)
        
        count = _build_count(residues, limit)
        current_L = _find_L(count, limit)
        
        # Local search loop
        max_iters = 2000
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Find the first gap
            if current_L >= limit:
                break
            
            gap = current_L + 1
            if count[gap] >= 2:
                # Update L
                while gap <= limit and count[gap] >= 2:
                    current_L = gap
                    gap += 1
                if current_L > best_L:
                    best_L = current_L
                    best_res = list(residues)
                continue
            
            # We need to cover 'gap' with at least 2 primes.
            # Currently count[gap] < 2.
            # Strategy: Pick a prime p that currently does NOT cover 'gap' (i.e., gap % p != residues[i])
            # and change its residue to gap % p.
            # To minimize damage, pick the prime p that covers the fewest points in [1, current_L]
            # where the count is exactly 2 (i.e., removing it would break coverage there).
            
            best_move = None
            min_cost = float('inf')
            
            # Only consider primes that are "large enough" or randomly selected to avoid O(N^2) if N is large
            # But N=168 is small, so we can check all.
            
            # Optimization: If we have many primes, we can sample. But let's stick to all for now.
            for i in range(_NPRIMES):
                p = _PRIMES[i]
                r_new = gap % p
                r_old = residues[i]
                
                if r_new == r_old:
                    continue
                
                # Cost: number of points x in [1, current_L] such that x is covered by prime i
                # AND count[x] == 2. If we move prime i, these points drop to 1.
                
                # Calculate cost efficiently
                start_old = p if r_old == 0 else r_old
                # We need to count how many x in [start_old, current_L] with step p have count[x] == 2
                if start_old <= current_L:
                    pos = np.arange(start_old, current_L + 1, p)
                    cost = np.sum(count[pos] == 2)
                else:
                    cost = 0
                
                # Bonus: if this prime was not covering 'gap', it adds 1 to gap.
                # We are moving it TO cover 'gap'.
                
                if cost < min_cost:
                    min_cost = cost
                    best_move = (i, r_new)
                    if cost == 0:
                        break
            
            if best_move is not None:
                i, r_new = best_move
                p = _PRIMES[i]
                r_old = residues[i]
                
                start_old = p if r_old == 0 else r_old
                start_new = p if r_new == 0 else r_new
                
                count[start_old::p] -= 1
                count[start_new::p] += 1
                residues[i] = r_new
                
                # Recalculate L (it might have jumped further)
                new_L = _find_L(count, limit)
                if new_L > current_L:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_res = list(residues)
                else:
                    # It didn't improve L? That's strange if we fixed the gap.
                    # Maybe it broke something earlier?
                    # If L didn't increase, we might want to revert if it didn't help.
                    # But the heuristic says "repair the first uncovered integer".
                    # If L didn't increase, it means we didn't actually cover the gap?
                    # Or we covered the gap but broke an earlier point?
                    # If we broke an earlier point, L would decrease.
                    # _find_L returns the first gap. If we broke a point < current_L, L would drop.
                    # If we covered 'gap' (which was current_L+1), L should become at least current_L+1.
                    # Unless 'gap' is still < 2?
                    # We moved a prime to cover 'gap'. So count[gap] increases by 1.
                    # If count[gap] was 0 or 1, it becomes 1 or 2.
                    # If it becomes 2, and all previous points are >= 2, then L increases.
                    # If we broke a previous point (cost > 0), L might drop.
                    # So if new_L <= current_L, it means we broke something earlier.
                    # Revert.
                    count[start_new::p] -= 1
                    count[start_old::p] += 1
                    residues[i] = r_old
            else:
                # No good move found. Stuck.
                # Try a random perturbation to escape local minimum
                i = random.randint(0, _NPRIMES - 1)
                p = _PRIMES[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                if new_r != old_r:
                    start_old = p if old_r == 0 else old_r
                    start_new = p if new_r == 0 else new_r
                    count[start_old::p] -= 1
                    count[start_new::p] += 1
                    residues[i] = new_r
                    current_L = _find_L(count, limit)
    
    return best_L, best_res

def _targeted_refinement(residues, deadline):
    """
    Refine the solution from the greedy phase.
    Use a more sophisticated local search that considers pairs of moves
    or looks further ahead.
    """
    limit = _L_LIMIT
    count = _build_count(residues, limit)
    current_L = _find_L(count, limit)
    best_L = current_L
    best_res = list(residues)
    
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        if current_L >= limit:
            break
        
        gap = current_L + 1
        if count[gap] >= 2:
            while gap <= limit and count[gap] >= 2:
                current_L = gap
                gap += 1
            if current_L > best_L:
                best_L = current_L
                best_res = list(residues)
            continue
        
        # Try to fix 'gap'
        # Look for a prime to move to 'gap' with minimal cost.
        # Cost: number of points in [1, current_L] where count == 2 and covered by this prime.
        
        best_move = None
        min_cost = float('inf')
        
        # Sample a subset of primes to speed up if needed, but 168 is small.
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            r_new = gap % p
            r_old = residues[i]
            
            if r_new == r_old:
                continue
            
            start_old = p if r_old == 0 else r_old
            if start_old <= current_L:
                pos = np.arange(start_old, current_L + 1, p)
                cost = np.sum(count[pos] == 2)
            else:
                cost = 0
            
            if cost < min_cost:
                min_cost = cost
                best_move = (i, r_new)
                if cost == 0:
                    break
        
        if best_move:
            i, r_new = best_move
            p = _PRIMES[i]
            r_old = residues[i]
            
            start_old = p if r_old == 0 else r_old
            start_new = p if r_new == 0 else r_new
            
            count[start_old::p] -= 1
            count[start_new::p] += 1
            residues[i] = r_new
            
            new_L = _find_L(count, limit)
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_res = list(residues)
            else:
                # Revert
                count[start_new::p] -= 1
                count[start_old::p] += 1
                residues[i] = r_old
        else:
            # Stuck, random kick
            i = random.randint(0, _NPRIMES - 1)
            p = _PRIMES[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r != old_r:
                start_old = p if old_r == 0 else old_r
                start_new = p if new_r == 0 else new_r
                count[start_old::p] -= 1
                count[start_new::p] += 1
                residues[i] = new_r
                current_L = _find_L(count, limit)
    
    return best_L, best_res

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    random.seed(42)
    
    best_L_overall = 0
    best_residues_overall = None
    
    # Phase 1: Coverage Deficit Greedy
    # Allocate 60% of time to this phase
    phase1_deadline = start_time + (deadline - start_time) * 0.6
    L1, res1 = _coverage_deficit_greedy(phase1_deadline)
    
    if L1 > best_L_overall:
        best_L_overall = L1
        best_residues_overall = res1
    
    # Phase 2: Targeted Refinement
    # Allocate remaining time to refinement
    if best_residues_overall is not None and time.time() < deadline - 10:
        L2, res2 = _targeted_refinement(best_residues_overall, deadline)
        if L2 > best_L_overall:
            best_L_overall = L2
            best_residues_overall = res2
    
    if best_residues_overall is None:
        best_residues_overall = [0] * _NPRIMES
    
    return [int(r) for r in best_residues_overall]
