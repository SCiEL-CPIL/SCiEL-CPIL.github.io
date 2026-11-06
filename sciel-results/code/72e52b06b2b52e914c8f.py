import time
import math
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _greedy_pass(primes, blocked, residues, deadline, random_seed=None):
    """
    Greedy pass to determine residues for each prime.
    If random_seed is not None, uses randomized tie-breaking.
    """
    rng = random.Random(random_seed) if random_seed is not None else None
    
    for p in primes:
        if time.time() > deadline:
            break
        
        # Count new blocks for each residue
        new_blocks_counts = []
        for r in range(p):
            new_blocks = 0
            for x in range(r, N + 1, p):
                if not blocked[x]:
                    new_blocks += 1
            new_blocks_counts.append(new_blocks)
        
        # Find minimum
        min_blocks = min(new_blocks_counts)
        
        # Find all residues that achieve the minimum
        best_residues = [r for r, c in enumerate(new_blocks_counts) if c == min_blocks]
        
        # Choose among best residues
        if len(best_residues) == 1:
            best_r = best_residues[0]
        else:
            if rng is not None:
                best_r = rng.choice(best_residues)
            else:
                best_r = best_residues[0]
        
        residues[p] = best_r
        
        # Update blocked array
        for x in range(best_r, N + 1, p):
            blocked[x] = True
    
    return residues

def _relaxation(primes, blocked, residues, active_primes, k, deadline):
    """
    Relaxation phase: try to remove constraints for primes q > k.
    Returns updated active_primes and k.
    """
    # Iterate primes in descending order
    for q in reversed(primes):
        if time.time() > deadline:
            break
            
        if q not in active_primes:
            continue
            
        if q <= k:
            # Cannot remove q if q <= k, because we need to constrain all p <= k
            continue
            
        # Try removing q
        r_q = residues[q]
        new_unblocked_count = 0
        
        for x in range(r_q, N + 1, q):
            if not blocked[x]:
                continue
            
            # Check if x is blocked by any other active prime
            is_blocked_by_other = False
            for p in active_primes:
                if p == q:
                    continue
                if x % p == residues[p]:
                    is_blocked_by_other = True
                    break
            
            if not is_blocked_by_other:
                new_unblocked_count += 1
        
        k_new = k + new_unblocked_count
        
        if k_new < q:
            active_primes.remove(q)
            k = k_new
    
    return active_primes, k

def _compute_result(active_primes, residues):
    """Compute the final result list."""
    result = []
    for x in range(N + 1):
        is_admissible = True
        for p in active_primes:
            if x % p == residues[p]:
                is_admissible = False
                break
        if is_admissible:
            result.append(x)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N + 1)
    
    # Multiple restarts with different random seeds
    best_result = []
    best_size = 0
    
    # First, do a deterministic pass
    blocked = [False] * (N + 1)
    residues = {}
    _greedy_pass(primes, blocked, residues, deadline, random_seed=None)
    
    active_primes = set(primes)
    k = sum(1 for b in blocked if not b)
    active_primes, k = _relaxation(primes, blocked, residues, active_primes, k, deadline)
    
    result = _compute_result(active_primes, residues)
    best_result = result
    best_size = len(result)
    
    # Now try multiple randomized restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Each restart gets a fraction of the remaining time
        remaining_time = deadline - time.time()
        if remaining_time < 5:  # Need at least 5 seconds for a restart
            break
        
        # Use a portion of remaining time for this restart
        restart_deadline = time.time() + min(remaining_time * 0.8, 20)  # Cap at 20s per restart
        
        blocked = [False] * (N + 1)
        residues = {}
        _greedy_pass(primes, blocked, residues, restart_deadline, random_seed=restart)
        
        active_primes = set(primes)
        k = sum(1 for b in blocked if not b)
        active_primes, k = _relaxation(primes, blocked, residues, active_primes, k, restart_deadline)
        
        result = _compute_result(active_primes, residues)
        if len(result) > best_size:
            best_result = result
            best_size = len(result)
    
    return best_result
