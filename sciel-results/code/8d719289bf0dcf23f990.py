"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _get_L(counts, max_check=100000):
    """Find the largest L such that counts[1..L] >= 2."""
    # We assume counts[0] is unused.
    # Find first index i >= 1 where counts[i] < 2
    below_two = np.where(counts[1:max_check+1] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return int(below_two[0])

def _greedy_construct(primes, order, max_check=100000):
    """
    Greedy set-cover heuristic.
    For each prime in 'order', pick residue to cover the first gap.
    """
    counts = np.zeros(max_check + 1, dtype=np.int32)
    residues = np.zeros(len(primes), dtype=np.int32)
    current_L = 0
    
    # Map prime value to index in primes list for easy access
    # primes is sorted, so index is just position.
    
    for p in order:
        # Find first gap > current_L
        if current_L < max_check:
            # Search for first x > current_L where counts[x] < 2
            # Optimization: if current_L is 0, start from 1.
            start_search = current_L + 1
            if start_search > max_check:
                g1 = max_check + 1
            else:
                # Find first index in counts[start_search:] where value < 2
                segment = counts[start_search:]
                below_two = np.where(segment < 2)[0]
                if len(below_two) == 0:
                    g1 = max_check + 1
                else:
                    g1 = start_search + int(below_two[0])
        else:
            g1 = max_check + 1
            
        # If g1 > max_check, we are already covering everything.
        # Pick arbitrary residue, e.g., 0.
        if g1 > max_check:
            r = 0
        else:
            # Optimal residue is g1 % p
            r = g1 % p
        
        # Update counts
        residues[primes.index(p)] = r
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
            
        # Update current_L
        new_L = _get_L(counts, max_check)
        if new_L > current_L:
            current_L = new_L
        else:
            # L should not decrease in greedy add-only phase
            current_L = new_L
            
    return residues, current_L

def _fast_hill_climb(residues, primes, counts, current_L, max_check=100000, deadline=None):
    """
    Fast local search using incremental gap logic.
    """
    n_primes = len(primes)
    improved = True
    passes = 0
    
    while improved and passes < 20:
        passes += 1
        improved = False
        
        for i in range(n_primes):
            if deadline and time.time() > deadline:
                return residues, counts, current_L
                
            p = primes[i]
            current_r = residues[i]
            
            # Try all residues for small p, sample for large
            if p <= 50:
                candidates = range(p)
            else:
                # Try current, and some randoms, and specifically the one that covers the current first gap
                # Find current first gap
                if current_L < max_check:
                    start_search = current_L + 1
                    segment = counts[start_search:]
                    below_two = np.where(segment < 2)[0]
                    if len(below_two) > 0:
                        g1 = start_search + int(below_two[0])
                        target_r = g1 % p
                        if target_r != current_r:
                            candidates = [current_r, target_r] + np.random.randint(0, p, size=5).tolist()
                        else:
                            candidates = [current_r] + np.random.randint(0, p, size=5).tolist()
                    else:
                        candidates = [current_r] + np.random.randint(0, p, size=5).tolist()
                else:
                    candidates = [current_r] + np.random.randint(0, p, size=5).tolist()
                    
            best_r = current_r
            best_L = current_L
            
            for r in candidates:
                if r == current_r:
                    continue
                
                # Apply change
                # Remove old
                old_start = current_r if current_r > 0 else p
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                
                # Add new
                new_start = r if r > 0 else p
                if new_start <= max_check:
                    counts[new_start::p] += 1
                
                # Evaluate new L
                # Check if L decreased
                # If L decreased, it must be at a position where we removed coverage
                # Positions removed: old_start, old_start+p, ...
                # Positions added: new_start, new_start+p, ...
                
                # Simple check: recompute L from scratch? 
                # _get_L is O(N). N=10^5. 168 primes * 50 candidates * 20 passes * O(10^5) is too slow?
                # 168 * 50 * 20 * 10^5 = 1.6 * 10^9. Too slow.
                
                # We need the fast logic.
                # If counts only changed at specific indices, we can check those.
                # But L can drop significantly.
                
                # Heuristic: If p is small, L drop is likely local.
                # Let's just use _get_L but optimize it?
                # _get_L uses np.where which is fast in C.
                # 1.6 * 10^9 operations in C might be ~1-2 seconds? 
                # np.where on 10^5 array is very fast, maybe 10us.
                # 168 * 50 * 20 = 168000 evaluations.
                # 168000 * 10us = 1.68 seconds. This is acceptable!
                
                new_L = _get_L(counts, max_check)
                
                if new_L > best_L:
                    best_L = new_L
                    best_r = r
                else:
                    # Revert
                    counts[new_start::p] -= 1
                    if old_start <= max_check:
                        counts[old_start::p] += 1
                    
            # Apply best
            if best_r != current_r:
                # Re-apply best_r (it was reverted in the loop if not best, or kept if best? 
                # In the loop, if new_L > best_L, we updated best_r but did NOT revert.
                # If new_L <= best_L, we reverted.
                # So if best_r != current_r, the current state in counts corresponds to best_r?
                # No. 
                # If r was best, we left it applied.
                # If r was not best, we reverted.
                # So if best_r != current_r, the counts currently reflect best_r.
                # If best_r == current_r, the counts reflect current_r (because all others were reverted).
                
                if best_r != current_r:
                    residues[i] = best_r
                    current_L = best_L
                    improved = True
                # If best_r == current_r, no change needed.
                
    return residues, counts, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    best_residues = None
    best_score = 0
    
    # Try a few different greedy orders
    orders = [
        primes,  # Increasing
        primes[::-1],  # Decreasing
        list(np.random.permutation(primes))
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
            
        residues, L = _greedy_construct(primes, order, max_check)
        
        # Refine with hill climbing
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for r, p in zip(residues, primes):
            start = r if r > 0 else p
            if start <= max_check:
                counts[start::p] += 1
        
        residues, counts, L = _fast_hill_climb(residues, primes, counts, L, max_check, deadline)
        
        if L > best_score:
            best_score = L
            best_residues = residues.copy()
            
    if best_residues is None:
        # Fallback
        best_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int32)
        
    return best_residues.tolist()
# EVOLVE-BLOCK-END
