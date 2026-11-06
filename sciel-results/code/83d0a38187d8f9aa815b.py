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
    """
    Find the largest L such that counts[1..L] >= 2.
    Returns max_check if all are >= 2.
    """
    # We only care about indices 1..max_check
    # counts[0] is unused but allocated.
    below_two = np.where(counts[1:max_check+1] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return int(below_two[0])

def _update_counts(counts, r, p, sign, max_check):
    """
    Update counts array by adding or subtracting 1 at positions r, r+p, ...
    sign: 1 for add, -1 for remove.
    """
    start = r if r > 0 else p
    if start <= max_check:
        counts[start::p] += sign

def _fast_hill_climb(residues, primes, max_check=100000, deadline=None):
    """
    Efficient hill-climbing that focuses on fixing the current bottleneck.
    """
    n_primes = len(primes)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    # Initialize counts
    for r, p in zip(residues, primes):
        _update_counts(counts, r, p, 1, max_check)
    
    current_L = _get_L(counts, max_check)
    best_residues = residues.copy()
    best_L = current_L
    
    # We will run multiple passes. In each pass, we try to improve L.
    # If L doesn't improve in a full pass, we might restart with randomization or stop.
    
    max_passes = 50
    pass_count = 0
    
    while pass_count < max_passes:
        pass_count += 1
        if deadline and time.time() > deadline:
            break
            
        improved_in_pass = False
        # Process primes in a randomized order to escape local minima
        # But small primes are more impactful, so maybe sort by p? 
        # Let's shuffle, but ensure we don't miss small primes too often.
        order = np.random.permutation(n_primes)
        
        for i in order:
            if deadline and time.time() > deadline:
                return best_residues, best_L
                
            p = primes[i]
            current_r = residues[i]
            
            # Determine candidate residues
            # 1. The residue that covers the current bottleneck (current_L + 1)
            bottleneck = current_L + 1
            target_r = bottleneck % p
            
            # 2. The current residue
            # 3. A few random residues
            # 4. For small primes, maybe try all? No, too slow.
            
            candidates = [current_r, target_r]
            
            # Add some random candidates
            if p <= 100:
                # For small primes, sample more aggressively
                rand_candidates = np.random.randint(0, p, size=min(10, p))
            else:
                rand_candidates = np.random.randint(0, p, size=5)
                
            candidates.extend(rand_candidates.tolist())
            # Remove duplicates
            candidates = list(set(candidates))
            
            best_local_r = current_r
            best_local_L = current_L
            
            # Try each candidate
            for r in candidates:
                if r == current_r:
                    continue
                    
                # Apply change
                _update_counts(counts, current_r, p, -1, max_check)
                _update_counts(counts, r, p, 1, max_check)
                
                # Evaluate new L
                new_L = _get_L(counts, max_check)
                
                if new_L > best_local_L:
                    best_local_L = new_L
                    best_local_r = r
                else:
                    # Revert immediately if not better
                    _update_counts(counts, r, p, -1, max_check)
                    _update_counts(counts, current_r, p, 1, max_check)
                    
            # Apply best local change
            if best_local_r != current_r:
                # We need to ensure counts reflect best_local_r
                # If best_local_r was found, counts currently reflect it (if it was the last one tried and not reverted)
                # Wait, the logic above reverts if not better.
                # If best_local_r is current_r, counts reflect current_r.
                # If best_local_r != current_r, counts reflect best_local_r (because we only revert if new_L <= best_local_L, 
                # and best_local_L starts at current_L. If a candidate is better, we don't revert it. 
                # If a later candidate is even better, we revert the previous best? No, we only compare to best_local_L.
                # So if candidate A is better, we keep it. If candidate B is better than A, we revert A? No, we didn't revert A.
                # We applied A. Then we tried B. We applied B (reverting A implicitly? No, we just applied B on top of A? 
                # No, the code does:
                # 1. Remove current_r
                # 2. Add r
                # 3. Check L
                # 4. If not better, Remove r, Add current_r.
                # So if we try A (better), counts have A. 
                # Then we try B. We remove current_r (which is gone? No, current_r was removed before A).
                # Ah, the code removes current_r every time.
                # So:
                # Start: counts have current_r.
                # Try A: Remove current_r, Add A. Counts have A.
                # Try B: Remove current_r (no effect, already removed? No, counts have A, not current_r).
                # This logic is flawed. We must revert to the state before trying the candidate.
                
                # Let's rewrite the inner loop properly.
                pass # Placeholder, will implement correctly below
                
            # Correct Implementation of Inner Loop
            # We must be careful with state management.
            
            # Let's restart the inner loop logic in the actual code below.
            
        if not improved_in_pass:
            # If no improvement in a full pass, we might be stuck.
            # Try a few random restarts or just break.
            # Given time constraints, breaking is safer.
            break
            
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    # Phase 1: Initialization
    # Strategy: Random initialization is often good enough for hill climbing.
    # Or a simple greedy: for each prime, pick r=0.
    # Let's try a "gap filling" greedy first.
    
    residues = np.zeros(n_primes, dtype=np.int32)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    current_L = 0
    
    # Greedy initialization:
    # Process primes in increasing order.
    # For each prime p, pick r such that it covers the current bottleneck (current_L + 1).
    for i, p in enumerate(primes):
        if deadline and time.time() > deadline:
            break
        bottleneck = current_L + 1
        if bottleneck > max_check:
            r = 0
        else:
            r = bottleneck % p
        residues[i] = r
        _update_counts(counts, r, p, 1, max_check)
        # Update current_L
        new_L = _get_L(counts, max_check)
        if new_L > current_L:
            current_L = new_L
    
    best_residues = residues.copy()
    best_L = current_L
    
    # Phase 2: Hill Climbing
    # We need a robust hill climber. The one above had a bug in the inner loop.
    # Let's implement a clean one.
    
    def _robust_hill_climb(residues, primes, max_check, deadline):
        n_primes = len(primes)
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for r, p in zip(residues, primes):
            _update_counts(counts, r, p, 1, max_check)
        
        current_L = _get_L(counts, max_check)
        best_residues = residues.copy()
        best_L = current_L
        
        # Run multiple passes
        for pass_num in range(100):
            if deadline and time.time() > deadline:
                break
                
            improved = False
            # Randomize order
            order = np.random.permutation(n_primes)
            
            for i in order:
                if deadline and time.time() > deadline:
                    return best_residues, best_L
                    
                p = primes[i]
                current_r = residues[i]
                
                # Generate candidates
                # 1. Current
                # 2. Target for current bottleneck
                bottleneck = current_L + 1
                if bottleneck <= max_check:
                    target_r = bottleneck % p
                else:
                    target_r = current_r
                    
                candidates = {current_r, target_r}
                
                # Add randoms
                if p < 50:
                    candidates.update(np.random.randint(0, p, size=10).tolist())
                else:
                    candidates.update(np.random.randint(0, p, size=5).tolist())
                
                best_local_r = current_r
                best_local_L = current_L
                
                # Save original state to revert
                # We will modify counts in place.
                
                for r in candidates:
                    if r == current_r:
                        continue
                        
                    # Remove current
                    _update_counts(counts, current_r, p, -1, max_check)
                    # Add new
                    _update_counts(counts, r, p, 1, max_check)
                    
                    new_L = _get_L(counts, max_check)
                    
                    if new_L > best_local_L:
                        best_local_L = new_L
                        best_local_r = r
                    else:
                        # Revert to current_r
                        _update_counts(counts, r, p, -1, max_check)
                        _update_counts(counts, current_r, p, 1, max_check)
                    
                # Apply best local
                if best_local_r != current_r:
                    # Counts currently reflect best_local_r?
                    # If best_local_r was found, the last successful update left it there.
                    # If it wasn't found, we reverted to current_r.
                    # So if best_local_r != current_r, counts have best_local_r.
                    residues[i] = best_local_r
                    if best_local_L > current_L:
                        current_L = best_local_L
                        improved = True
                    # If best_local_L == current_L, we didn't improve L, but we changed residue.
                    # This might help in future steps?
                    # Actually, if L didn't increase, we shouldn't necessarily accept it unless it helps future steps.
                    # But for simplicity, only accept if L increases.
                    # If we don't accept, we must revert counts to current_r.
                    if best_local_L <= current_L:
                         # Revert
                         _update_counts(counts, best_local_r, p, -1, max_check)
                         _update_counts(counts, current_r, p, 1, max_check)
                    else:
                         # Keep best_local_r
                         best_residues = residues.copy()
                         best_L = current_L
                         
                # If best_local_r == current_r, counts are already correct.
            
            if not improved:
                break
                
        return best_residues, best_L

    # Run the robust hill climber
    best_residues, best_L = _robust_hill_climb(best_residues, primes, max_check, deadline)
    
    # Phase 3: If time remains, try a second round with different random seeds or more passes
    if time.time() < deadline - 5:
         # Restart from best found
         best_residues, best_L = _robust_hill_climb(best_residues, primes, max_check, deadline)
         
    return best_residues.tolist()

# EVOLVE-BLOCK-END
