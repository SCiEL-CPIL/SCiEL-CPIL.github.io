import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    primes = _get_primes(1000)
    n_primes = len(primes)

    # We will search for L up to a reasonable max.
    MAX_CHECK = 20000

    # Precompute indices for each prime and residue
    indices_cache = [None] * n_primes
    for i, p in enumerate(primes):
        indices_cache[i] = [None] * p
        for r in range(p):
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_CHECK:
                indices_cache[i][r] = np.arange(start, MAX_CHECK + 1, p, dtype=np.int32)
            else:
                indices_cache[i][r] = np.array([], dtype=np.int32)

    def calculate_score(residues):
        """
        Calculate the largest L such that every integer 1..L satisfies at least two congruences.
        """
        counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            idx = indices_cache[i][r]
            if len(idx) > 0:
                counts[idx] += 1

        # Find the first index where count < 2
        # We care about 1..L
        if MAX_CHECK == 0:
            return 0

        # Create a boolean array where counts < 2
        fail_mask = counts < 2
        # We only care about indices 1..MAX_CHECK
        # If fail_mask[1:] has any True, find the first one.
        if np.any(fail_mask[1:]):
            first_fail = np.argmax(fail_mask[1:]) + 1
            return first_fail - 1
        else:
            return MAX_CHECK

    def local_search(residues, current_score, time_budget):
        max_no_improve = 0
        no_improve_limit = 80
        
        # Precompute counts once at start
        counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            idx = indices_cache[i][r]
            if len(idx) > 0:
                counts[idx] += 1
        
        # Precompute prime order: smaller primes first (they affect more numbers)
        prime_order = sorted(range(n_primes), key=lambda i: primes[i])
        
        while time.time() - start_time < time_budget:
            if max_no_improve > no_improve_limit:
                break
            
            # Find first failure using current counts
            fail_mask = counts < 2
            if not np.any(fail_mask[1:]):
                if current_score < MAX_CHECK:
                    current_score = MAX_CHECK
                break
                
            first_fail_idx = np.argmax(fail_mask[1:]) + 1
            x_fail = first_fail_idx
            
            # Try to fix x_fail - try multiple candidate moves
            move_made = False
            improved = False
            
            for i in prime_order:
                if time.time() - start_time > time_budget:
                    break
                p = primes[i]
                new_r = int(x_fail % p)
                if new_r == residues[i]:
                    continue
                
                # Check if this move is valid (keeps prefix [1, x_fail-1] intact)
                old_r = residues[i]
                old_indices = indices_cache[i][old_r]
                new_indices = indices_cache[i][new_r]
                
                # Check if removing old coverage breaks any x < x_fail
                if len(old_indices) > 0:
                    cutoff_idx = np.searchsorted(old_indices, x_fail, side='left')
                    if cutoff_idx > 0:
                        affected = old_indices[:cutoff_idx]
                        if np.any(counts[affected] == 2):
                            continue
                
                # Apply move incrementally
                if len(old_indices) > 0:
                    counts[old_indices] -= 1
                if len(new_indices) > 0:
                    counts[new_indices] += 1
                
                # Recalculate score from updated counts
                fail_mask_new = counts < 2
                if np.any(fail_mask_new[1:]):
                    new_score = np.argmax(fail_mask_new[1:]) + 1 - 1
                else:
                    new_score = MAX_CHECK
                
                if new_score > current_score:
                    current_score = new_score
                    residues[i] = new_r
                    move_made = True
                    improved = True
                    max_no_improve = 0
                    break
                else:
                    # Revert move
                    if len(old_indices) > 0:
                        counts[old_indices] += 1
                    if len(new_indices) > 0:
                        counts[new_indices] -= 1
                    residues[i] = old_r
                    
            if not move_made:
                # Stuck. Try random perturbation
                escaped = False
                for _ in range(30):
                    if time.time() - start_time > time_budget:
                        break
                    i = int(np.random.randint(0, n_primes))
                    p = primes[i]
                    old_r = residues[i]
                    new_r = int(np.random.randint(0, p))
                    if new_r == old_r:
                        continue
                    
                    old_indices = indices_cache[i][old_r]
                    new_indices = indices_cache[i][new_r]
                    
                    # Apply move incrementally
                    if len(old_indices) > 0:
                        counts[old_indices] -= 1
                    if len(new_indices) > 0:
                        counts[new_indices] += 1
                    
                    # Recalculate score from updated counts
                    fail_mask_new = counts < 2
                    if np.any(fail_mask_new[1:]):
                        new_score = np.argmax(fail_mask_new[1:]) + 1 - 1
                    else:
                        new_score = MAX_CHECK
                    
                    if new_score > current_score:
                        current_score = new_score
                        residues[i] = new_r
                        escaped = True
                        max_no_improve = 0
                        break
                    elif new_score == current_score and max_no_improve > 15:
                        # Accept neutral move to escape local minima
                        residues[i] = new_r
                        escaped = True
                        max_no_improve += 1
                        break
                    else:
                        # Revert move
                        if len(old_indices) > 0:
                            counts[old_indices] += 1
                        if len(new_indices) > 0:
                            counts[new_indices] -= 1
                        residues[i] = old_r
                        
                if not escaped:
                    max_no_improve += 1
                    if max_no_improve > no_improve_limit:
                        break
        
        return residues, current_score

    def greedy_construct():
        """
        Greedy construction: process primes in order, for each prime choose the residue
        that maximizes the minimum coverage in the prefix.
        """
        residues = [0] * n_primes
        counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
        
        # Process primes in increasing order
        for i in range(n_primes):
            p = primes[i]
            best_r = 0
            best_min_coverage = -1
            
            for r in range(p):
                idx = indices_cache[i][r]
                if len(idx) == 0:
                    continue
                
                # Tentatively apply
                counts[idx] += 1
                
                # Find the minimum count in 1..MAX_CHECK
                min_cov = int(np.min(counts[1:]))
                
                if min_cov > best_min_coverage:
                    best_min_coverage = min_cov
                    best_r = r
                
                # Revert
                counts[idx] -= 1
            
            residues[i] = best_r
            idx = indices_cache[i][best_r]
            if len(idx) > 0:
                counts[idx] += 1
        
        return residues, calculate_score(residues)

    # Start with greedy construction
    residues, best_score = greedy_construct()
    best_residues = residues[:]
    
    # Run local search
    residues, score = local_search(residues[:], best_score, time_limit - 15)
    if score > best_score:
        best_score = score
        best_residues = residues[:]
    
    # Try a few random restarts
    if time.time() - start_time < time_limit - 25:
        for restart in range(4):
            if time.time() - start_time > time_limit - 15:
                break
            
            # Random initialization with bias toward covering early numbers
            residues = [0] * n_primes
            # For small primes, try to cover 1..100 well
            for i in range(min(10, n_primes)):
                p = primes[i]
                # Choose residue that covers the most numbers in 1..100
                best_r = 0
                best_count = -1
                for r in range(p):
                    # Count how many x in 1..100 satisfy x % p == r
                    c = 0
                    for x in range(1, 101):
                        if x % p == r:
                            c += 1
                    if c > best_count:
                        best_count = c
                        best_r = r
                residues[i] = best_r
            
            current_score = calculate_score(residues)
            
            # Local search
            residues, score = local_search(residues, current_score, time_limit - 10)
            
            if score > best_score:
                best_score = score
                best_residues = residues[:]
    
    # Final polish: one more local search pass on the best solution
    if time.time() - start_time < time_limit - 5:
        residues, score = local_search(best_residues[:], best_score, time_limit - 3)
        if score > best_score:
            best_score = score
            best_residues = residues[:]
    
    # Return the best residues found
    if best_residues is None:
        # Fallback
        return [int(((i + 2) // 2) % p) for i, p in enumerate(primes)]
        
    return [int(r) for r in best_residues]
