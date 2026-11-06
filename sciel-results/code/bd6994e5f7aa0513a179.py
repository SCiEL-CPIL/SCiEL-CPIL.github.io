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
    time_limit = 115.0  # Safety margin before 150s kill (leaves 35s buffer)
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We will search for L up to a reasonable max. 
    MAX_CHECK = 20000
    
    def calculate_score(residues):
        """
        Calculate the largest L such that every integer 1..L satisfies at least two congruences.
        """
        counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_CHECK:
                idx = np.arange(start, MAX_CHECK + 1, p)
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

    def is_valid_move(residues, counts, prime_idx, new_r, x_fail):
        """
        Check if changing residues[prime_idx] to new_r keeps the prefix [1, x_fail-1] valid.
        """
        p = primes[prime_idx]
        old_r = residues[prime_idx]
        
        if old_r == new_r:
            return True
            
        # Check counts for these y's where y % p == old_r
        if old_r == 0:
            start_y = p
        else:
            start_y = old_r
            
        # We only care about y < x_fail
        if start_y >= x_fail:
            return True # No affected y in the prefix
            
        # Check counts for these y's
        for y in range(start_y, x_fail, p):
            if counts[y] == 2:
                return False
                
        return True

    def greedy_construct():
        """
        Greedy set-cover heuristic: process primes in increasing order.
        For each prime, select the residue that maximizes the number of integers
        in the current critical interval [1, current_L] that are covered by exactly one congruence.
        """
        residues = [0] * n_primes
        # Start with a reasonable L estimate
        current_L = 1
        
        # We'll maintain counts incrementally? No, just recompute for simplicity in greedy phase 
        # as n_primes is small (168) and MAX_CHECK is moderate.
        
        for i, p in enumerate(primes):
            best_r = 0
            best_gain = -1
            
            # Try all residues for this prime
            for r in range(p):
                residues[i] = r
                # Calculate score
                score = calculate_score(residues)
                if score > best_gain:
                    best_gain = score
                    best_r = r
                # Early stop if we reached MAX_CHECK
                if best_gain == MAX_CHECK:
                    break
            
            residues[i] = best_r
            current_L = best_gain
            
            # Check time
            if time.time() - start_time > time_limit - 5:
                break
        
        return residues, current_L

    best_score = 0
    best_residues = None

    # Phase 1: Greedy Construction
    residues, score = greedy_construct()
    if score > best_score:
        best_score = score
        best_residues = residues[:]
    
    if time.time() - start_time > time_limit - 10:
        return [int(r) for r in best_residues] if best_residues else [0]*n_primes

    # Phase 2: Local Search / Hill Climbing
    # We will do a few restarts or just polish the greedy result.
    
    def local_search(residues, current_score, time_budget):
        max_no_improve = 0
        no_improve_limit = 50
        
        while time.time() - start_time < time_budget:
            if max_no_improve > no_improve_limit:
                break
            
            # Find first failure
            counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
            for i, p in enumerate(primes):
                r = residues[i]
                if r == 0:
                    start = p
                else:
                    start = r
                if start <= MAX_CHECK:
                    idx = np.arange(start, MAX_CHECK + 1, p)
                    counts[idx] += 1
                    
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
            
            # Try each prime, preferring smaller primes first (they affect more numbers)
            prime_order = sorted(range(n_primes), key=lambda i: primes[i])
            
            for i in prime_order:
                p = primes[i]
                new_r = int(x_fail % p)
                if new_r == residues[i]:
                    continue
                
                # Check if this move is valid (keeps prefix [1, x_fail-1] intact)
                if is_valid_move(residues, counts, i, new_r, x_fail):
                    # Apply move
                    old_r = residues[i]
                    residues[i] = new_r
                    # Recalculate score
                    new_score = calculate_score(residues)
                    
                    if new_score > current_score:
                        current_score = new_score
                        move_made = True
                        improved = True
                        max_no_improve = 0
                        break # Restart the while loop to find new first failure
                    elif new_score == current_score:
                        # Accept neutral moves to escape local minima
                        if max_no_improve > 5:
                            move_made = True
                            max_no_improve += 1
                            break
                        else:
                            # Revert
                            residues[i] = old_r
                    else:
                        # Revert
                        residues[i] = old_r
                        
            if not move_made:
                # Stuck. Try random perturbation
                escaped = False
                for _ in range(20):
                    if time.time() - start_time > time_budget:
                        break
                    i = int(np.random.randint(0, n_primes))
                    p = primes[i]
                    old_r = residues[i]
                    new_r = int(np.random.randint(0, p))
                    if new_r == old_r:
                        continue
                    residues[i] = new_r
                    new_score = calculate_score(residues)
                    if new_score > current_score:
                        current_score = new_score
                        escaped = True
                        max_no_improve = 0
                        break
                    elif new_score == current_score and max_no_improve > 10:
                        # Accept neutral move
                        escaped = True
                        max_no_improve += 1
                        break
                    else:
                        # Revert
                        residues[i] = old_r
                        
                if not escaped:
                    max_no_improve += 1
                    if max_no_improve > no_improve_limit:
                        break # Restart this search instance
        
        return residues, current_score

    # Run local search on the greedy result
    residues, score = local_search(residues[:], best_score, time_limit - 5)
    if score > best_score:
        best_score = score
        best_residues = residues[:]
    
    # If we have time, try one more restart with random init + local search
    if time.time() - start_time < time_limit - 20:
        for restart in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Random initialization
            residues = [int(np.random.randint(0, p)) for p in primes]
            current_score = calculate_score(residues)
            
            # Local search
            residues, score = local_search(residues, current_score, time_limit - 5)
            
            if score > best_score:
                best_score = score
                best_residues = residues[:]
    
    # Return the best residues found
    if best_residues is None:
        # Fallback
        return [int(((i + 2) // 2) % p) for i, p in enumerate(primes)]
        
    return [int(r) for r in best_residues]
