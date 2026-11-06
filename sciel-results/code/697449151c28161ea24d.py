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
    time_limit = 120.0  # Safety margin before 150s kill
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We will search for L up to a reasonable max. 
    # The task says score is L. 
    # Let's cap the check at 5000 to be safe but efficient.
    MAX_CHECK = 5000
    
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
            
        # If all are >= 2, return MAX_CHECK
        # np.argmax returns 0 if condition is never true? No, it returns 0 if the first element is True.
        # We want the first index where counts < 2.
        # If no such index, return MAX_CHECK.
        
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
        We know x_fail is the first failure in the current state.
        We want to ensure that for all y < x_fail, the new count is >= 2.
        The count of y changes only if y is congruent to old_r or new_r mod p.
        """
        p = primes[prime_idx]
        old_r = residues[prime_idx]
        
        if old_r == new_r:
            return True
            
        # We need to check y in [1, x_fail - 1]
        # y is affected if y % p == old_r or y % p == new_r
        
        # Optimization: 
        # The count decreases by 1 for y where y % p == old_r and y % p != new_r.
        # The count increases by 1 for y where y % p == new_r and y % p != old_r.
        # The count is unchanged otherwise.
        
        # Failure condition:
        # 1. y % p == old_r and y % p != new_r and counts[y] == 2.
        #    (Count drops from 2 to 1)
        
        # We need to iterate through y < x_fail that are congruent to old_r mod p.
        # There are at most x_fail / p such y's.
        
        # Find all y in [1, x_fail-1] such that y % p == old_r
        # y = k*p + old_r
        # k*p + old_r >= 1 => k >= (1 - old_r) / p
        # k*p + old_r < x_fail => k < (x_fail - old_r) / p
        
        # Let's just loop, it's small.
        # But we can be smarter.
        
        # If old_r == 0, y are p, 2p, ...
        # If old_r > 0, y are old_r, old_r+p, ...
        
        # We check if any such y has counts[y] == 2.
        # Note: if y % p == old_r AND y % p == new_r, then old_r == new_r mod p. 
        # Since 0 <= r < p, this implies old_r == new_r. We handled that above.
        # So the sets are disjoint.
        
        # So we just need to check if any y in [1, x_fail-1] with y % p == old_r has counts[y] == 2.
        
        # Generate candidates
        if old_r == 0:
            start_y = p
        else:
            start_y = old_r
            
        # We only care about y < x_fail
        if start_y >= x_fail:
            return True # No affected y in the prefix
            
        # Check counts for these y's
        # y = start_y, start_y+p, ... < x_fail
        for y in range(start_y, x_fail, p):
            if counts[y] == 2:
                return False
                
        return True

    best_score = 0
    best_residues = None
    
    # Initial setup: random restarts
    num_restarts = 50
    restart_time_limit = time_limit / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Initialize residues
        # Random initialization
        residues = [int(np.random.randint(0, p)) for p in primes]
        
        # Calculate initial score
        current_score = calculate_score(residues)
        
        # Local search
        # We maintain counts array to speed up checks? 
        # calculate_score recalculates from scratch. It's O(MAX_CHECK * sum 1/p).
        # For MAX_CHECK=5000, this is ~5000 * 3 = 15000 ops. Very fast.
        # So we can just recalculate score after each move.
        
        while time.time() - start_time < time_limit - 5:
            # Find first failure
            # We can reuse the logic from calculate_score to get the failure index
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
                # Maximized for this restart
                if current_score < MAX_CHECK:
                    current_score = MAX_CHECK
                break
                
            first_fail_idx = np.argmax(fail_mask[1:]) + 1
            x_fail = first_fail_idx
            
            # Try to fix x_fail
            move_made = False
            # Try each prime
            # Sort primes by size? Smaller primes affect more numbers, so they are riskier but might be more useful?
            # Let's just iterate.
            for i, p in enumerate(primes):
                new_r = int(x_fail % p)
                if new_r == residues[i]:
                    continue
                
                # Check if this move is valid (keeps prefix [1, x_fail-1] intact)
                if is_valid_move(residues, counts, i, new_r, x_fail):
                    # Apply move
                    residues[i] = new_r
                    # Recalculate score
                    new_score = calculate_score(residues)
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                        move_made = True
                        break # Restart the while loop to find new first failure
                    else:
                        # Score didn't improve? 
                        # If we fixed x_fail, score should improve unless x_fail was not the bottleneck?
                        # If x_fail was count 1, it becomes 2. Score should increase by at least 1.
                        # If x_fail was count 0, it becomes 1. Score stays same.
                        # So if new_score == current_score, we are stuck with count 0.
                        # We should not accept this move as it doesn't help.
                        pass
                        
            if not move_made:
                # Stuck. Try random perturbation?
                # Or just break and restart.
                # Let's try a few random moves to escape local minima
                escaped = False
                for _ in range(10):
                    if time.time() - start_time > time_limit - 5:
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
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                        escaped = True
                        break
                    else:
                        # Revert
                        residues[i] = old_r
                        
                if not escaped:
                    break # Restart this search instance
        
        # End of local search for this restart

    # Return the best residues found
    if best_residues is None:
        # Fallback to seed-like behavior if search failed
        return [int(((i + 2) // 2) % p) for i, p in enumerate(primes)]
        
    return [int(r) for r in best_residues]
