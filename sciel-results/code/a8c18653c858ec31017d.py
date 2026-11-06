import time
import numpy as np

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    # Initialize residues
    # Random initialization is fine, but maybe start with r_p = p/2?
    residues = np.array([np.random.randint(0, p) for p in primes], dtype=np.int32)
    
    # Compute initial counts
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
            
    # Find first gap
    def find_first_gap(counts):
        below_two = np.where(counts[1:] < 2)[0]
        if len(below_two) == 0:
            return max_check + 1 # No gap in range
        return int(below_two[0]) + 1

    current_L = find_first_gap(counts)
    
    # Optimization Loop
    while time.time() < deadline:
        x_gap = current_L
        
        if x_gap > max_check:
            break
            
        improved = False
        
        # 1. Look for Free Moves (p > x_gap and r_p >= x_gap)
        # This is fast: just check primes > x_gap
        # Since primes are sorted, we can binary search or just iterate from end
        # But n_primes is 168, so linear scan is fine.
        
        # We want to cover x_gap.
        # For a prime p, the required residue is r_req = x_gap % p.
        
        # Strategy A: Free Moves
        # If p > x_gap, r_req = x_gap.
        # Condition for free: current r_p >= x_gap.
        # If so, change r_p to x_gap.
        
        # Strategy B: Safe Moves
        # For any p, r_req = x_gap % p.
        # If r_p == r_req, skip.
        # Check if changing r_p to r_req breaks prefix [1, x_gap-1].
        # Old positions: x_old such that x_old % p == r_p and 1 <= x_old < x_gap.
        # If any counts[x_old] == 2, unsafe.
        
        # Iterate primes. 
        # To maximize progress, maybe try larger primes first for Free moves?
        # Or just any.
        
        # Let's combine:
        # Iterate all primes.
        for i in range(n_primes):
            if time.time() > deadline:
                break
                
            p = primes[i]
            r_old = residues[i]
            r_new = x_gap % p
            
            if r_old == r_new:
                continue
                
            # Check if this move is "Free"
            is_free = False
            if p > x_gap:
                # Only one hit in [1, x_gap-1] possible, at r_old if r_old < x_gap
                if r_old >= x_gap:
                    is_free = True
            else:
                # Multiple hits possible. Not "free" in the simple sense, 
                # but could be safe.
                pass
            
            # Check if Safe
            is_safe = False
            if is_free:
                is_safe = True
            else:
                # Check all x in [1, x_gap-1] where x % p == r_old
                # Generate these indices
                # First such index:
                # r_old is in [0, p-1].
                # If r_old == 0, indices are p, 2p, ...
                # If r_old > 0, indices are r_old, r_old+p, ...
                
                # We need to check counts[x] > 2 for all such x < x_gap.
                
                # Optimization: 
                # If p is large, few indices.
                # If p is small, many indices.
                
                # Let's check.
                if r_old == 0:
                    start_idx = p
                else:
                    start_idx = r_old
                    
                # Iterate x from start_idx to x_gap - 1 step p
                safe = True
                x = start_idx
                while x < x_gap:
                    if counts[x] == 2:
                        safe = False
                        break
                    x += p
                is_safe = safe
            
            if is_safe:
                # Apply change
                # Update counts
                # Remove old coverage
                if r_old == 0:
                    start_idx = p
                else:
                    start_idx = r_old
                counts[start_idx::p] -= 1
                
                # Add new coverage
                if r_new == 0:
                    start_idx_new = p
                else:
                    start_idx_new = r_new
                counts[start_idx_new::p] += 1
                
                residues[i] = r_new
                improved = True
                break # Restart loop with new counts
        
        if not improved:
            # Stuck. 
            # Try a random kick to escape local optimum.
            # Pick a random prime, change residue randomly.
            # This might lower L temporarily, but we track best L?
            # No, we must return the best L found.
            # If we lower L, we lose progress.
            # Better kick: Find a prime p where r_p can be changed to cover x_gap, 
            # even if it breaks the prefix, but hope that the new gap is later?
            # No, if it breaks prefix, new gap is earlier.
            
            # Alternative: Accept that we are at a local optimum.
            # To avoid infinite loop, we can break or do a random perturbation 
            # that is guaranteed to be safe?
            # If no safe move exists, we are stuck.
            
            # Let's try a "2-opt" like move? Too slow.
            
            # Break out to avoid wasting time.
            break

    # Final score calculation
    # The current L is stored in current_L? 
    # We should recompute to be sure.
    final_L = find_first_gap(counts)
    
    return residues.tolist()
