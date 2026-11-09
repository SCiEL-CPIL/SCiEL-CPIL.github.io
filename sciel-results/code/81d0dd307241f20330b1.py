import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def _compute_score(residues, primes, limit):
    """
    Compute the largest L <= limit such that every integer 1..L 
    satisfies at least two congruences.
    Uses vectorized NumPy operations.
    """
    if limit <= 0:
        return 0
    
    # Create an array of indices 1..limit
    # We need to check coverage for x in 1..limit
    # x % p == residues[i]
    
    # To speed up, we can process in chunks if limit is very large, 
    # but for limit up to 10000, a single vectorized op is fast enough.
    # Let's use a chunk size of 5000 to manage memory and cache efficiency if needed, 
    # but 10000 is small.
    
    max_check = limit
    coverage = np.zeros(max_check, dtype=np.int32)
    
    # We iterate over primes. 
    # For each prime p, the numbers satisfying x % p == r are r, r+p, r+2p, ...
    # We can create a boolean array or just increment the coverage at those indices.
    # Using slicing is efficient.
    
    for i, p in enumerate(primes):
        r = residues[i]
        # Indices in 1..max_check that are congruent to r mod p
        # Note: x starts from 1.
        # If r == 0, x is p, 2p, ...
        # If r != 0, x is r, r+p, ... (provided r <= max_check)
        
        # Generate the slice
        # start = r if r != 0 else p
        # But we need to be careful with the range 1..max_check
        
        # Let's just use modulo on the whole array? 
        # x_arr = np.arange(1, max_check + 1)
        # mask = (x_arr % p == r)
        # coverage[mask] += 1
        # This creates a temporary array of size max_check for each prime.
        # 168 * 10000 = 1.68M operations. This is fast.
        
        # Alternative: direct indexing
        # The values are r, r+p, r+2p...
        # We need values in [1, max_check]
        
        # If r == 0:
        #   values are p, 2p, ...
        # Else:
        #   values are r, r+p, ...
        
        if r == 0:
            start = p
        else:
            start = r
            
        if start <= max_check:
            # indices in 0-based array for values start, start+p, ...
            # value v corresponds to index v-1
            # so index is start-1, start+p-1, ...
            indices = np.arange(start - 1, max_check, p)
            coverage[indices] += 1
            
    # Find the first index where coverage < 2
    # We want the largest L such that coverage[0:L] are all >= 2
    # coverage[0] corresponds to x=1
    
    if max_check == 0:
        return 0
        
    # np.argmax returns the first occurrence of the max value.
    # We want the first occurrence of < 2.
    # If all are >= 2, argmax of (coverage < 2) will be 0 if none are < 2? 
    # No, if none are < 2, the array is all False (0). argmax is 0.
    # We need to check if any are < 2.
    
    gaps = coverage < 2
    if not np.any(gaps):
        return max_check
        
    first_gap_idx = np.argmax(gaps)
    # first_gap_idx is 0-based index in coverage array, which corresponds to x = first_gap_idx + 1
    # The valid range is 1 .. first_gap_idx
    # So L = first_gap_idx
    
    return int(first_gap_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 115 # 5s safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Initial random residues
    random.seed(42)
    current_residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    # Start with a small limit to get a quick score
    current_limit = 1000
    current_score = _compute_score(current_residues, primes, current_limit)
    
    best_residues = current_residues.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.9995
    
    iter_count = 0
    stagnant = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        # Pick a random prime to change
        idx = random.randint(0, n_primes - 1)
        p = primes[idx]
        old_r = current_residues[idx]
        
        # Pick a new random residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        current_residues[idx] = new_r
        
        # Evaluate new score
        # If current_score is high, we might want to check a larger limit occasionally
        # But for speed, we stick to current_limit for the SA loop.
        # We will periodically re-evaluate best_score with a larger limit.
        
        new_score = _compute_score(current_residues, primes, current_limit)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues.copy()
                stagnant = 0
                
                # If we improved, maybe expand the limit slightly to see if we can go further
                # But we must be careful not to slow down the loop too much.
                # Let's keep current_limit fixed for the SA step, 
                # but we can update current_limit if we hit the cap.
                if new_score == current_limit:
                    # We hit the cap, try to expand the window for future checks
                    # But don't do it every time, only if we have time
                    if time.time() < deadline - 5:
                        current_limit = min(current_limit * 2, 10000)
                        # Re-evaluate current score with new limit to see if it's still valid
                        # Actually, if we hit the cap, the score is at least current_limit.
                        # The true score might be higher.
                        # For the SA acceptance, using the capped score is fine.
            else:
                stagnant += 1
        else:
            # Revert
            current_residues[idx] = old_r
            stagnant += 1
            
        # Cool down
        T *= cooling
        
        # If stagnant for too long, restart from best
        if stagnant > 1000:
            current_residues = best_residues.copy()
            current_score = best_score
            stagnant = 0
            T = 5.0 # Reheat slightly
        
        # Periodically check the true score of the best solution with a larger limit
        # to ensure we are tracking the real progress.
        if iter_count % 500 == 0 and time.time() < deadline - 10:
            # Check best_residues with a larger limit
            check_limit = min(best_score + 1000, 10000)
            true_best_score = _compute_score(best_residues, primes, check_limit)
            if true_best_score > best_score:
                best_score = true_best_score
                # Update current_limit if we found a better score
                if best_score == current_limit:
                    current_limit = min(current_limit * 2, 10000)

    # Final evaluation of the best solution found
    # We want the exact score up to 100000
    # But _compute_score with 100000 might be slow if done multiple times.
    # We only do it once at the end.
    final_limit = 100000
    if time.time() < deadline - 5:
        final_score = _compute_score(best_residues, primes, final_limit)
    else:
        final_score = best_score # Fallback if no time
    
    # The task asks for a list of residues
    return list(best_residues)

if __name__ == "__main__":
    # Just for local testing
    res = construct()
    print(len(res))
