"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    start_time = time.time()
    # Safety margin: finish well before 300s
    deadline = start_time + 280.0
    
    # Start with the full set
    A = set(range(N + 1))
    
    primes = _primes(N)
    
    for i, p in enumerate(primes):
        # Check time
        if time.time() > deadline:
            break
        
        # Only need to satisfy condition for p <= len(A)
        if p > len(A):
            break
        
        # Count residues mod p
        counts = [0] * p
        for x in A:
            counts[x % p] += 1
        
        min_count = min(counts)
        
        # If a residue is already missed, no action needed
        if min_count == 0:
            continue
        
        # All residues are present. We must remove one class.
        # Identify candidates (classes with minimum count)
        candidates = [r for r in range(p) if counts[r] == min_count]
        
        r_to_remove = candidates[0]
        
        # Lookahead: if there are multiple candidates, pick the one that
        # maximizes the minimum count for the next prime, to minimize future loss.
        if len(candidates) > 1 and i + 1 < len(primes):
            p_next = primes[i + 1]
            
            # Estimate size after removal
            new_size_est = len(A) - min_count
            
            # If p_next > new_size_est, we won't process p_next, so cost is 0 (inf score)
            if p_next > new_size_est:
                # All candidates are equally good regarding future constraints
                # We can just pick the first one or random
                r_to_remove = candidates[0]
            else:
                best_r = candidates[0]
                best_score = -1
                
                for r in candidates:
                    next_counts = [0] * p_next
                    for x in A:
                        if x % p != r:
                            next_counts[x % p_next] += 1
                            
                    # We want to maximize the minimum count of the next step
                    score = min(next_counts)
                    
                    if score > best_score:
                        best_score = score
                        best_r = r
                        
                r_to_remove = best_r
        
        # Remove the chosen residue class
        # Optimization: use set comprehension
        A = {x for x in A if x % p != r_to_remove}
        
        # If A becomes empty, break (though unlikely to be optimal)
        if not A:
            break

    return list(A)
# EVOLVE-BLOCK-END
