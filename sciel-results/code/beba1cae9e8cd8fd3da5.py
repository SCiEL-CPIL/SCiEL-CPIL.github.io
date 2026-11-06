import time
import math

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N + 1)
    
    # Step 1: Greedy determination of residues r_p
    # blocked[x] is True if x is removed due to constraints of primes processed so far
    blocked = [False] * (N + 1)
    residues = {}
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Find residue r that minimizes new blocks
        best_r = 0
        min_new_blocks = N + 1
        
        for r in range(p):
            new_blocks = 0
            # Count how many x == r mod p are not already blocked
            for x in range(r, N + 1, p):
                if not blocked[x]:
                    new_blocks += 1
            if new_blocks < min_new_blocks:
                min_new_blocks = new_blocks
                best_r = r
                if new_blocks == 0:
                    break
        
        residues[p] = best_r
        
        # Update blocked array
        for x in range(best_r, N + 1, p):
            blocked[x] = True
            
    # Step 2: Relaxation
    # active_primes: set of primes currently enforcing constraints
    active_primes = set(primes)
    
    # Current size k
    k = sum(1 for b in blocked if not b)
    
    # We want to remove constraints for primes q > k
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
        # Candidates: x such that x % q == residues[q] and x is currently blocked
        # We need to check if they are blocked by ANY OTHER active prime
        r_q = residues[q]
        new_unblocked_count = 0
        
        for x in range(r_q, N + 1, q):
            if not blocked[x]:
                continue # Already unblocked, shouldn't happen if q was active and x%q==r_q?
                         # Wait, if x is not blocked, it means it's not removed by ANY active prime.
                         # But if q is active and x%q==r_q, x MUST be blocked.
                         # So this check is redundant but safe.
            
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
            # Accept removal
            # We don't actually need to update 'blocked' array for the final count if we just track k
            # But for subsequent iterations, we need to know which elements are unblocked.
            # However, 'blocked' is only used to determine if an element was blocked by 'previous' primes.
            # In the relaxation phase, we are only adding elements back.
            # We can just update k.
            # But wait, if we remove q, and then try to remove p < q,
            # the check for p uses 'blocked' which still has q's effects?
            # No, the check "is_blocked_by_other" iterates over active_primes.
            # If we mark q as inactive, it won't be in the loop.
            # So we just need to update k and active_primes.
            
            active_primes.remove(q)
            k = k_new
        else:
            # Reject removal
            pass
            
    # Final result: list of unblocked elements
    # We need to reconstruct the set based on active_primes and residues
    # Or simply return the indices where blocked is False?
    # No, 'blocked' was updated in Step 1 for ALL primes.
    # In Step 2, we only tracked k.
    # We need to generate the actual list.
    
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
