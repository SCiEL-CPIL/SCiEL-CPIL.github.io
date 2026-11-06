import numpy as np

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_forbidden(x, residues, primes, p_to_check):
    """Check if x is forbidden by any prime in primes except p_to_check."""
    for q in primes:
        if q == p_to_check:
            continue
        if x % q == residues[q]:
            return True
    return False

def construct():
    # 1. Initialize with Greedy
    S_init = list(range(N + 1))
    primes = _primes(N)
    residues = {}
    
    i = 0
    while i < len(primes):
        p = primes[i]
        if p > len(S_init):
            break
        counts = [0] * p
        for x in S_init:
            counts[x % p] += 1
        min_count = min(counts)
        if min_count == 0:
            i += 1
        else:
            r = counts.index(min_count)
            residues[p] = r
            S_init = [x for x in S_init if x % p != r]
            if not S_init:
                break
                
    # Build initial state
    # in_S[x] is True if x is in the set
    in_S = [False] * (N + 1)
    for x in S_init:
        in_S[x] = True
        
    current_size = len(S_init)
    current_residues = residues.copy()
    
    # 2. Local Search
    # We will try to improve by swapping residues for primes
    # To make it efficient, we only consider primes p <= current_size
    # And we only swap if it increases the size
    
    improved = True
    while improved:
        improved = False
        # Sort primes by size? Or just iterate?
        # Iterate through primes that are active (p <= current_size)
        active_primes = [p for p in primes if p <= current_size]
        
        for p in active_primes:
            r_old = current_residues.get(p)
            if r_old is None:
                # Prime wasn't assigned? Should not happen if p <= size
                continue
                
            best_gain = 0
            best_r_new = None
            
            # Try all other residues
            for r_new in range(p):
                if r_new == r_old:
                    continue
                
                # Calculate change in size
                # 1. Elements x == r_old mod p might enter S
                # 2. Elements x == r_new mod p might leave S
                
                gain = 0
                # Check elements that were forbidden by p (r_old)
                # They are in S if they are NOT forbidden by any other prime
                for x in range(r_old, N + 1, p):
                    # x was not in S before (because forbidden by p)
                    # Check if x is forbidden by any OTHER prime
                    allowed = True
                    for q in active_primes:
                        if q == p:
                            continue
                        if q in current_residues:
                            if x % q == current_residues[q]:
                                allowed = False
                                break
                    if allowed:
                        gain += 1
                        # Note: We must also ensure that x is NOT forbidden by r_new of p?
                        # No, x is r_old mod p. r_new != r_old. So x is not forbidden by new p.
                        # So if allowed by others, it enters.
                
                # Check elements that will be forbidden by p (r_new)
                # They are in S if they are NOT forbidden by any other prime
                # If they are in S now, they will leave.
                for x in range(r_new, N + 1, p):
                    # x was potentially in S
                    if in_S[x]:
                        # Check if x is forbidden by any OTHER prime
                        # If not, it means it was in S only because it wasn't forbidden by p?
                        # No, if it's in S, it's not forbidden by ANY prime.
                        # So if we forbid it by p, it leaves.
                        # We don't need to check other primes because in_S[x] is True.
                        gain -= 1
                        # Wait, if in_S[x] is True, it is not forbidden by any other prime.
                        # So it will definitely leave.
                        # If in_S[x] is False, it is forbidden by some other prime, so it stays out.
                        # So we just subtract 1 for each x in r_new mod p that is currently in S.
                        
                if gain > best_gain:
                    best_gain = gain
                    best_r_new = r_new
            
            if best_gain > 0:
                # Accept swap
                current_residues[p] = best_r_new
                # Update in_S
                # Remove r_old elements that are now allowed
                for x in range(r_old, N + 1, p):
                    # Check if allowed by others
                    allowed = True
                    for q in active_primes:
                        if q == p:
                            continue
                        if q in current_residues:
                            if x % q == current_residues[q]:
                                allowed = False
                                break
                    if allowed:
                        in_S[x] = True
                        current_size += 1
                        improved = True
                            
                # Add r_new elements to forbidden
                for x in range(best_r_new, N + 1, p):
                    if in_S[x]:
                        in_S[x] = False
                        current_size -= 1
                        improved = True
                
                # Restart loop as size changed
                break
                
    # Construct final list
    final_S = [x for x in range(N + 1) if in_S[x]]
    return final_S
