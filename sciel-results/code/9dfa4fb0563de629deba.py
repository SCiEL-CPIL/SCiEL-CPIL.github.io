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

def construct():
    # Start with all integers in [0, N]
    S = list(range(N + 1))
    primes = _primes(N)
    
    i = 0
    while i < len(primes):
        p = primes[i]
        if p > len(S):
            break
            
        # Count residues mod p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
            
        min_count = min(counts)
        
        # Identify candidates: residues with count == min_count
        # If we want to be aggressive, we can look at all residues,
        # but removing non-min is costly. Let's stick to min_count first.
        candidates = [r for r in range(p) if counts[r] == min_count]
        
        if len(candidates) == 1:
            r = candidates[0]
        else:
            # Lookahead: check next prime q
            if i + 1 < len(primes):
                q = primes[i + 1]
                if q <= len(S): # Only lookahead if q is relevant
                    best_r = candidates[0]
                    best_m = float('inf')
                    
                    # Pre-compute S without each candidate?
                    # That's expensive. 
                    # Instead, for each candidate r, we can calculate the counts mod q 
                    # of the set S \ {x % p == r}.
                    
                    # Optimization: 
                    # counts_q[r_q] = total_count_q[r_q] - removed_count_q[r_q]
                    # where removed_count_q[r_q] is the number of x in S such that x%p==r and x%q==r_q.
                    
                    # Let's compute a joint distribution table for p and q.
                    # joint[r][rq] = count of x in S such that x%p==r and x%q==rq
                    joint = [[0]*q for _ in range(p)]
                    for x in S:
                        joint[x % p][x % q] += 1
                    
                    # Current counts for q
                    counts_q = [0] * q
                    for rq in range(q):
                        counts_q[rq] = sum(joint[r][rq] for r in range(p))
                        # Actually, just summing joint column is fine, or we can just use S counts.
                        # We already have S, so we can just count S % q.
                    
                    # Recalculate counts_q directly for simplicity, it's O(N)
                    counts_q = [0] * q
                    for x in S:
                        counts_q[x % q] += 1

                    for r in candidates:
                        # Calculate new counts for q if we remove residue r mod p
                        new_counts_q = counts_q.copy()
                        for rq in range(q):
                            new_counts_q[rq] -= joint[r][rq]
                        
                        m = min(new_counts_q)
                        if m < best_m:
                            best_m = m
                            best_r = r
                    r = best_r
                else:
                    r = candidates[0]
            else:
                r = candidates[0]

        # Remove all elements congruent to r mod p
        S = [x for x in S if x % p != r]
        # Do not increment i, re-check p with new S
        if not S:
            break
        i += 1
                
    return S
