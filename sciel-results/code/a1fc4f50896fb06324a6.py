import sys
import time

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
    deadline = start_time + 280  # Safety margin
    
    primes = _primes(N)
    
    # 1. Determine best residue for each prime
    # best_res[p] is the residue r in [0, p-1] that appears most frequently in [0, N]
    # If tie, pick the smallest r.
    best_res = {}
    for p in primes:
        counts = [0] * p
        # Count residues in 0..N
        # 0..N has N+1 elements
        # Residue r appears floor((N-r)/p) + 1 times if r <= N, else 0
        # Simpler: iterate
        # But N is small, so iterating is fine.
        # Even faster:
        # Number of elements is N+1
        # Base count = (N+1) // p
        # Remainder = (N+1) % p
        # The first 'Remainder' residues (0, 1, ..., Remainder-1) have count Base+1
        # The rest have count Base
        base = (N + 1) // p
        rem = (N + 1) % p
        max_count = base
        best_r = 0
        for r in range(p):
            c = base + (1 if r < rem else 0)
            if c > max_count:
                max_count = c
                best_r = r
        best_res[p] = best_r
        
    # 2. For each x in [0, N], find min_bad[x]
    # min_bad[x] is the smallest prime p such that x % p == best_res[p]
    # If no such prime, min_bad[x] = infinity
    INF = float('inf')
    min_bad = [INF] * (N + 1)
    
    # Optimization: only check primes that could be small
    # We only care about min_bad up to some limit, but let's compute for all primes.
    # If x % 2 == best_res[2], min_bad is at most 2.
    
    for x in range(N + 1):
        min_val = INF
        for p in primes:
            if p > min_val:
                break
            if x % p == best_res[p]:
                min_val = p
                if min_val == 2:
                    break
        min_bad[x] = min_val
        
    # 3. Build histogram of min_bad values
    # We need to answer: for a given k, how many x have min_bad[x] > k?
    # Let H[m] be count of x with min_bad[x] == m.
    # m can be a prime or INF.
    # We only need to sum H[m] for m > k.
    # Since k <= N, we can just create an array of size N+2.
    # H[i] for i in 0..N. H[INF] is separate or mapped to N+1.
    
    H = [0] * (N + 2)
    for x in range(N + 1):
        mb = min_bad[x]
        if mb > N:
            H[N + 1] += 1
        else:
            H[mb] += 1
            
    # Suffix sums
    # S[i] = sum(H[j] for j >= i)
    S = [0] * (N + 3)
    for i in range(N + 1, -1, -1):
        S[i] = S[i + 1] + H[i]
        
    # 4. Find largest k such that S[k+1] >= k
    # Note: S[k+1] is count of x with min_bad[x] >= k+1, i.e., > k.
    best_k = 0
    for k in range(N, 0, -1):
        if S[k + 1] >= k:
            best_k = k
            break
            
    if best_k == 0:
        # Fallback: just return [0] or something valid
        # k=1: primes p<=1: none. Condition vacuously true.
        # Any single integer is admissible.
        return [0]
        
    # 5. Construct the tuple
    # We need best_k distinct integers from [0, N] such that min_bad[x] > best_k
    result = []
    for x in range(N + 1):
        if min_bad[x] > best_k:
            result.append(x)
            if len(result) == best_k:
                break
                
    return result

# EVOLVE-BLOCK-END
