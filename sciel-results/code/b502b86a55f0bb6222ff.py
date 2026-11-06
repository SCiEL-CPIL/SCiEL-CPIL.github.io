import math
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

PRIMES = _primes(N)

def _try_construct(K, start_time=None, timeout=320):
    """
    Attempts to construct an admissible tuple of size at least K.
    Uses a greedy strategy to select missing residue classes for each prime p <= K.
    Returns a list of integers if successful, None otherwise.
    """
    if start_time is None:
        start_time = time.time()
    
    if K <= 0:
        return []
    
    # We only care about primes p <= K
    relevant_primes = [p for p in PRIMES if p <= K]
    
    # If K is larger than N, we can't have distinct integers in [0, N]
    if K > N + 1: 
        return None

    forbidden = [False] * (N + 1)
    
    for p in relevant_primes:
        if time.time() - start_time > timeout:
            return None
            
        best_r = 0
        min_loss = N + 1
        
        # Try each residue
        for r in range(p):
            loss = 0
            # Check how many currently allowed numbers would be forbidden
            for x in range(r, N + 1, p):
                if not forbidden[x]:
                    loss += 1
            if loss < min_loss:
                min_loss = loss
                best_r = r
                
        # Apply the best residue
        for x in range(best_r, N + 1, p):
            forbidden[x] = True
            
    # Count how many numbers are still allowed
    allowed = [x for x in range(N + 1) if not forbidden[x]]
    
    if len(allowed) >= K:
        # Return the first K allowed numbers (or any K)
        # Sorting is not strictly required for validity, but good for determinism
        return allowed[:K]
    else:
        return None

def construct():
    start_time = time.time()
    timeout = 320 # Safety margin for 330s limit
    
    # Binary search for the maximum K
    low = 1
    high = N + 1
    best_tuple = []
    
    # We know K=1 is always possible (e.g., [0])
    # Let's start from a reasonable estimate to save time, but binary search is fast enough.
    
    while low <= high:
        if time.time() - start_time > timeout:
            break
            
        mid = (low + high) // 2
        
        res = _try_construct(mid, start_time, timeout)
        
        if res is not None:
            best_tuple = res
            low = mid + 1
        else:
            high = mid - 1
            
    return best_tuple
