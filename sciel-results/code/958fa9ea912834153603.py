"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def _sigma_sieve(n):
    """Compute sigma(k) for all k from 1 to n using a sieve."""
    sigma = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            sigma[j] += i
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Start with a moderate limit, increase if time permits
    # First pass: up to 200000 (should be fast enough)
    N = 200000
    sigma = _sigma_sieve(N)
    
    # Build dict: M -> list of k values
    from collections import defaultdict
    m_to_ks = defaultdict(list)
    
    for k in range(1, N + 1):
        m = k * sigma[k]
        if m <= 10**18 and k <= math.isqrt(m):
            m_to_ks[m].append(k)
    
    # Find best collision
    best_m = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    for m, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_m = m
            best_ks = ks
    
    # If time permits, try a larger sieve
    elapsed = time.time() - start_time
    if elapsed < 100:
        N2 = 500000
        sigma2 = _sigma_sieve(N2)
        m_to_ks2 = defaultdict(list)
        for k in range(1, N2 + 1):
            m = k * sigma2[k]
            if m <= 10**18 and k <= math.isqrt(m):
                m_to_ks2[m].append(k)
        
        for m, ks in m_to_ks2.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_m = m
                best_ks = ks
        
        elapsed = time.time() - start_time
    
    if elapsed < 120:
        N3 = 1000000
        sigma3 = _sigma_sieve(N3)
        m_to_ks3 = defaultdict(list)
        for k in range(1, N3 + 1):
            m = k * sigma3[k]
            if m <= 10**18 and k <= math.isqrt(m):
                m_to_ks3[m].append(k)
        
        for m, ks in m_to_ks3.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_m = m
                best_ks = ks
    
    # Verify and return
    # Ensure types are correct
    best_m = int(best_m)
    best_ks = list(set(int(k) for k in best_ks))
    best_ks.sort()
    
    return {'M': best_m, 'k': best_ks}
# EVOLVE-BLOCK-END
