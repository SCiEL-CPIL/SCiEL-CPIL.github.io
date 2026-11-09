"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _sieve_sigma(n):
    """Compute sigma(i) for all i in [1, n] using a sieve."""
    sigma = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            sigma[j] += i
    return sigma

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    best_m = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Phase 1: Sieve up to 3M
    K1 = 3000000
    sigma = _sieve_sigma(K1)
    m_to_ks = defaultdict(list)
    for k in range(1, K1 + 1):
        m = k * sigma[k]
        if m <= 10**18:
            m_to_ks[m].append(k)
    
    for m, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_m = m
            best_ks = ks[:]
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE * 0.35:
        isq = math.isqrt(best_m)
        verified_ks = [k for k in best_ks if k <= isq]
        return {'M': int(best_m), 'k': [int(k) for k in verified_ks]}
    
    # Phase 2: Extend sieve to 7M
    if time.time() - start_time < DEADLINE * 0.5:
        K2 = 7000000
        sigma2 = _sieve_sigma(K2)
        for k in range(1, K2 + 1):
            m = k * sigma2[k]
            if m <= 10**18:
                m_to_ks[m].append(k)
        
        for m in list(m_to_ks.keys()):
            m_to_ks[m] = list(set(m_to_ks[m]))
        
        for m, ks in m_to_ks.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_m = m
                best_ks = ks[:]
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE * 0.75:
        isq = math.isqrt(best_m)
        verified_ks = [k for k in best_ks if k <= isq]
        return {'M': int(best_m), 'k': [int(k) for k in verified_ks]}
    
    # Phase 3: Extend to 12M if time permits
    if time.time() - start_time < DEADLINE * 0.6:
        K3 = 12000000
        sigma3 = _sieve_sigma(K3)
        for k in range(1, K3 + 1):
            m = k * sigma3[k]
            if m <= 10**18:
                m_to_ks[m].append(k)
        
        for m in list(m_to_ks.keys()):
            m_to_ks[m] = list(set(m_to_ks[m]))
        
        for m, ks in m_to_ks.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_m = m
                best_ks = ks[:]
    
    # Verify: ensure k_i <= isqrt(M)
    isq = math.isqrt(best_m)
    verified_ks = [k for k in best_ks if k <= isq]
    
    if not verified_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': int(best_m), 'k': [int(k) for k in verified_ks]}

# EVOLVE-BLOCK-END
