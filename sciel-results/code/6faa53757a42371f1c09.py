"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import defaultdict

def isqrt(n):
    return math.isqrt(n)

def construct():
    # We need to find M with many k such that k*sigma(k) = M
    # Strategy: enumerate k up to a limit, compute k*sigma(k), group by M
    
    # First, let's try known small M values with multiple solutions
    # 336*sigma(336) = 336 * (1+2+4+7+8+14+16+21+28+42+56+84+112+168+336)
    # sigma(336) = 1008, so 336*1008 = 338688... let me recalculate
    
    # Let me compute sigma properly for candidate k values
    # and find M values that appear multiple times
    
    # Sieve for sigma up to some limit
    LIMIT = 200000
    sigma = [0] * (LIMIT + 1)
    for i in range(1, LIMIT + 1):
        for j in range(i, LIMIT + 1, i):
            sigma[j] += i
    
    # Compute k*sigma(k) for all k and group by M
    m_to_k = defaultdict(list)
    for k in range(1, LIMIT + 1):
        m_val = k * sigma[k]
        if m_val <= 10**18:
            m_to_k[m_val].append(k)
    
    # Find M with most k solutions where k <= isqrt(M)
    best_m = None
    best_k_list = []
    for m_val, k_list in m_to_k.items():
        # Filter k values: must have k <= isqrt(m_val)
        valid_k = [k for k in k_list if k <= isqrt(m_val)]
        if len(valid_k) > len(best_k_list):
            best_k_list = valid_k
            best_m = m_val
    
    # Also check some known good candidates with larger k
    # Let's extend search with targeted candidates
    # Numbers with many small prime factors tend to have "nice" sigma values
    
    # Try specific constructions: k = product of small primes
    # k = 2^a * 3^b * 5^c * 7^d etc.
    
    # Let's also try to find M where k*sigma(k) collides more often
    # by searching a larger range with smarter enumeration
    
    # Extended search: try k up to 10^6 with sigma computation
    LIMIT2 = 1000000
    sigma2 = [0] * (LIMIT2 + 1)
    for i in range(1, LIMIT2 + 1):
        for j in range(i, LIMIT2 + 1, i):
            sigma2[j] += i
    
    m_to_k2 = defaultdict(list)
    for k in range(1, LIMIT2 + 1):
        m_val = k * sigma2[k]
        if m_val <= 10**18:
            m_to_k2[m_val].append(k)
    
    for m_val, k_list in m_to_k2.items():
        valid_k = [k for k in k_list if k <= isqrt(m_val)]
        if len(valid_k) > len(best_k_list):
            best_k_list = valid_k
            best_m = m_val
    
    if best_m is None:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Verify all solutions
    verified_k = []
    for k in best_k_list:
        # Compute sigma(k) exactly
        s = 0
        n = k
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                p_pow = 1
                while temp % d == 0:
                    p_pow *= d
                    temp //= d
                s_factor = (p_pow * d - 1) // (d - 1)
                s = 0 if s == 0 else s * s_factor
                if s == 0:
                    s = s_factor
            d += 1
        if temp > 1:
            s_factor = 1 + temp
            s = 0 if s == 0 else s * s_factor
            if s == 0:
                s = s_factor
        if s == 0:
            s = 1
        if k * s == best_m:
            verified_k.append(k)
    
    # Ensure distinct and sorted
    verified_k = sorted(set(verified_k))
    
    if len(verified_k) >= 3:
        return {'M': int(best_m), 'k': [int(k) for k in verified_k]}
    else:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}

# EVOLVE-BLOCK-END
