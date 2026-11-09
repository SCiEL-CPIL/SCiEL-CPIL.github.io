import math
import time

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n."""
    if n == 1:
        return 1
    result = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            power_sum = 1
            power = 1
            while temp % d == 0:
                temp //= d
                power *= d
                power_sum += power
            result *= power_sum
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _sigma_fast(n):
    """Compute sigma(n) using trial division up to sqrt(n)."""
    if n == 1:
        return 1
    result = 1
    temp = n
    # Factor out 2s
    if temp % 2 == 0:
        power_sum = 1
        power = 1
        while temp % 2 == 0:
            temp //= 2
            power *= 2
            power_sum += power
        result *= power_sum
    d = 3
    while d * d <= temp:
        if temp % d == 0:
            power_sum = 1
            power = 1
            while temp % d == 0:
                temp //= d
                power *= d
                power_sum += power
            result *= power_sum
        d += 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin
    
    # Search for M with many k such that k*sigma(k) = M
    # We'll compute k*sigma(k) for k up to some limit and group by value
    
    # Strategy: compute f(k) = k*sigma(k) for k in a range, find collisions
    # k*sigma(k) = M, and we need k <= sqrt(M) which is always true as shown
    
    # Let's search k from 1 up to some max_k
    # For each k, compute M = k*sigma(k)
    # Group by M, find M with most k's
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # We need M <= 10^18, so k*sigma(k) <= 10^18
    # sigma(k) can be up to ~k*log(log(k)) for highly composite numbers
    # So k can be up to roughly 10^18 / (10^18/100) = 100... no
    # sigma(k) >= k, so k*sigma(k) >= k^2, so k <= sqrt(10^18) = 10^9
    # But sigma(k) is usually much larger than k for composite k
    
    # Let's try to find M by searching k up to some limit
    # We'll use a dictionary to count occurrences
    
    from collections import defaultdict
    m_to_ks = defaultdict(list)
    
    max_k_search = 200000  # search k up to 200000 initially
    
    for k in range(1, max_k_search + 1):
        if time.time() > deadline:
            break
        s = _sigma_fast(k)
        M = k * s
        if M <= 10**18:
            m_to_ks[M].append(k)
    
    # Check for best M
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_M = M
            best_k = ks
            best_count = len(ks)
    
    # If we found something better, return it
    # Also try extending the search if we have time
    remaining = deadline - time.time()
    if remaining > 10 and best_count < 5:
        # Try a different range or extend
        max_k_search2 = 500000
        for k in range(max_k_search + 1, max_k_search2 + 1):
            if time.time() > deadline:
                break
            s = _sigma_fast(k)
            M = k * s
            if M <= 10**18:
                m_to_ks[M].append(k)
        
        for M, ks in m_to_ks.items():
            if len(ks) > best_count:
                best_M = M
                best_k = ks
                best_count = len(ks)
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(x) for x in best_k]
    
    return {'M': best_M, 'k': best_k}
