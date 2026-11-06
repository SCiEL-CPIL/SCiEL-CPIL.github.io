import time
import math

def _find_egyptian(k, p, q, min_d, max_d=10000000):
    """
    Find k distinct unit fractions 1/d_i such that sum = p/q.
    d_i >= min_d and d_i <= max_d.
    Returns list of denominators or None.
    """
    if k == 0:
        return [] if p == 0 else None
    
    if k == 1:
        if p == 0:
            return []
        # p/q = 1/d => d = q/p
        if q % p == 0:
            d = q // p
            if min_d <= d <= max_d:
                return [d]
        return None

    # Pruning: max possible sum is k / min_d
    # If p/q > k / min_d, impossible
    if p * min_d > q * k:
        return None
    
    # Pruning: min possible sum is k / max_d (roughly, but distinctness matters)
    # Actually, just iterate d from min_d up to some limit
    
    # Upper bound for d:
    # 1/d <= p/q => d >= q/p
    # Also, remaining k-1 fractions must sum to p/q - 1/d.
    # Max sum of k-1 fractions is (k-1)/d_next where d_next > d.
    # So p/q - 1/d <= (k-1)/(d+1)
    # This gives an upper bound on d.
    
    # Let's just iterate d from min_d.
    # To speed up, we can bound d.
    # 1/d < p/q => d > q/p.
    # Also 1/d + (k-1)/(d+1) >= p/q.
    
    start_d = max(min_d, math.ceil(q / p))
    # If start_d > max_d, return None
    if start_d > max_d:
        return None
        
    # Upper bound for d:
    # We need p/q - 1/d >= (k-1)/max_d ? No.
    # We need p/q - 1/d to be representable by k-1 fractions >= d+1.
    # Max sum of k-1 fractions >= d+1 is (k-1)/(d+1).
    # So p/q - 1/d <= (k-1)/(d+1)
    # p/q <= 1/d + (k-1)/(d+1) = (d + 1 + kd - k) / (d(d+1)) = (k+1)d + (1-k) / (d^2+d)
    # For large d, p/q approx (k+1)/d => d approx (k+1)q/p.
    
    # Let's just loop d from start_d to a reasonable limit.
    # If k is small, the limit is small.
    # d_max = int((k * q) / p) + 100 is a safe over-estimate?
    # Actually, if d is too large, 1/d is too small, and we can't make up the sum with k-1 fractions.
    # Max sum of k-1 fractions with min denom d+1 is (k-1)/(d+1).
    # So we require p/q - 1/d <= (k-1)/(d+1).
    # p/q <= 1/d + (k-1)/(d+1).
    # This inequality holds for small d. For large d, RHS decreases.
    # So there is an upper bound on d.
    
    # Let's compute the bound.
    # p/q <= (k+1)d + (1-k) / (d^2+d)
    # p d^2 + p d <= q ( (k+1)d + 1-k )
    # p d^2 + p d - q(k+1)d - q(1-k) <= 0
    # p d^2 + (p - q(k+1))d - q(1-k) <= 0
    # Roots of p d^2 + (p - q(k+1))d + q(k-1) = 0
    # d <= (-B + sqrt(B^2 - 4AC)) / 2A
    A = p
    B = p - q * (k + 1)
    C = q * (k - 1)
    
    disc = B * B - 4 * A * C
    if disc < 0:
        return None
    sqrt_disc = math.isqrt(disc)
    if sqrt_disc * sqrt_disc != disc:
        # Adjust for integer arithmetic safety
        sqrt_disc += 1
    d_max = (-B + sqrt_disc) // (2 * A)
    
    if d_max < start_d:
        return None
        
    for d in range(start_d, d_max + 1):
        if p * d == q:
            # 1/d = p/q, so remaining sum is 0.
            # But we need k-1 more fractions. Impossible unless k=1.
            # If k>1, we can't have 0 remaining.
            continue
            
        rem_p = p * d - q
        rem_q = q * d
        if rem_p < 0:
            continue
            
        # Reduce fraction
        g = math.gcd(rem_p, rem_q)
        rem_p //= g
        rem_q //= g
        
        res = _find_egyptian(k - 1, rem_p, rem_q, d + 1, max_d)
        if res is not None:
            return [d] + res
            
    return None

def construct():
    start_time = time.time()
    deadline = 55.0 # 55s limit
    
    # Try n = 5, 6, ...
    for n in range(5, 257):
        if time.time() > deadline:
            break
            
        # 1 = 1/1. p=1, q=1.
        res = _find_egyptian(n, 1, 1, 5, 10000000)
        if res is not None:
            return res
            
    # Fallback to a known solution or the original DP if search fails (unlikely to fail for small n)
    # The original program found a 12-term solution.
    # Let's hardcode a 12-term solution as fallback just in case.
    # From original code: target 2520.
    # It returned something.
    # But the search above should find n=5 or 6 quickly.
    
    raise RuntimeError("No solution found within time limit")
