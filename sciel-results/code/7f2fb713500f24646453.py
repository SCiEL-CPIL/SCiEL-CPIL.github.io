import math
import time

def _square_free_part(n):
    """Return the square-free part of n."""
    sf = 1
    d = 2
    while d * d <= n:
        while n % (d * d) == 0:
            n //= (d * d)
        d += 1
    return sf * n

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s guard
    
    # We want to minimize M - 210.
    # Iterate M from 211 upwards.
    for M in range(211, 10211):
        if time.time() > deadline:
            break
            
        # Calculate K = S(210 * M)
        # S(210 * M) = S(S(210) * S(M)) = S(210 * S(M))
        # Since 210 is square-free.
        m_sf = _square_free_part(M)
        val = 210 * m_sf
        K = _square_free_part(val)
        
        # We need x = K * j^2 such that 210 < x < M
        # j^2 > 210 / K
        # j > sqrt(210 / K)
        
        # Calculate j_min
        # j_min is the smallest integer j such that K * j^2 > 210
        # j^2 > 210 / K
        # j > sqrt(210 / K)
        
        # To avoid float issues:
        # Find smallest j such that K * j * j > 210
        # j * j > 210 / K
        # j > sqrt(210 / K)
        
        # Let's estimate j
        # If K > 210, j=1 might work?
        # If K=1, j > sqrt(210) ~ 14.4 -> j=15.
        
        # Integer math for bound:
        # We want smallest j such that j*j > 210//K (if 210%K==0) or ceil(sqrt(210/K))
        # Let's just iterate j from 1 upwards? No, K can be small, j can be large.
        # But M < 10210, so x < 10210.
        # K * j^2 < 10210.
        # If K=1, j^2 < 10210, j < 101.
        # So j is small.
        
        j = 1
        found = False
        while K * j * j <= 210:
            j += 1
            
        # Now K * j^2 > 210
        x = K * j * j
        
        if x < M:
            # Check distinctness (x != M is guaranteed by x < M)
            # Check range (x > 210 guaranteed by loop)
            # Check product is square
            # prod = 210 * x * M
            # k = math.isqrt(prod)
            # if k*k == prod:
            #     return [x, M]
            
            # The construction guarantees S(210*x*M)=1
            # 210*x*M = 210 * (K j^2) * M
            # S(210 * M) = K
            # So 210 * M = K * s^2 for some s?
            # No, S(210 M) = K means 210 M = K * y^2.
            # Then 210 * x * M = (K y^2) * (K j^2) = K^2 (yj)^2 = (K y j)^2.
            # Yes, it is a perfect square.
            
            return [x, M]
            
    # Fallback if no M found (should not happen)
    return [840]
