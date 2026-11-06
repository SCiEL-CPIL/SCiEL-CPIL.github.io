import math
import numpy as np

def construct():
    # Primes 1 mod 4
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97]
    N = 1
    for p in primes:
        N *= p
        
    # N is approx 9.9 * 10^16
    # We need u^2 + v^2 = N, u < v
    # u < sqrt(N/2)
    limit = int(math.isqrt(N // 2))
    
    pairs = []
    chunk_size = 10**6
    
    for start in range(1, limit + 1, chunk_size):
        end = min(start + chunk_size, limit + 1)
        u = np.arange(start, end)
        v2 = N - u * u
        # Check if v2 is a perfect square
        # v = isqrt(v2)
        # v*v == v2
        v = np.sqrt(v2)
        # Round to nearest integer
        v_rounded = np.round(v).astype(np.int64)
        # Check equality
        # Note: v2 can be large, up to 10^18. int64 is fine.
        # But np.sqrt returns float64. Precision of float64 is 53 bits ~ 15-16 decimal digits.
        # 10^18 is 60 bits. So float64 sqrt might lose precision.
        # We need to be careful.
        # Instead of np.sqrt, we can use integer sqrt if available in numpy? No.
        # We can check (v_rounded * v_rounded == v2)
        # But v_rounded * v_rounded might overflow int64? No, 10^18 < 9.2*10^18.
        # However, float precision issue:
        # If v2 is close to a square, sqrt might be off by 1 ulp.
        # So we should check v_rounded, v_rounded-1, v_rounded+1.
        
        candidates = []
        # This loop inside vectorized is bad.
        # Better:
        # v1 = v_rounded
        # v2_check = v1 * v1
        # mask = (v2_check == v2)
        # But if precision fails, we miss it.
        
        # Alternative: Use math.isqrt in a loop? Too slow.
        # Alternative: Use a precise integer sqrt check.
        # We can compute v_int = np.sqrt(v2)
        # Then check v_int*v_int, (v_int-1)*(v_int-1), (v_int+1)*(v_int+1)
        
        v_int = np.sqrt(v2).astype(np.int64)
        
        # Check v_int
        mask1 = (v_int * v_int == v2)
        # Check v_int - 1
        mask2 = ((v_int - 1) * (v_int - 1) == v2)
        # Check v_int + 1
        mask3 = ((v_int + 1) * (v_int + 1) == v2)
        
        valid_mask = mask1 | mask2 | mask3
        
        valid_u = u[valid_mask]
        valid_v2 = v2[valid_mask]
        
        # For each valid, find the exact v
        # We can just recompute isqrt for the few valid ones
        for uu, vv2 in zip(valid_u, valid_v2):
            vv = math.isqrt(int(vv2))
            if vv * vv == vv2:
                # Ensure u < v
                if uu < vv:
                    pairs.append([int(uu * uu), int(vv * vv)])
            
    return {'N': N, 'pairs': pairs}
