import time
import math
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _can_match_hall(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1..b_n with i | b_i
    using Hall's condition with precomputed bitsets.
    """
    # Build candidate bitsets for each index i (1..n)
    # bitset[i] is a Python int where bit j is set if value (m+1+j) is divisible by i
    # We only care about values in (m, m+L], i.e., m+1, m+2, ..., m+L
    # But we need to consider that candidates for different i overlap
    
    # For Hall's condition: for every subset S of {1,...,n}, |union of C(S)| >= |S|
    # where C(i) = {v in (m,m+L] : i divides v}
    
    # Precompute bitsets for each i
    # Bit j represents value m+1+j (0-indexed within the window)
    bitsets = [0] * (n + 1)  # 1-indexed
    for i in range(1, n + 1):
        bs = 0
        for j in range(L):
            val = m + 1 + j
            if val % i == 0:
                bs |= (1 << j)
        bitsets[i] = bs
    
    # Check Hall's condition for all 2^n subsets
    # Use DP: for each subset mask, compute union of bitsets
    # union_bits[mask] = OR of bitsets[i] for i in mask
    # popcount(mask) = number of indices in mask
    
    # Precompute popcount for all masks
    popcount = [0] * (1 << n)
    for mask in range(1, 1 << n):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Compute union bitsets for all subsets using DP
    # union_bits[mask] = union_bits[mask without lowest set bit] | bitsets[lowest_set_bit]
    union_bits = [0] * (1 << n)
    for mask in range(1, 1 << n):
        # Find lowest set bit
        lb = mask & (-mask)
        idx = lb.bit_length() - 1  # 0-indexed, so index = idx+1 in 1-indexed
        prev = mask ^ lb
        union_bits[mask] = union_bits[prev] | bitsets[idx + 1]
    
    # Check Hall's condition: for all masks with popcount > 0, popcount(union_bits[mask]) >= popcount(mask)
    for mask in range(1, 1 << n):
        if popcount[mask] > 0:
            union_pop = bin(union_bits[mask]).count('1')
            if union_pop < popcount[mask]:
                return False
    
    return True

def _f_hall(m, n=12):
    """Binary search smallest L in [1, 144] using Hall's condition check."""
    lo, hi = 1, 144
    if not _can_match_hall(m, hi, n):
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match_hall(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Calculate LCM of 1..12
    p = reduce(_lcm, range(1, 13))
    
    # Generate critical m in [0, p-1]
    # m is critical if m+1 is a multiple of some i in 1..12
    critical_ms = set()
    for i in range(1, 13):
        for k in range(1, p // i + 1):
            m_val = k * i - 1
            if 0 <= m_val < p:
                critical_ms.add(m_val)
            
    # Also include 0 just in case
    critical_ms.add(0)
    
    # Sort for deterministic order
    critical_list = sorted(critical_ms)
    
    best_m = 0
    best_f = _f_hall(0)
    
    for m in critical_list:
        if time.time() > deadline:
            break
        f_val = _f_hall(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # If we have time left, also scan a broader range
    # The period is p=27720, so m mod p determines the answer
    # But we can also check non-critical m values if time permits
    if time.time() < deadline - 5:
        # Scan all m in [0, min(p-1, 1000000)] that we haven't checked
        # Actually, since the function is periodic with period p, checking all m in [0,p-1] is sufficient
        # But critical_ms already covers the interesting ones. Let's verify by checking a few more.
        for m in range(0, min(p, 1000001)):
            if time.time() > deadline:
                break
            if m not in critical_ms:
                f_val = _f_hall(m)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m
    
    return int(best_m)
