"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def _squarefree_kernel(n):
    """Return the square-free kernel of n (product of primes with odd exponent)."""
    result = 1
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            n //= d
            count += 1
        if count % 2 == 1:
            result *= d
        d += 1 if d == 2 else 2  # skip even numbers after 2
    if n > 1:
        result *= n
    return result

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _validate(nums):
    """Check if 210 * product(nums) is a perfect square."""
    prod = 210
    for n in nums:
        prod *= n
    return _is_square(prod)

def _find_best_pair(max_val):
    """Find best pair (n1, n2) with 210 < n1 <= n2 <= max_val minimizing max_val - 210."""
    best_raw = float('inf')
    best_pair = None
    
    # Precompute square-free kernels for all candidates
    kernels = {}
    for n in range(211, max_val + 1):
        kernels[n] = _squarefree_kernel(n)
    
    # For 210 * n1 * n2 to be square, we need sf(210 * n1 * n2) = 1
    # sf(210) = 210, so we need sf(n1 * n2) = 210 (in parity sense)
    # Actually: the parity vector of (210 * n1 * n2) must be all zeros
    # parity(210) XOR parity(n1) XOR parity(n2) = 0
    # So parity(n1) XOR parity(n2) = parity(210)
    # This means sf(n1) * sf(n2) has the same square-free part as 210
    # i.e., sf(n1 * n2) = 210
    
    # More directly: for each n1, we need sf(n2) such that sf(n1*n2) = 210
    # Let's just check directly using square-free kernels
    # sf(n1*n2) = sf(n1) * sf(n2) / sf(gcd_sf) ... this is complex
    # Better: use the property that sf(a*b) can be computed, but let's just
    # use: 210*n1*n2 is square iff sf(210*n1*n2) == 1
    
    # Alternative: for each n1, compute what sf(n2) should be
    # sf(210*n1) * sf(n2) should have sf = 1
    # i.e., sf(210*n1) must equal sf(n2) (since sf(a)*sf(b) has sf=1 iff sf(a)=sf(b))
    # Wait: sf(a*b) = 1 iff a*b is a square iff sf(a) = sf(b) (both have same square-free part)
    # So we need sf(210*n1) = sf(n2)
    
    target_map = {}  # sf_value -> list of n2 with that sf
    for n2 in range(211, max_val + 1):
        k = kernels[n2]
        if k not in target_map:
            target_map[k] = []
        target_map[k].append(n2)
    
    for n1 in range(211, max_val + 1):
        target_sf = _squarefree_kernel(210 * n1)
        if target_sf in target_map:
            for n2 in target_map[target_sf]:
                if n2 >= n1:  # avoid duplicates
                    raw = n2 - 210
                    if raw < best_raw:
                        best_raw = raw
                        best_pair = (n1, n2)
    
    return best_raw, best_pair

def _find_best_triple(max_val):
    """Find best triple minimizing max element."""
    best_raw = float('inf')
    best_triple = None
    
    # Precompute square-free kernels
    kernels = {}
    for n in range(211, max_val + 1):
        kernels[n] = _squarefree_kernel(n)
    
    # For triple: sf(210*n1*n2*n3) = 1
    # sf(210*n1*n2) must equal sf(n3)
    # Precompute sf(210*n1*n2) for pairs, then look up n3
    
    # Build map: sf_value -> smallest n with that sf
    sf_to_min_n = {}
    for n in range(211, max_val + 1):
        k = kernels[n]
        if k not in sf_to_min_n or n < sf_to_min_n[k]:
            sf_to_min_n[k] = n
    
    # For each pair (n1, n2), compute sf(210*n1*n2), then find n3
    # But this is O(n^2) which might be slow for large max_val
    # Let's limit to reasonable max_val
    
    for n1 in range(211, max_val + 1):
        for n2 in range(n1, max_val + 1):
            sf_210_n1_n2 = _squarefree_kernel(210 * n1 * n2)
            if sf_210_n1_n2 in sf_to_min_n:
                n3 = sf_to_min_n[sf_210_n1_n2]
                if n3 >= n2:  # ensure distinct and ordered
                    # Check distinctness
                    if n1 != n2 and n2 != n3 and n1 != n3:
                        raw = n3 - 210
                        if raw < best_raw:
                            best_raw = raw
                            best_triple = (n1, n2, n3)
    
    return best_raw, best_triple

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    # Start with known good solution
    best_raw = 30
    best_result = [224, 240]
    
    # Try to find better pairs
    # Search for pairs with max element from 231 upward
    for max_val in range(231, 400):
        if time.time() > deadline:
            break
        raw, pair = _find_best_pair(max_val)
        if pair is not None and raw < best_raw:
            best_raw = raw
            best_result = list(pair)
        if raw < max_val - 210 + 1:
            # Found the best possible for this max_val range
            # Actually if we found a pair with max = max_val, we can stop
            # since we're searching in increasing max_val order
            break
    
    # Try triples for even better results
    if time.time() < deadline - 5:
        for max_val in range(231, 300):
            if time.time() > deadline:
                break
            raw, triple = _find_best_triple(max_val)
            if triple is not None and raw < best_raw:
                best_raw = raw
                best_result = list(triple)
    
    # Validate final result
    if not _validate(best_result):
        # Fallback to known good
        best_result = [224, 240]
    
    return best_result

# EVOLVE-BLOCK-END
