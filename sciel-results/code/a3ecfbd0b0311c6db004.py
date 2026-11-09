import math
import time

def _squarefree_kernel(n):
    """Compute the squarefree kernel of n."""
    result = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            if cnt % 2 == 1:
                result *= d
        d += 1 if d == 2 else 2
    if n > 1:
        result *= n
    return result

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    target_kernel = 210  # sf(210) = 210 since 210 is squarefree
    
    # For each U from 211 upward, check if valid subset exists
    # Use incremental approach: maintain list of numbers and their kernels
    
    # Precompute kernels for numbers up to some limit
    max_val = 10210
    
    # We'll try increasing max values
    # For efficiency, use a set-based approach to find subsets
    
    # Key insight: we need product of kernels to have squarefree part = 210
    # This means: in GF(2) vector space, sum of kernel vectors = vector of 210
    
    # For small U, brute force subsets is infeasible if U-210 is large
    # But we can use the fact that we only need 1-100 elements
    
    # Alternative: find pairs, triples, etc. whose kernels multiply to 210 mod squares
    # i.e., kernel(a) * kernel(b) * ... has squarefree part 210
    
    # Let's try: find numbers a, b, ..., k such that sf(a*b*...*k) = 210
    
    # Simple approach: try to find small sets
    # For a single number: sf(n) = 210 means n = 210 * m^2 for some m
    # 210 * m^2 > 210 and <= 10210 => m^2 <= 48.6 => m <= 6
    # m=1: 210 (not > 210)
    # m=2: 840
    # m=3: 1890
    # m=4: 3360
    # m=5: 5250
    # m=6: 7560
    # So single-number solutions exist! Smallest is 840.
    # max - 210 = 840 - 210 = 630
    
    # Can we do better with multiple numbers?
    # For two numbers a, b: sf(a*b) = 210
    # This means sf(a) * sf(b) has squarefree part 210
    # i.e., sf(a) * sf(b) = 210 * s^2 for some squarefree s... no
    # Actually sf(a*b) = sf(sf(a) * sf(b)) since sf is multiplicative up to squares
    # sf(a*b) = 210 means sf(a)*sf(b) = 210 * t^2 for some integer t
    # But sf(a) and sf(b) are squarefree, so sf(a)*sf(b) is squarefree iff they're coprime
    # If not coprime, sf(a)*sf(b) may have repeated factors
    
    # Let me just try: for each U from 211 to 840, check if any subset works
    # Use a practical search
    
    # Precompute squarefree kernels
    kernels = {}
    for i in range(211, min(max_val + 1, 1500)):  # start with smaller range
        kernels[i] = _squarefree_kernel(i)
    
    # Check single numbers
    best_max = None
    for n in range(211, min(max_val + 1, 1500)):
        if kernels.get(n) == target_kernel or _squarefree_kernel(n) == target_kernel:
            if best_max is None or n < best_max:
                best_max = n
            break  # first one is smallest
    
    # Actually let me be more careful
    best_solution = None
    best_max_val = max_val + 1
    
    # Single number check
    for m in range(2, 100):
        n = 210 * m * m
        if n > 210 and n <= max_val:
            if n < best_max_val:
                best_max_val = n
                best_solution = [n]
    
    # Try pairs
    # sf(a) * sf(b) should have squarefree part 210
    # Let's try small a, b
    if time.time() < deadline:
        for a in range(211, min(500, max_val + 1)):
            if time.time() > deadline:
                break
            ka = _squarefree_kernel(a)
            for b in range(a + 1, min(500, max_val + 1)):
                if time.time() > deadline:
                    break
                kb = _squarefree_kernel(b)
                # Check if sf(ka * kb) == 210
                prod = ka * kb
                if _squarefree_kernel(prod) == target_kernel:
                    if b < best_max_val:
                        best_max_val = b
                        best_solution = [a, b]
    
    # Try triples
    if time.time() < deadline:
        for a in range(211, min(350, max_val + 1)):
            if time.time() > deadline:
                break
            ka = _squarefree_kernel(a)
            for b in range(a + 1, min(350, max_val + 1)):
                if time.time() > deadline:
                    break
                kb = _squarefree_kernel(b)
                for c in range(b + 1, min(350, max_val + 1)):
                    if time.time() > deadline:
                        break
                    kc = _squarefree_kernel(c)
                    prod = ka * kb * kc
                    if _squarefree_kernel(prod) == target_kernel:
                        if c < best_max_val:
                            best_max_val = c
                            best_solution = [a, b, c]
    
    # Verify best solution
    if best_solution is None:
        # Fallback: use 840
        best_solution = [840]
        best_max_val = 840
    
    # Final verification
    prod = 210
    for x in best_solution:
        prod *= x
    if not _is_square(prod):
        # Fallback
        best_solution = [840]
        best_max_val = 840
    
    # Ensure types
    result = [int(x) for x in best_solution]
    return result
