"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def _get_primes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def _squarefree_kernel(n, primes):
    """Return the squarefree kernel of n as a frozenset of prime indices."""
    kernel = set()
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count % 2 == 1:
                kernel.add(p)
    if n > 1:
        kernel.add(n)
    return frozenset(kernel)

def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    root = math.isqrt(n)
    return root * root == n

def _validate(selected, n=210):
    """Validate that the selection meets all constraints."""
    if not isinstance(selected, (list, tuple)):
        return False
    if len(selected) < 1 or len(selected) > 100:
        return False
    if len(set(selected)) != len(selected):
        return False
    for x in selected:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= 210 or x > 10210:
            return False
    product = n
    for x in selected:
        product *= x
    return _is_square(product)

def _find_best_pair(n=210, max_val=10210, deadline=None):
    """Find a pair (a, b) with a, b > n, a*b*n is a perfect square, minimizing max(a,b)-n."""
    if deadline is not None and time.time() > deadline:
        return None
    
    primes = _get_primes(max_val)
    
    # Compute squarefree kernels for all candidates
    candidates = list(range(n + 1, max_val + 1))
    kernels = {}
    kernel_to_nums = defaultdict(list)
    
    for x in candidates:
        k = _squarefree_kernel(x, primes)
        kernels[x] = k
        kernel_to_nums[k].append(x)
    
    # The target kernel: we need kernel(a) XOR kernel(b) = kernel(n)
    target = _squarefree_kernel(n, primes)
    
    best = None
    best_val = float('inf')
    
    # For each kernel, if we have multiple numbers, we can pick the smallest pair
    # We need kernel_a XOR kernel_b = target, i.e., kernel_b = kernel_a XOR target
    
    for k_a, nums_a in kernel_to_nums.items():
        if deadline is not None and time.time() > deadline:
            break
        # Compute k_b = k_a XOR target
        k_b = k_a ^ target
        if k_b in kernel_to_nums:
            nums_b = kernel_to_nums[k_b]
            # We need distinct a and b
            if k_a == k_b:
                # Need at least 2 distinct numbers with this kernel
                if len(nums_a) >= 2:
                    a, b = nums_a[0], nums_a[1]
                    val = max(a, b) - n
                    if val < best_val:
                        best_val = val
                        best = [a, b]
            else:
                a = nums_a[0]
                b = nums_b[0]
                val = max(a, b) - n
                if val < best_val:
                    best_val = val
                    best = [a, b]
    
    return best

def _find_best_triple(n=210, max_val=10210, deadline=None):
    """Find a triple (a, b, c) with a*b*c*n is a perfect square, minimizing max(a,b,c)-n."""
    if deadline is not None and time.time() > deadline:
        return None
    
    primes = _get_primes(max_val)
    
    # Target: kernel(a) XOR kernel(b) XOR kernel(c) = kernel(n)
    target = _squarefree_kernel(n, primes)
    
    # Precompute kernels for small range first
    candidates = list(range(n + 1, min(n + 500, max_val + 1)))
    kernels = {}
    kernel_to_nums = defaultdict(list)
    
    for x in candidates:
        k = _squarefree_kernel(x, primes)
        kernels[x] = k
        kernel_to_nums[k].append(x)
    
    best = None
    best_val = float('inf')
    
    # Try pairs from small range, then find third
    small_candidates = candidates[:100]
    
    for i, a in enumerate(small_candidates):
        if deadline is not None and time.time() > deadline:
            break
        for b in small_candidates[i+1:]:
            if deadline is not None and time.time() > deadline:
                break
            k_ab = kernels[a] ^ kernels[b]
            k_c_needed = k_ab ^ target
            
            if k_c_needed in kernel_to_nums:
                for c in kernel_to_nums[k_c_needed]:
                    if c != a and c != b:
                        val = max(a, b, c) - n
                        if val < best_val:
                            best_val = val
                            best = [a, b, c]
                        break  # Take first valid c for this pair
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline for safety
    
    n = 210
    
    # Try to find a pair
    result = _find_best_pair(n, 10210, deadline)
    
    if result is not None:
        if _validate(result, n):
            return list(result)
    
    # Try to find a triple
    result = _find_best_triple(n, 10210, deadline)
    
    if result is not None:
        if _validate(result, n):
            return list(result)
    
    # Fallback: try single element
    # 210 * x must be a square. 210 = 2*3*5*7, so x must be 2*3*5*7 * k^2 = 210 * k^2
    # x > 210, so k >= 2: x = 210*4 = 840
    x = 210 * 4
    if x <= 10210:
        if _validate([x], n):
            return [x]
    
    # Last resort fallback
    return [840]
# EVOLVE-BLOCK-END
