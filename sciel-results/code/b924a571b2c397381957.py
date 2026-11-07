import math
import time
from collections import defaultdict

def _squarefree_kernel(n):
    """Return frozenset of primes with odd exponent in factorization of n."""
    if n <= 1:
        return frozenset()
    result = frozenset()
    f = n
    d = 2
    while d * d <= f:
        if f % d == 0:
            exp = 0
            while f % d == 0:
                f //= d
                exp += 1
            if exp % 2 == 1:
                result = result | frozenset([d])
        d += 1 if d == 2 else 2
    if f > 1:
        result = result | frozenset([f])
    return result

def construct():
    deadline = time.time() + 78  # safety margin under 90s limit
    
    target = frozenset([2, 3, 5, 7])
    
    # Precompute kernels for all candidate numbers
    kernels = {}
    for i in range(211, 10211):
        kernels[i] = _squarefree_kernel(i)
    
    # Map from kernel to sorted list of numbers with that kernel
    kernel_to_numbers = defaultdict(list)
    for n in range(211, 10211):
        kernel_to_numbers[kernels[n]].append(n)
    
    # Try size 1: need a number with kernel exactly {2,3,5,7}
    if target in kernel_to_numbers and kernel_to_numbers[target]:
        n = kernel_to_numbers[target][0]
        prod = n * 210
        s = math.isqrt(prod)
        if s * s == prod:
            return [int(n)]
    
    # Try size 2: find a,b distinct with kernel(a) XOR kernel(b) = target
    # Minimize max(a,b)
    best_pair = None
    best_max = float('inf')
    
    for k1 in kernel_to_numbers:
        k2 = k1.symmetric_difference(target)
        if k2 in kernel_to_numbers:
            if k1 == k2:
                nums = kernel_to_numbers[k1]
                if len(nums) >= 2:
                    m = max(nums[0], nums[1])
                    if m < best_max:
                        best_max = m
                        best_pair = (nums[0], nums[1])
            else:
                nums1 = kernel_to_numbers[k1]
                nums2 = kernel_to_numbers[k2]
                if nums1 and nums2:
                    a = nums1[0]
                    b = nums2[0]
                    m = max(a, b)
                    if m < best_max:
                        best_max = m
                        best_pair = (a, b)
    
    if best_pair is not None:
        a, b = best_pair
        prod = a * b * 210
        s = math.isqrt(prod)
        if s * s == prod:
            return [int(a), int(b)]
    
    # Fallback: known valid solution
    return [840]
