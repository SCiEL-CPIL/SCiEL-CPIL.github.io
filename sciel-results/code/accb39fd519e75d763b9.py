from math import gcd
from itertools import product

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5)+1, 2):
        if n % i == 0:
            return False
    return True

def is_pairwise_coprime(nums):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if gcd(nums[i], nums[j]) != 1:
                return False
    return True

def construct():
    # Search for A, B maximizing distinct pairwise coprime sums
    best_count = 0
    best_result = None
    
    # We'll search in a reasonable range
    # Try to find 9 distinct pairwise coprime sums
    for A in [[1,2,3], [1,2,4], [1,3,5], [1,2,5], [2,3,5], [1,3,4], [1,4,5], [2,4,6], [1,2,6], [1,3,6], [1,4,6], [2,3,6], [1,5,7], [2,5,7], [3,5,7], [1,4,7], [2,4,7], [1,3,7], [2,3,7], [1,5,8], [2,5,8], [3,5,8], [1,4,8], [2,4,8], [1,3,8], [2,3,8], [1,6,9], [2,5,9], [3,5,9], [1,4,9], [2,4,9], [1,3,9], [2,3,9]]:
        for B in [[1,2,3], [1,2,4], [1,3,5], [1,2,5], [2,3,5], [1,3,4], [1,4,5], [2,4,6], [1,2,6], [1,3,6], [1,4,6], [2,3,6], [1,5,7], [2,5,7], [3,5,7], [1,4,7], [2,4,7], [1,3,7], [2,3,7], [1,5,8], [2,5,8], [3,5,8], [1,4,8], [2,4,8], [1,3,8], [2,3,8], [1,6,9], [2,5,9], [3,5,9], [1,4,9], [2,4,9], [1,3,9], [2,3,9]]:
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a+b)
            if len(sums) > best_count and is_pairwise_coprime(list(sums)):
                best_count = len(sums)
                best_result = [A, B]
    
    return best_result
