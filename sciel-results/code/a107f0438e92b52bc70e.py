import math

def construct():
    target = 255
    # Find the smallest prime factor of the target
    p = 2
    while target % p != 0:
        p += 1
        if p * p > target:
            p = target
            break
    
    # The optimal set for pairwise non-coprime containing 'target'
    # is typically the multiples of its smallest prime factor.
    result = set()
    for i in range(1, 256):
        if i % p == 0:
            result.add(i)
            
    return sorted(result)
