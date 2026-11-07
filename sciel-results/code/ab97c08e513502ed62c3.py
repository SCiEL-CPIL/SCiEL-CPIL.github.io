# EVOLVE-BLOCK-START
import math

def construct():
    # Primes > 16
    primes_gt_16 = [
        17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 
        101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 
        179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251
    ]
    
    # Selected composites/powers for small primes
    # 5^1 * 7^2 = 245
    # 2^8 = 256
    # 3^5 = 243
    # 11^2 = 121
    # 13^2 = 169
    small_part = [1, 245, 256, 243, 121, 169]
    
    result = small_part + primes_gt_16
    
    # Verify pairwise coprime
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback to all primes + 1 if logic fails
                return [1] + [p for p in range(2, 257) if all(p % d != 0 for d in range(2, int(math.isqrt(p)) + 1))]
                
    return result
# EVOLVE-BLOCK-END
