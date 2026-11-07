import math

N = 210
MAX_VAL = 10210
MAX_SIZE = 100

# Precompute squarefree kernel for numbers up to MAX_VAL
def compute_squarefree_kernels(limit):
    kernels = [1] * (limit + 1)
    for i in range(2, limit + 1):
        if kernels[i] == 1:  # i is prime
            for j in range(i, limit + 1, i):
                temp = j
                exp = 0
                while temp % i == 0:
                    temp //= i
                    exp += 1
                if exp % 2 == 1:
                    kernels[j] *= i
    return kernels

def is_perfect_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    # Use integer square root
    x = math.isqrt(n)
    return x * x == n

def construct():
    # Target: find up to 100 distinct integers in (210, 10210]
    # such that their product * 210 is a perfect square.
    # Minimize max(selected) - 210.
    
    # 210 = 2 * 3 * 5 * 7. All exponents are 1 (odd).
    # So we need the product of selected numbers to have odd exponents for 2, 3, 5, 7
    # to make the total exponents even. Actually, wait:
    # Product = 210 * P. 210 has squarefree part 210.
    # We need squarefree(210 * P) = 1.
    # squarefree(210 * P) = squarefree(210) * squarefree(P) mod squares?
    # More precisely, the parity vector of exponents in 210 is [1,1,1,1,0,0...] for primes 2,3,5,7.
    # We need the parity vector of P to be identical to that of 210, so that their sum is [0,0,0,0,...].
    # Let v(n) be the parity vector of prime exponents of n (squarefree kernel encoded as product of primes with odd exponents).
    # We need v(P) = v(210) = 210.
    # So the product of the squarefree kernels of the selected numbers must be divisible by 210 in the group of squarefree integers under multiplication modulo squares?
    # Actually, if we represent each number by its squarefree kernel k, then the product of numbers has squarefree kernel equal to the product of k's, reduced modulo squares.
    # We want the squarefree kernel of the product to be 210.
    
    # Strategy:
    # To minimize the maximum value, we should pick small numbers.
    # We can try to find a set of numbers with small values whose squarefree kernels multiply to 210 (mod squares).
    # Since we can pick up to 100 numbers, we have a lot of flexibility.
    # We can simply pick numbers that are squares (kernel 1) to fill up the list, and a small set of numbers that fix the parity.
    
    # Let's find a small set S of numbers > 210 such that product of their kernels is 210 (mod squares).
    # Then we can add squares (numbers whose kernel is 1) to reach any size, but we don't need to reach 100, we just need at most 100.
    # The task says "1..100 distinct integers". So we can return just the small set S if it works.
    # But we must minimize max(S) - 210.
    
    # Let's search for a combination of numbers from (210, some upper bound) whose kernel product is 210.
    # We can do a BFS or dynamic programming on the squarefree kernel?
    # The squarefree kernel can be large. But we only care about the primes involved.
    # Actually, we can just check combinations of small numbers.
    
    # Let's try to find a single number n > 210 such that kernel(n) = 210.
    # Then n itself works? No, we need product * 210 to be a square.
    # If we pick one number n, then n * 210 must be a square.
    # This means kernel(n) * kernel(210) must be a square. Since kernels are squarefree, this means kernel(n) = kernel(210) = 210.
    # So if we find n > 210 with kernel 210, we can return [n].
    # The smallest such n is 210 * 1^2 = 210 (not > 210). Next is 210 * 2^2 = 840.
    # So [840] is a valid solution with max - 210 = 630.
    
    # Can we do better with 2 numbers?
    # n1 * n2 * 210 is a square.
    # kernel(n1) * kernel(n2) = 210 (mod squares).
    # Let's try to find n1, n2 > 210 with small max.
    # Suppose n1 = 210 * a^2, n2 = 210 * b^2. Then n1*n2*210 = 210^3 * a^2 b^2, not a square.
    # We need kernel(n1) * kernel(n2) = 210.
    # Let kernel(n1) = k1, kernel(n2) = k2. k1*k2 = 210.
    # Possible pairs (k1, k2): (1, 210), (2, 105), (3, 70), (5, 42), (6, 35), (7, 30), (10, 21), (14, 15), and symmetric.
    # For each pair, find smallest n1 > 210 with kernel k1, and smallest n2 > 210 with kernel k2.
    # Then check if n1 != n2.
    
    # Let's compute this systematically.
    kernels_to_try = [1, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 210]
    # For each kernel k, find smallest n > 210 with kernel(k).
    # n = k * m^2. We need k * m^2 > 210.
    # m^2 > 210 / k. m > sqrt(210/k).
    
    def smallest_number_with_kernel(k, min_val=211):
        # Find smallest n >= min_val such that kernel(n) = k.
        # n = k * m^2.
        # m^2 >= ceil(min_val / k).
        # m = ceil(sqrt(ceil(min_val / k))).
        if k < min_val:
            m2 = math.ceil(min_val / k)
        else:
            m2 = 1  # if k >= min_val, m=1 might work, but we need n >= min_val. If k >= min_val, n=k works if kernel(k)=k (which is true).
        
        m = math.isqrt(m2)
        if m*m < m2:
            m += 1
        return k * m * m

    # Check all pairs
    best_max = float('inf')
    best_set = None
    
    # Single number
    for k in [210]:
        n = smallest_number_with_kernel(k)
        if n > 210:
            if n < best_max:
                best_max = n
                best_set = [n]
    
    # Two numbers
    for i in range(len(kernels_to_try)):
        for j in range(i+1, len(kernels_to_try)):
            k1 = kernels_to_try[i]
            k2 = kernels_to_try[j]
            if k1 * k2 != 210:
                continue
            n1 = smallest_number_with_kernel(k1)
            n2 = smallest_number_with_kernel(k2)
            if n1 > 210 and n2 > 210 and n1 != n2:
                mx = max(n1, n2)
                if mx < best_max:
                    best_max = mx
                    best_set = [n1, n2]
    
    # Three numbers? Might get smaller max.
    # k1*k2*k3 = 210.
    # This is more complex. Let's first see if 2 numbers give a good result.
    
    # If best_set is None, fall back to single number or generate more.
    if best_set is None:
        best_set = [840]
    
    # Verify the result
    prod = 210
    for x in best_set:
        prod *= x
    if not is_perfect_square(prod):
        # Fallback to known good
        return [840]
    
    # Ensure distinctness and constraints
    if len(best_set) != len(set(best_set)):
        return [840]
    if any(x <= 210 or x > 10210 for x in best_set):
        return [840]
    if len(best_set) > 100:
        return [840]
    
    return list(best_set)

# Test at module level to ensure correctness
if __name__ == "__main__":
    res = construct()
    prod = 210
    for x in res:
        prod *= x
    sqrt_p = math.isqrt(prod)
    print(f"Result: {res}")
    print(f"Max - 210: {max(res) - 210}")
    print(f"Is square: {sqrt_p * sqrt_p == prod}")
