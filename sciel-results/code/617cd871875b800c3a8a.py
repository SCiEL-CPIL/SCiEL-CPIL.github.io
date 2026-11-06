import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    # Precompute primes up to 500 for fast factorization of b
    # max b is 256 * 192 = 49152
    # sqrt(49152) is ~221.
    primes = []
    is_prime = [True] * 500
    for i in range(2, 500):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 500, i):
                is_prime[j] = False

    def get_divisors_of_square(b):
        # Factorize b
        n_val = b
        factors = {}
        for p in primes:
            if p * p > n_val:
                break
            while n_val % p == 0:
                factors[p] = factors.get(p, 0) + 1
                n_val //= p
        if n_val > 1:
            factors[n_val] = factors.get(n_val, 0) + 1
        
        # Generate divisors of b^2
        # b = prod p_i^e_i => b^2 = prod p_i^(2e_i)
        divs = [1]
        for p, e in factors.items():
            exp = 2 * e
            current_divs = []
            for d in divs:
                val = d
                for _ in range(exp + 1):
                    current_divs.append(val)
                    val *= p
            divs = current_divs
        
        # Filter divisors k where k < b (to ensure y < z if k positive)
        # And also consider negative divisors if needed, but for positive y,z
        # and A, B > 0, k must be positive.
        # If A < 0 (x < n/4), not possible.
        # So we only return positive divisors less than b.
        # However, if k > b, then X > B, Y < B.
        # If X > Y, we just swap them.
        # To keep y < z, we need X < Y.
        # So we just need to iterate k such that k < sqrt(b^2) = b.
        return [d for d in divs if d < b]

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 1/x > 4/(3n) (since y,z > x => 1/y, 1/z < 1/x => 4/n < 3/x => x < 3n/4)
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            b = n * x
            
            divs = get_divisors_of_square(b)
            
            for k in divs:
                # k is X. Y = b^2 / k.
                # Since k < b, Y > b, so X < Y, so y < z.
                
                # Check divisibility for y
                if (k + b) % a != 0:
                    continue
                
                y = (k + b) // a
                if y <= x:
                    continue
                
                # Check divisibility for z
                y_term = b * b // k
                if (y_term + b) % a != 0:
                    continue
                
                z = (y_term + b) // a
                
                # Check distinctness and bounds
                if z <= y:
                    continue
                
                if z > 10**12:
                    continue
                
                # Verify cross multiplication just in case (though derived exactly)
                # 1/x + 1/y + 1/z == 4/n
                # n(yz + xz + xy) == 4xyz
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
