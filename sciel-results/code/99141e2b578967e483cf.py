import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0
    target = 123457
    
    # Primes up to 200
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    def generate_smooth(prime_list, limit):
        """Generate all numbers <= limit whose prime factors are all in prime_list."""
        smooths = set([1])
        current = [1]
        for p in prime_list:
            new_vals = []
            for x in current:
                val = x
                while val * p <= limit:
                    val *= p
                    if val not in smooths:
                        smooths.add(val)
                        new_vals.append(val)
            current.extend(new_vals)
        return smooths

    def largest_prime_factor(n, prime_list):
        """Find the largest prime factor of n, assuming n is smooth with respect to prime_list."""
        if n == 1:
            return 1
        max_p = 1
        for p in prime_list:
            if p * p > n:
                break
            if n % p == 0:
                max_p = p
                while n % p == 0:
                    n //= p
        if n > 1:
            max_p = n
        return max_p

    # Generate smooth numbers
    smooths = generate_smooth(primes, target)
    
    best_score = float('inf')
    best_pair = (1, target - 1)
    
    # Check pairs
    # To avoid duplicates (a,b) and (b,a), we can just check all, or a < b.
    # Since we want min max(P(a),P(b)), symmetry holds.
    for a in smooths:
        if time.time() > deadline:
            break
        b = target - a
        if b < 1:
            continue
        if b in smooths:
            pa = largest_prime_factor(a, primes)
            pb = largest_prime_factor(b, primes)
            score = max(pa, pb)
            if score < best_score:
                best_score = score
                best_pair = (a, b)
                
    return list(best_pair)
# EVOLVE-BLOCK-END
