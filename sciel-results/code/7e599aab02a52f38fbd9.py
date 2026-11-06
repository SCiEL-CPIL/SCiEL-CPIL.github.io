import math

def construct():
    target = math.factorial(12)
    best_max = target # Initial upper bound is the single factor 12!
    
    # We will store the best solution found
    best_solution = [target]
    
    def dfs(rem, start, current_factors):
        nonlocal best_max, best_solution
        
        if rem == 1:
            # We have a valid factorization
            # Since we pick in increasing order, the last factor is the largest
            current_max = current_factors[-1]
            if current_max < best_max:
                best_max = current_max
                best_solution = current_factors[:]
            return
        
        # Pruning: 
        # 1. If rem is prime, it must be the last factor.
        #    It must be > start and < best_max.
        if is_prime(rem):
            if rem > start and rem < best_max:
                current_factors.append(rem)
                dfs(1, rem, current_factors)
                current_factors.pop()
            return
        
        # 2. If rem is composite, we try to pick a factor 'f'
        #    f must be >= start
        #    f must be <= sqrt(rem) to avoid duplicate sets (since we pick in order)
        #    However, we also allow f to be rem itself if rem is not prime? No, if rem is composite, 
        #    picking rem itself means the next rem is 1. But we checked is_prime(rem) above.
        #    So if rem is composite, we must split it.
        
        # The largest possible value for 'f' in this step is min(sqrt(rem), best_max - 1)
        # Why best_max - 1? Because if f >= best_max, the max will be at least f, which is not an improvement.
        # Also, we need at least one more factor after f, so rem // f must be > f.
        # So f < sqrt(rem) is a natural bound for the first factor of a pair if we want to maintain order f < rem//f.
        # But we can have more than 2 factors left.
        
        # Let's iterate f from start to min(sqrt(rem), best_max - 1)
        upper_bound = min(int(math.isqrt(rem)), best_max - 1)
        
        for f in range(start, upper_bound + 1):
            if rem % f == 0:
                # Check if the remaining part rem//f can be formed by factors > f
                # The remaining product is rem // f.
                # We recursively call dfs(rem // f, f + 1, ...)
                # Pruning: if rem // f <= f, it cannot be formed by factors > f (unless it's 1, but rem is composite so rem//f > 1)
                if rem // f > f: 
                    current_factors.append(f)
                    dfs(rem // f, f + 1, current_factors)
                    current_factors.pop()
                    
        # What if rem itself is a valid factor?
        # This is covered if we allowed f to go up to rem. But we restricted to sqrt(rem).
        # If rem is composite, it has factors <= sqrt(rem). 
        # If we don't find any factor f in [start, sqrt(rem)], then rem has no factor in that range.
        # This implies rem is prime (contradiction) or all prime factors are > sqrt(rem) (impossible for composite).
        # Wait, what if rem is a square of a prime > start? e.g. rem = p^2. sqrt(rem) = p.
        # If p >= start, we try f=p. rem%p==0. rem//p = p. 
        # Condition `rem // f > f` becomes `p > p` which is False.
        # So we prune. But p * p = rem is a valid factorization into two distinct factors? 
        # No, the factors must be DISTINCT. So p and p are not distinct.
        # So if rem = p^2, we cannot use p twice. 
        # Can we use other factors? No, prime factorization is unique.
        # So if rem is a square of a prime, it cannot be factored into distinct integers > 1 (unless we include 1, but factors > 12).
        # Actually, if rem = p^2, the only factors are 1, p, p^2.
        # We can't use 1. We can't use p twice. So rem=p^2 is a dead end for distinct factors > 1.
        
        # What if rem is a prime power p^k, k>2?
        # Factors must be distinct.
        # This logic holds: if no f in [start, sqrt(rem)] divides rem, then rem is prime.
        # But we already checked is_prime(rem).
        
        # There is one case: rem is composite, but its smallest factor is > sqrt(rem)? Impossible.
        
        # So the loop covers all cases where rem can be split into at least two factors f and g with f <= g.
        # If f < g, we proceed.
        # If f == g, we skip (distinctness).
        
        # Is it possible that the optimal solution has rem as a single factor?
        # Yes, if rem is prime. Handled above.
        # If rem is composite, it MUST be split into at least two factors.
        
    # Helper: Miller-Rabin or simple trial division for is_prime
    # Since 12! is small, simple trial division up to sqrt(n) is fast enough.
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    # Start DFS
    # The first factor must be > 12.
    # We start searching for factors from 13.
    dfs(target, 13, [])
    
    return best_solution
