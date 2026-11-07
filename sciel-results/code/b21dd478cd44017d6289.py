"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    
    def check_conflict(new_elem, S):
        """Check if adding new_elem to S creates any conflict."""
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                a = S[i]
                b = S[j]
                # Check triple (new_elem, a, b): new_elem must not divide a*b
                if a * b % new_elem == 0:
                    return True
                # Check triple (a, new_elem, b): a must not divide new_elem*b
                if (new_elem * b) % a == 0:
                    return True
                # Check triple (b, new_elem, a): b must not divide new_elem*a
                if (new_elem * a) % b == 0:
                    return True
        return False
    
    # Try different orderings and pick the best
    best_set = []
    
    # Strategy 1: Greedy from largest to smallest
    S = []
    for x in range(256, 1, -1):
        if time.time() > deadline:
            break
        if not check_conflict(x, S):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S[:]
    
    # Strategy 2: Greedy from smallest to largest
    S = []
    for x in range(2, 257):
        if time.time() > deadline:
            break
        if not check_conflict(x, S):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S[:]
    
    # Strategy 3: Primes first, then try to add composites
    primes = [x for x in range(2, 257) if all(x % d for d in range(2, int(x**0.5) + 1))]
    S = primes[:]
    for x in range(2, 257):
        if x in primes:
            continue
        if time.time() > deadline:
            break
        if not check_conflict(x, S):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S[:]
    
    # Strategy 4: Only numbers > 128
    S = []
    for x in range(256, 128, -1):
        if time.time() > deadline:
            break
        if not check_conflict(x, S):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S[:]
    
    # Strategy 5: Numbers in [100, 256]
    S = []
    for x in range(256, 99, -1):
        if time.time() > deadline:
            break
        if not check_conflict(x, S):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S[:]
    
    # Strategy 6: Try to add small composites to prime set
    primes = [x for x in range(2, 257) if all(x % d for d in range(2, int(x**0.5) + 1))]
    S = primes[:]
    # Try composites in order of "safety" - those with large prime factors
    composites = [x for x in range(2, 257) if x not in primes]
    # Sort composites by largest prime factor descending (safer to include)
    def largest_prime_factor(n):
        lp = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                lp = d
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            lp = n
        return lp
    composites.sort(key=largest_prime_factor, reverse=True)
    for x in composites:
        if time.time() > deadline:
            break
        if not check_conflict(x, S):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S[:]
    
    return list(best_set)
# EVOLVE-BLOCK-END
