import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 256
    
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(x**0.5) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    def is_valid(S):
        """Check if for all distinct a, b, c in S, a does not divide b*c.
        Uses gcd optimization: a | b*c iff (a/gcd(a,b)) | c."""
        S_list = list(S)
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if i == j:
                    continue
                g = gcd(a, b)
                need = a // g
                for k, c in enumerate(S_list):
                    if i == k or j == k:
                        continue
                    if c % need == 0:
                        return False
        return True
    
    # Start with all primes (known valid)
    primes = [i for i in range(2, n + 1) if is_prime(i)]
    best_set = set(primes)
    best_size = len(best_set)
    
    # Try to add composites greedily
    composites = [i for i in range(2, n + 1) if not is_prime(i)]
    
    # Sort by number of multiples (fewer multiples = easier to add)
    def difficulty(x):
        return sum(1 for v in range(2, n + 1) if v % x == 0)
    
    composites.sort(key=difficulty)
    
    for c in composites:
        if time.time() > deadline - 20:
            break
        if c in best_set:
            continue
        new_set = best_set | {c}
        if is_valid(new_set):
            best_set = new_set
            best_size = len(best_set)
    
    # Local search: try swaps (remove one, add one) to increase size
    all_composites = [i for i in range(2, n + 1) if not is_prime(i)]
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for a in list(best_set):
            if time.time() > deadline - 10:
                break
            for c in all_composites:
                if c in best_set:
                    continue
                new_set = (best_set - {a}) | {c}
                if len(new_set) > best_size:
                    if is_valid(new_set):
                        best_set = new_set
                        best_size = len(best_set)
                        improved = True
                        break
    
    return list(best_set)
