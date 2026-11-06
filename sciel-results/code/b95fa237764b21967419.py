import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def is_prime(n):
        if n < 2:
            return False
        for q in range(2, int(n**0.5) + 1):
            if n % q == 0:
                return False
        return True
    
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Initial solution: primes from 13 upward
    initial_primes = [p for p in primes if p >= 13]
    initial_residues = [(i + 1) % p for i, p in enumerate(initial_primes)]
    
    classes = list(zip(initial_primes, initial_residues))
    
    max_L = 10000
    
    # Build coverage count array
    count = [0] * (max_L + 1)
    for p, r in classes:
        for n in range(r, max_L + 1, p):
            if n >= 1:
                count[n] += 1
    
    def find_prefix(count):
        for n in range(1, max_L + 1):
            if count[n] == 0:
                return n - 1
        return max_L
    
    best_L = find_prefix(count)
    
    # Local search: optimize residues
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for i in range(len(classes)):
            p, r = classes[i]
            best_r = r
            best_L_local = best_L
            for new_r in range(p):
                if new_r == r:
                    continue
                # Update counts
                for n in range(r, max_L + 1, p):
                    if n >= 1:
                        count[n] -= 1
                for n in range(new_r, max_L + 1, p):
                    if n >= 1:
                        count[n] += 1
                L_local = find_prefix(count)
                if L_local > best_L_local:
                    best_L_local = L_local
                    best_r = new_r
                # Revert
                for n in range(new_r, max_L + 1, p):
                    if n >= 1:
                        count[n] -= 1
                for n in range(r, max_L + 1, p):
                    if n >= 1:
                        count[n] += 1
                if time.time() > deadline - 15:
                    break
            if best_r != r:
                for n in range(r, max_L + 1, p):
                    if n >= 1:
                        count[n] -= 1
                for n in range(best_r, max_L + 1, p):
                    if n >= 1:
                        count[n] += 1
                classes[i] = (p, best_r)
                best_L = best_L_local
                improved = True
            if time.time() > deadline - 15:
                break
    
    return [[p, r] for p, r in classes]
