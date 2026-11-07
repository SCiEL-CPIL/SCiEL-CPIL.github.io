import time
import random
from itertools import combinations

# Precompute vectors
def get_vectors(n_max):
    primes = []
    is_prime = [True] * (n_max + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n_max + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    vecs = {}
    for x in range(2, n_max + 1):
        v = [0] * len(primes)
        temp = x
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[x] = tuple(v)
    return primes, vecs

# Check if adding vec to set of vecs maintains distinct subset sums
# sums is a set of tuples (the subset sums of the current set)
def can_add(vec, sums):
    # New sums will be sums U {s + vec for s in sums}
    # We need to check if any (s1 + vec) == s2 for s1, s2 in sums
    # Or simply generate the new set and check size
    # Optimization: if len(sums) is large, this is slow.
    # But max size is likely < 20, so 2^20 = 1M sums.
    # 1M operations per check is okay if we do it few times.
    
    # Faster check:
    # For each s in sums, check if s + vec is in sums.
    # If yes, collision.
    # Wait, if s1 + vec == s2, then s2 - vec == s1.
    # So we just need to check if (s + vec) is in sums for any s in sums?
    # No, that checks if s+vec collides with an EXISTING sum.
    # We also need to check if s1+vec == s2+vec? No, that implies s1==s2.
    # So we only need to check if the new sums collide with the OLD sums.
    # The new sums are distinct among themselves because old sums were.
    # So:
    for s in sums:
        new_s = tuple(a + b for a, b in zip(s, vec))
        if new_s in sums:
            return False
    return True

def construct():
    start_time = time.time()
    timeout = 85.0
    
    n_max = 40
    primes, vecs = get_vectors(n_max)
    candidates = list(vecs.items())
    
    # Greedy initialization
    # Sort by something? Maybe sum of exponents?
    # Or just random order?
    # Let's try sorting by value to get small numbers first?
    # Or random restarts.
    
    best_set = []
    best_sums = {tuple([0]*len(primes))}
    
    # Greedy pass
    # Try multiple random orders
    for _ in range(100):
        if time.time() - start_time > timeout:
            break
            
        random.shuffle(candidates)
        current_set = []
        current_sums = {tuple([0]*len(primes))}
        
        for val, vec in candidates:
            if can_add(vec, current_sums):
                current_set.append(val)
                # Update sums
                new_sums = set(current_sums)
                for s in current_sums:
                    new_s = tuple(a + b for a, b in zip(s, vec))
                    new_sums.add(new_s)
                current_sums = new_sums
        
        if len(current_set) > len(best_set):
            best_set = current_set
            best_sums = current_sums
            # If we reached a high number, we might stop early?
            # But let's keep searching.
            
    # Local search?
    # Try to swap one element out and two in?
    # Or just return best found.
    
    # Ensure types are correct
    return [int(x) for x in best_set]
