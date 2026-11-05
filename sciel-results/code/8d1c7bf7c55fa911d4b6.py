"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    N = 512
    
    # Precompute weights: w(a) = 1 / (512 - a)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    # Get all primes < 512
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(N)  # primes < 512
    prime_set = set(primes)
    
    # Start with all primes (they're pairwise coprime)
    # Then try to replace pairs of primes with their product if product < 512
    # This is the 2-for-1 swap that exploits convexity
    
    # Greedy: start with all primes
    chosen = list(primes)
    chosen_set = set(chosen)
    
    def calc_score(s):
        return sum(weights[a] for a in s)
    
    def is_coprime_to_set(num, s, exclude=None):
        for b in s:
            if b == exclude:
                continue
            if gcd(num, b) != 1:
                return False
        return True
    
    current_score = calc_score(chosen)
    best_score = current_score
    best_chosen = list(chosen)
    
    # 2-for-1 swap: replace two primes p, q with p*q if p*q < 512 and p*q is coprime to all others
    # Gain = 1/(512-p*q) - 1/(512-p) - 1/(512-q)
    # This gain is positive when p*q is large enough (close to 512)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        n = len(chosen)
        found_swap = False
        
        for i in range(n):
            if time.time() > deadline:
                break
            if found_swap:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                
                p = chosen[i]
                q = chosen[j]
                prod = p * q
                if prod >= N:
                    continue
                
                # Check if prod is coprime to all other chosen elements
                others = [chosen[k] for k in range(n) if k != i and k != j]
                if not is_coprime_to_set(prod, others):
                    continue
                
                gain = weights[prod] - weights[p] - weights[q]
                if gain > 1e-12:
                    # Perform the swap
                    chosen[i] = prod
                    chosen.pop(j)
                    chosen_set.remove(p)
                    chosen_set.remove(q)
                    chosen_set.add(prod)
                    current_score += gain
                    improved = True
                    found_swap = True
                    if current_score > best_score:
                        best_score = current_score
                        best_chosen = list(chosen)
                    break
        
    # 1-for-1 swap: try to replace a single element with a better one
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        n = len(chosen)
        for i in range(n):
            if time.time() > deadline:
                break
            
            a_remove = chosen[i]
            weight_remove = weights[a_remove]
            others = [chosen[k] for k in range(n) if k != i]
            
            # Find best candidate to replace a_remove
            best_add = -1
            best_gain = 0.0
            for a_add in range(N - 1, 0, -1):
                if a_add in chosen_set:
                    continue
                if not is_coprime_to_set(a_add, others):
                    continue
                
                gain = weights[a_add] - weight_remove
                if gain > best_gain:
                    best_gain = gain
                    best_add = a_add
                
                # Since we iterate from largest, first valid with positive gain is likely good
                # But let's check a few more to be safe
                if best_gain > 0 and a_add < a_remove:
                    break
            
            if best_add != -1 and best_gain > 1e-12:
                chosen[i] = best_add
                chosen_set.remove(a_remove)
                chosen_set.add(best_add)
                current_score += best_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_chosen = list(chosen)
                break  # Restart loop
    
    # More aggressive 2-for-1: also try replacing a prime and a composite with their product
    # Or two composites
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        n = len(chosen)
        found_swap = False
        
        for i in range(n):
            if time.time() > deadline:
                break
            if found_swap:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                
                a1 = chosen[i]
                a2 = chosen[j]
                prod = a1 * a2
                if prod >= N:
                    continue
                
                # Check if prod is coprime to all other chosen elements
                others = [chosen[k] for k in range(n) if k != i and k != j]
                if not is_coprime_to_set(prod, others):
                    continue
                
                gain = weights[prod] - weights[a1] - weights[a2]
                if gain > 1e-12:
                    chosen[i] = prod
                    chosen.pop(j)
                    chosen_set.remove(a1)
                    chosen_set.remove(a2)
                    chosen_set.add(prod)
                    current_score += gain
                    improved = True
                    found_swap = True
                    if current_score > best_score:
                        best_score = current_score
                        best_chosen = list(chosen)
                    break
    
    return best_chosen
# EVOLVE-BLOCK-END
