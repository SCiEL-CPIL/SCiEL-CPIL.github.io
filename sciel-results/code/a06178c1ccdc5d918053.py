import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precomputation
    # Primes below 512
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(512)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)
    
    # For each a in 1..511, compute:
    # - weight: 1/(512-a) as Fraction for exact arithmetic
    # - prime_factors: set of prime indices dividing a
    weights_frac = [Fraction(0, 1)] * 512
    prime_factor_sets = [set() for _ in range(512)]
    
    for a in range(1, 512):
        weights_frac[a] = Fraction(1, 512 - a)
        n = a
        for i, p in enumerate(primes):
            if p * p > n:
                break
            if n % p == 0:
                prime_factor_sets[a].add(i)
                while n % p == 0:
                    n //= p
        if n > 1:
            prime_factor_sets[a].add(prime_to_idx[n])
    
    # 2. Build candidate list sorted by weight descending
    # Higher a means smaller (512-a), so larger weight
    candidates = list(range(511, 0, -1))  # Already sorted by weight descending
    
    # 3. Greedy initialization
    def greedy_init():
        chosen = []
        used_primes = set()
        for a in candidates:
            if prime_factor_sets[a].isdisjoint(used_primes):
                chosen.append(a)
                used_primes.update(prime_factor_sets[a])
        return chosen
    
    def calculate_score(chosen):
        total = Fraction(0, 1)
        for a in chosen:
            total += weights_frac[a]
        return total
    
    # 4. Local search: try removing one element and re-greedy filling
    def local_search(chosen):
        current_score = calculate_score(chosen)
        chosen_set = set(chosen)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for a in list(chosen):
                if time.time() >= deadline - 5.0:
                    break
                
                # Remove a
                temp_chosen = [x for x in chosen if x != a]
                temp_set = chosen_set - {a}
                
                # Compute used primes from temp_chosen
                used_primes = set()
                for x in temp_chosen:
                    used_primes.update(prime_factor_sets[x])
                
                # Re-greedy fill
                new_chosen = list(temp_chosen)
                new_score = current_score - weights_frac[a]
                
                for b in candidates:
                    if b in temp_set:
                        continue
                    if prime_factor_sets[b].isdisjoint(used_primes):
                        new_chosen.append(b)
                        used_primes.update(prime_factor_sets[b])
                        new_score += weights_frac[b]
                
                if new_score > current_score:
                    chosen = new_chosen
                    chosen_set = set(chosen)
                    current_score = new_score
                    improved = True
                    break
        
        return chosen, current_score
    
    # 5. 2-for-1 swap: remove two elements, add up to two new ones
    def two_for_one_search(chosen):
        current_score = calculate_score(chosen)
        chosen_set = set(chosen)
        
        # Try all pairs to remove
        for i in range(len(chosen)):
            if time.time() >= deadline - 5.0:
                break
            for j in range(i + 1, len(chosen)):
                if time.time() >= deadline - 5.0:
                    break
                
                a1, a2 = chosen[i], chosen[j]
                temp_chosen = [x for x in chosen if x != a1 and x != a2]
                temp_set = chosen_set - {a1, a2}
                
                used_primes = set()
                for x in temp_chosen:
                    used_primes.update(prime_factor_sets[x])
                
                new_chosen = list(temp_chosen)
                new_score = current_score - weights_frac[a1] - weights_frac[a2]
                
                # Greedy add up to 2 new elements
                added = 0
                for b in candidates:
                    if b in temp_set:
                        continue
                    if prime_factor_sets[b].isdisjoint(used_primes):
                        new_chosen.append(b)
                        used_primes.update(prime_factor_sets[b])
                        new_score += weights_frac[b]
                        added += 1
                        if added >= 2:
                            break
                
                if new_score > current_score:
                    return new_chosen, new_score
        
        return chosen, current_score
    
    # 6. Main loop
    best_chosen = greedy_init()
    best_score = calculate_score(best_chosen)
    
    # Apply local search
    refined, refined_score = local_search(best_chosen)
    if refined_score > best_score:
        best_chosen = refined
        best_score = refined_score
    
    # Apply 2-for-1 search
    refined2, refined_score2 = two_for_one_search(best_chosen)
    if refined_score2 > best_score:
        best_chosen = refined2
        best_score = refined_score2
    
    # More local search passes
    while time.time() < deadline - 10.0:
        refined, refined_score = local_search(best_chosen)
        if refined_score > best_score:
            best_chosen = refined
            best_score = refined_score
        
        refined2, refined_score2 = two_for_one_search(best_chosen)
        if refined_score2 > best_score:
            best_chosen = refined2
            best_score = refined_score2
        else:
            # Check if we've converged
            if not any(True for _ in []):  # Placeholder for convergence check
                pass
    
    return best_chosen
