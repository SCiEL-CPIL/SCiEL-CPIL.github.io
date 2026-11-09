"""Optimized search for maximizing tau_perp(n) over n <= 10^18."""
import math
import random
import time
import heapq
from itertools import combinations

def generate_divisors_squarefree(primes):
    """Generate all divisors of a squarefree number given its prime list."""
    n = len(primes)
    divisors = [1]
    for p in primes:
        new_divs = [d * p for d in divisors]
        divisors.extend(new_divs)
    return divisors

def count_coprime_adjacent(divisors):
    """Count adjacent pairs with gcd 1 in a sorted list of divisors."""
    divs = sorted(divisors)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def evaluate_squarefree(primes):
    """Evaluate tau_perp for a squarefree number defined by prime set."""
    # Check product <= 10^18
    prod = 1
    for p in primes:
        prod *= p
        if prod > 10**18:
            return 0
    divs = generate_divisors_squarefree(primes)
    return count_coprime_adjacent(divs)

def get_primes_up_to(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = 130  # safety margin before 150s kill
    
    # Get a pool of primes
    primes_pool = get_primes_up_to(500)
    
    # Start with the first 15 primes (max squarefree within 10^18)
    first_15 = primes_pool[:15]
    best_score = evaluate_squarefree(first_15)
    best_primes = list(first_15)
    
    # Simulated annealing over prime subsets
    # State: a set of primes (represented as indices into primes_pool)
    # We want to maximize the score
    
    current = list(range(len(first_15)))  # indices of first 15 primes
    current_score = best_score
    
    # Temperature schedule
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    
    # Precompute: for each prime index, its value
    prime_vals = [primes_pool[i] for i in range(len(primes_pool))]
    
    def product_of_indices(indices):
        prod = 1
        for idx in indices:
            prod *= prime_vals[idx]
            if prod > 10**18:
                return None
        return prod
    
    def score_from_indices(indices):
        """Compute score for a set of prime indices."""
        prod = product_of_indices(indices)
        if prod is None or prod > 10**18:
            return 0
        prime_set = [prime_vals[i] for i in indices]
        return evaluate_squarefree(prime_set)
    
    # Simulated annealing
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        if T < min_T:
            break
        
        # Random move: swap one prime or add/remove one
        move_type = random.random()
        
        if move_type < 0.7:
            # Swap: remove one, add another
            if len(current) < 2:
                continue
            remove_idx = random.choice(current)
            # Choose a candidate to add (not already in set)
            candidates = [i for i in range(len(primes_pool)) if i not in current]
            if not candidates:
                continue
            add_idx = random.choice(candidates)
            
            new_current = [i for i in current if i != remove_idx]
            new_current.append(add_idx)
            new_score = score_from_indices(new_current)
        elif move_type < 0.85:
            # Add a prime
            candidates = [i for i in range(len(primes_pool)) if i not in current]
            if not candidates:
                continue
            add_idx = random.choice(candidates)
            new_current = current + [add_idx]
            new_score = score_from_indices(new_current)
        else:
            # Remove a prime
            if len(current) < 2:
                continue
            remove_idx = random.choice(current)
            new_current = [i for i in current if i != remove_idx]
            new_score = score_from_indices(new_current)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current = new_current
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_primes = [prime_vals[i] for i in current]
        
        T *= cooling
    
    # Final greedy polish
    # Try to improve by swapping individual primes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_primes)
        for p_out in list(current_set):
            for p_in in primes_pool:
                if p_in in current_set:
                    continue
                new_set = list(current_set - {p_out}) + [p_in]
                new_set.sort()
                prod = 1
                valid = True
                for p in new_set:
                    prod *= p
                    if prod > 10**18:
                        valid = False
                        break
                if not valid:
                    continue
                s = evaluate_squarefree(new_set)
                if s > best_score:
                    best_score = s
                    best_primes = new_set
                    current_set = set(new_set)
                    improved = True
                    break
            if improved:
                break
    
    # Construct the final number
    result = 1
    for p in best_primes:
        result *= p
    
    if result < 2:
        result = 2
    if result > 10**18:
        # Fallback to seed
        result = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    return result

# Run at import time to verify
if __name__ == "__main__":
    n = construct()
    print(f"n = {n}")
    print(f"Score: {evaluate_squarefree([p for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]])}")
