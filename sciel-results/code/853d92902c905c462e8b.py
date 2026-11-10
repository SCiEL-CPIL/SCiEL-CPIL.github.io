import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s
    
    # Precompute smallest prime factor for numbers up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, isqrt(MAX_SUM) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def prime_factors(n):
        """Return set of prime factors of n."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def evaluate(numbers):
        """Return the set of all distinct prime factors in pairwise sums."""
        all_primes = set()
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                s = numbers[i] + numbers[j]
                all_primes.update(prime_factors(s))
        return all_primes
    
    # Initial solution: seed
    best_numbers = list(range(1, 13))
    best_primes = evaluate(best_numbers)
    best_score = len(best_primes)
    
    # Try to improve using local search
    # Strategy: try to find a set of 12 numbers whose pairwise sums use few primes
    
    # First, try a greedy construction with a target prime set
    # Try small prime sets and see if we can find 12 numbers
    
    # Generate candidate numbers: smooth numbers are good candidates
    # Let's generate numbers that are products of small primes
    
    # Try different strategies:
    
    # Strategy 1: Greedy with prime set constraint
    # For a target prime set P, find 12 numbers where all pairwise sums are P-smooth
    
    def is_p_smooth(n, prime_set):
        """Check if n has only prime factors in prime_set."""
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1
    
    # Try small prime sets
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # For each subset of small primes (starting small), try to find 12 numbers
    # such that all pairwise sums are smooth with respect to that set
    
    # Generate candidate numbers up to 10000
    # A number is "compatible" with prime set P if it can form valid sums
    
    # Let's use a different approach: local search
    
    # Start with current best, try swaps
    import copy
    
    current = list(best_numbers)
    current_primes = evaluate(current)
    current_score = len(current_primes)
    
    # Local search: try swapping one element at a time
    # For each position, try replacing with a number that reduces prime count
    
    # Generate a pool of candidate numbers to swap in
    # Focus on numbers that are smooth or have few prime factors
    candidate_pool = list(range(1, 10001))
    random.shuffle(candidate_pool)
    
    # Precompute prime factors for candidates to speed up evaluation
    # Actually, let's just do direct evaluation but be smart about it
    
    # Better: incremental evaluation
    def eval_with_replacement(numbers, pos, new_val):
        """Evaluate prime set if we replace numbers[pos] with new_val."""
        all_primes = set()
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                if i == pos:
                    s = new_val + numbers[j]
                elif j == pos:
                    s = numbers[i] + new_val
                else:
                    s = numbers[i] + numbers[j]
                all_primes.update(prime_factors(s))
        return all_primes
    
    # Simulated annealing / hill climbing
    best_found = list(best_numbers)
    best_found_score = best_score
    
    # Try multiple random restarts with local search
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            current = list(best_numbers)
        else:
            # Random start with some structure
            # Pick numbers that are smooth
            smooth_candidates = []
            for n in range(1, 10001):
                pf = prime_factors(n)
                if len(pf) <= 3:  # at most 3 distinct prime factors
                    smooth_candidates.append(n)
            random.shuffle(smooth_candidates)
            if len(smooth_candidates) >= 12:
                current = smooth_candidates[:12]
            else:
                current = random.sample(range(1, 10001), 12)
        
        current_primes = evaluate(current)
        current_score = len(current_primes)
        
        if current_score < best_found_score:
            best_found_score = current_score
            best_found = list(current)
        
        # Local search: try to improve
        no_improve_count = 0
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            # Try swapping a random position with a random candidate
            pos = random.randint(0, 11)
            # Choose a candidate not already in current
            new_val = random.randint(1, 10000)
            if new_val in current:
                continue
            
            new_primes = eval_with_replacement(current, pos, new_val)
            new_score = len(new_primes)
            
            if new_score <= current_score:
                current[pos] = new_val
                current_primes = new_primes
                current_score = new_score
                no_improve_count = 0
                
                if current_score < best_found_score:
                    best_found_score = current_score
                    best_found = list(current)
            else:
                no_improve_count += 1
                # Simulated annealing: accept worse with some probability
                if random.random() < 0.1:
                    current[pos] = new_val
                    current_primes = new_primes
                    current_score = new_score
        
        if best_found_score <= 3:
            break
    
    return best_found
