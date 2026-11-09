import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin from 150s limit
    
    # Precompute prime factors for 1..511
    # factors[i] is a set of prime factors of i
    factors = [set() for _ in range(512)]
    for i in range(2, 512):
        if not factors[i]:
            for j in range(i, 512, i):
                factors[j].add(i)
    
    # Precompute values
    values = [0.0] * 512
    for i in range(1, 512):
        values[i] = 1.0 / (512.0 - i)
        
    # Sort candidates by value descending
    candidates = list(range(511, 0, -1))
    
    # Greedy initialization
    chosen = []
    used_primes = set()
    for a in candidates:
        if time.time() > deadline:
            break
        if all(p not in used_primes for p in factors[a]):
            chosen.append(a)
            used_primes.update(factors[a])
            
    best_sum = sum(values[a] for a in chosen)
    best_solution = list(chosen)
    
    # Local Search: Delete and Repair
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Sort current solution by value descending to try removing "expensive" ones first?
        # Or just iterate through all.
        for i in range(len(chosen)):
            if time.time() > deadline - 5.0:
                break
            
            a_to_remove = chosen[i]
            rest = [x for x in chosen if x != a_to_remove]
            
            # Calculate primes used by rest
            rest_primes = set()
            for x in rest:
                rest_primes.update(factors[x])
                
            # Find candidates that can be added
            # A candidate c is valid if:
            # 1. c is not in rest
            # 2. factors[c] are disjoint from rest_primes
            # 3. factors[c] are disjoint from factors of other candidates we pick
            
            # We want to pick a set S from valid candidates to maximize sum
            # Heuristic: Greedy on valid candidates sorted by value desc
            
            valid_candidates = []
            for c in range(511, 0, -1):
                if c in rest:
                    continue
                if c == a_to_remove:
                    continue
                # Check if c's primes are disjoint from rest_primes
                if factors[c].isdisjoint(rest_primes):
                    valid_candidates.append(c)
                    
            # Greedy pick from valid_candidates
            new_picks = []
            new_primes = set()
            for c in valid_candidates:
                if factors[c].isdisjoint(new_primes):
                    new_picks.append(c)
                    new_primes.update(factors[c])
                    
            new_solution = rest + new_picks
            new_sum = sum(values[x] for x in new_solution)
            
            if new_sum > best_sum + 1e-9:
                best_sum = new_sum
                best_solution = new_solution
                chosen = new_solution
                improved = True
                # Restart the loop to check again with the new solution
                break
                
    return best_solution
