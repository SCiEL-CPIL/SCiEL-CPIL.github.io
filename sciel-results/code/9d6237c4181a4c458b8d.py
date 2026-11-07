"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from typing import List, Set, Tuple

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    # Precompute prime factors for all numbers 2..256
    def get_prime_factors(x):
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors
    
    prime_factors = {}
    for v in range(2, 257):
        prime_factors[v] = get_prime_factors(v)
    
    # Greedy initialization: select numbers in descending order
    def greedy_init():
        selected = set()
        used_primes = set()
        for val in range(256, 1, -1):
            pf = prime_factors[val]
            if not pf & used_primes:
                selected.add(val)
                used_primes |= pf
        return selected
    
    best_solution = greedy_init()
    best_score = sum(best_solution)
    
    # Simulated annealing
    def can_add(selected: Set[int], val: int) -> bool:
        pf = prime_factors[val]
        for s in selected:
            if pf & prime_factors[s]:
                return False
        return True
    
    def get_conflicts(selected: Set[int], val: int) -> Set[int]:
        """Return elements in selected that conflict with val."""
        pf = prime_factors[val]
        conflicts = set()
        for s in selected:
            if pf & prime_factors[s]:
                conflicts.add(s)
        return conflicts
    
    # Simulated annealing
    T = 1000.0
    cooling = 0.9995
    iterations = 0
    max_iterations = 200000
    
    current_solution = set(best_solution)
    current_score = sum(current_solution)
    
    while iterations < max_iterations:
        if time.time() > deadline - 5:
            break
        
        iterations += 1
        T *= cooling
        
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random number
            candidates = [v for v in range(2, 257) if v not in current_solution]
            if not candidates:
                continue
            val = random.choice(candidates)
            if can_add(current_solution, val):
                new_solution = current_solution | {val}
                new_score = current_score + val
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 0.001)):
                    current_solution = new_solution
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = set(current_solution)
        
        elif move_type < 0.6:
            # Remove a random number
            if not current_solution:
                continue
            val = random.choice(list(current_solution))
            new_solution = current_solution - {val}
            new_score = current_score - val
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 0.001)):
                current_solution = new_solution
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_solution = set(current_solution)
        
        elif move_type < 0.85:
            # 1-for-1 swap
            if not current_solution:
                continue
            val_out = random.choice(list(current_solution))
            candidates = [v for v in range(2, 257) if v not in current_solution]
            if not candidates:
                continue
            val_in = random.choice(candidates)
            
            # Check if val_in can replace val_out
            temp_solution = (current_solution - {val_out}) | {val_in}
            valid = True
            for s in temp_solution:
                if s == val_in:
                    continue
                if prime_factors[s] & prime_factors[val_in]:
                    valid = False
                    break
            
            if valid:
                new_score = current_score - val_out + val_in
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 0.001)):
                    current_solution = temp_solution
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = set(current_solution)
        
        else:
            # 2-for-1 swap: remove two, add one
            if len(current_solution) < 2:
                continue
            vals_out = random.sample(list(current_solution), 2)
            candidates = [v for v in range(2, 257) if v not in current_solution]
            if not candidates:
                continue
            val_in = random.choice(candidates)
            
            temp_solution = (current_solution - set(vals_out)) | {val_in}
            valid = True
            for s in temp_solution:
                if s == val_in:
                    continue
                if prime_factors[s] & prime_factors[val_in]:
                    valid = False
                    break
            
            if valid:
                new_score = current_score - sum(vals_out) + val_in
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 0.001)):
                    current_solution = temp_solution
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = set(current_solution)
    
    # Local search: try all possible 1-for-1 swaps
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for val_out in list(best_solution):
            for val_in in range(2, 257):
                if val_in in best_solution:
                    continue
                temp_solution = (best_solution - {val_out}) | {val_in}
                valid = True
                for s in temp_solution:
                    if s == val_in:
                        continue
                    if prime_factors[s] & prime_factors[val_in]:
                        valid = False
                        break
                if valid:
                    new_score = sum(temp_solution)
                    if new_score > best_score:
                        best_solution = temp_solution
                        best_score = new_score
                        improved = True
                        break
            if improved:
                break
    
    # Try to add 1 (it's coprime with everything)
    if 1 not in best_solution:
        best_solution.add(1)
        best_score += 1
    
    # Return as list of ints
    result = [int(x) for x in sorted(best_solution, reverse=True)]
    return result
# EVOLVE-BLOCK-END
