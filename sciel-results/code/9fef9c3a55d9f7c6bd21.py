"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    n = 256
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Precompute prime factors for each number 2..n
    prime_factors = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if not prime_factors[i]:  # i is prime
            for j in range(i, n + 1, i):
                prime_factors[j].append(i)
    
    # Get list of all numbers 2..n with their prime factors
    numbers = list(range(n, 1, -1))  # descending order for greedy
    
    # Helper: check if adding num to current set is valid
    def can_add(num, used_primes):
        for p in prime_factors[num]:
            if p in used_primes:
                return False
        return True
    
    # Helper: try to improve current solution by local search
    def local_search(solution, used_primes):
        """Try to improve by swapping or adding numbers."""
        improved = True
        while improved:
            improved = False
            sol_sum = sum(solution)
            sol_set = set(solution)
            
            # Try to add any number not in solution
            for num in range(n, 1, -1):
                if num in sol_set:
                    continue
                if can_add(num, used_primes):
                    solution.append(num)
                    used_primes.update(prime_factors[num])
                    sol_set.add(num)
                    sol_sum += num
                    improved = True
            
            # Try to remove one number and add two that conflict with it
            for i in range(len(solution)):
                num_remove = solution[i]
                # Temporarily remove
                temp_used = set()
                for p in prime_factors[num_remove]:
                    if p in used_primes and used_primes[p] == 1:
                        # This prime is only used by num_remove
                        pass
                    elif p in used_primes:
                        temp_used.add(p)
                
                # Actually, let's do a proper swap search
                # Remove num_remove, then try to add other numbers
                removed_primes = set()
                for p in prime_factors[num_remove]:
                    if p in used_primes:
                        # Check if any other number in solution uses this prime
                        count = 0
                        for other in solution:
                            if other == num_remove:
                                continue
                            if p in prime_factors[other]:
                                count += 1
                        if count == 0:
                            removed_primes.add(p)
                
                # Try to find replacements
                best_replacement_sum = 0
                best_replacements = []
                
                # Try adding single numbers
                for num_add in range(n, 1, -1):
                    if num_add in sol_set or num_add == num_remove:
                        continue
                    # Check if num_add can be added after removing num_remove
                    # The primes that become available are removed_primes
                    # But also, num_remove's primes that are still used by others remain
                    new_used = set(used_primes - removed_primes)
                    # Wait, this is getting complex. Let me simplify.
                    # Actually, let's just do a simpler approach:
                    # Remove num_remove, then greedily add the best possible numbers
                    pass
                
                # Simpler approach: just try removing and see if we can add more
                # This is getting too complex for local search. Let me use a different strategy.
                break
        
        return solution, used_primes
    
    # Better approach: use simulated annealing or iterated local search
    # Start with greedy solution
    def greedy_solution():
        used_primes = set()
        solution = [1]
        for num in range(n, 1, -1):
            if can_add(num, used_primes):
                solution.append(num)
                for p in prime_factors[num]:
                    used_primes.add(p)
        return solution, used_primes
    
    # Initial greedy solution
    best_solution, best_used = greedy_solution()
    best_sum = sum(best_solution)
    
    # Iterated local search with random restarts
    random.seed(42)
    
    def perturbation(solution, used_primes, strength=1):
        """Create a perturbed version of the solution."""
        sol = list(solution)
        used = set(used_primes)
        
        # Remove 1-3 random elements (not 1)
        removable = [x for x in sol if x != 1]
        if not removable:
            return sol, used
        
        k = min(strength, len(removable))
        to_remove = random.sample(removable, k)
        for num in to_remove:
            sol.remove(num)
            # Remove primes that are no longer used
            for p in prime_factors[num]:
                if p in used:
                    # Check if any other number in sol uses this prime
                    count = 0
                    for other in sol:
                        if p in prime_factors[other]:
                            count += 1
                    if count == 0:
                        used.discard(p)
        
        # Now greedily add back the best possible numbers
        for num in range(n, 1, -1):
            if num in sol:
                continue
            if can_add(num, used):
                sol.append(num)
                for p in prime_factors[num]:
                    used.add(p)
        
        return sol, used
    
    def hill_climb(solution, used_primes):
        """Improve solution by adding any valid numbers."""
        sol = list(solution)
        used = set(used_primes)
        sol_set = set(sol)
        
        changed = True
        while changed:
            changed = False
            for num in range(n, 1, -1):
                if num in sol_set:
                    continue
                if can_add(num, used):
                    sol.append(num)
                    sol_set.add(num)
                    for p in prime_factors[num]:
                        used.add(p)
                    changed = True
        
        return sol, used
    
    # Iterated local search
    current_solution, current_used = greedy_solution()
    current_sum = sum(current_solution)
    
    num_iterations = 0
    while time.time() < deadline:
        num_iterations += 1
        
        # Perturb
        pert_strength = 1 + (num_iterations % 3)  # 1, 2, or 3
        sol, used = perturbation(current_solution, current_used, pert_strength)
        
        # Hill climb
        sol, used = hill_climb(sol, used)
        new_sum = sum(sol)
        
        # Accept if better
        if new_sum > current_sum:
            current_solution = sol
            current_used = used
            current_sum = new_sum
        
        # Update best
        if current_sum > best_sum:
            best_solution = list(current_solution)
            best_used = set(current_used)
            best_sum = current_sum
    
    # Final hill climb on best solution
    best_solution, best_used = hill_climb(best_solution, best_used)
    
    # Ensure return type is list of Python ints
    result = [int(x) for x in best_solution]
    return result
# EVOLVE-BLOCK-END
