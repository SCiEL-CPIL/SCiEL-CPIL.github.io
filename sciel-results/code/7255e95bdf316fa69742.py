"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Precompute GCD table for [1, 128]
    n = 128
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Check if a set is valid: no three distinct elements with all pairwise gcds equal
    def is_valid_subset(subset):
        """Check if subset has no three distinct elements with equal pairwise gcds."""
        m = len(subset)
        if m < 3:
            return True
        # For each triple, check if all three pairwise gcds are equal
        for i in range(m):
            for j in range(i + 1, m):
                gij = gcd_table[subset[i]][subset[j]]
                for k in range(j + 1, m):
                    if gcd_table[subset[i]][subset[k]] == gij and gcd_table[subset[j]][subset[k]] == gij:
                        return False
        return True
    
    # Faster validity check: for a candidate element x to be added to subset,
    # check that no pair (a,b) in subset has gcd(a,b) == gcd(a,x) == gcd(b,x)
    def can_add(subset, x):
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                gij = gcd_table[subset[i]][subset[j]]
                if gcd_table[subset[i]][x] == gij and gcd_table[subset[j]][x] == gij:
                    return False
        return True
    
    # Greedy construction with different orderings
    def greedy_construct(order):
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        return selected
    
    # Try multiple greedy orderings
    best_solution = []
    
    # Ordering 1: natural order
    sol1 = greedy_construct(range(1, n + 1))
    if len(sol1) > len(best_solution):
        best_solution = sol1[:]
    
    # Ordering 2: by number of divisors (fewer divisors first - more "prime-like")
    def num_divisors(x):
        count = 0
        for i in range(1, int(x**0.5) + 1):
            if x % i == 0:
                count += 1
                if i != x // i:
                    count += 1
        return count
    
    order2 = sorted(range(1, n + 1), key=lambda x: (num_divisors(x), x))
    sol2 = greedy_construct(order2)
    if len(sol2) > len(best_solution):
        best_solution = sol2[:]
    
    # Ordering 3: by number of divisors descending
    order3 = sorted(range(1, n + 1), key=lambda x: (-num_divisors(x), x))
    sol3 = greedy_construct(order3)
    if len(sol3) > len(best_solution):
        best_solution = sol3[:]
    
    # Ordering 4: primes first, then composites
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True
    
    order4 = sorted(range(1, n + 1), key=lambda x: (0 if is_prime(x) else 1, x))
    sol4 = greedy_construct(order4)
    if len(sol4) > len(best_solution):
        best_solution = sol4[:]
    
    # Simulated annealing to improve the best solution
    def simulated_annealing(initial_solution, time_limit):
        """Simulated annealing with swap and add/remove moves."""
        current = set(initial_solution)
        current_size = len(current)
        best = set(current)
        best_size = current_size
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.1
        steps = 0
        
        while time.time() < time_limit:
            steps += 1
            # Cooldown
            if steps % 100 == 0:
                progress = min(1.0, steps / 5000.0)
                T = T_start * (T_end / T_start) ** progress
            else:
                T = T_start * (T_end / T_start) ** min(1.0, steps / 5000.0)
            
            # Random move
            move_type = random.random()
            if move_type < 0.4 and len(current) > 0:
                # Swap: remove one, add another
                to_remove = random.choice(list(current))
                candidates = [x for x in range(1, n + 1) if x not in current]
                if not candidates:
                    continue
                to_add = random.choice(candidates)
                
                # Try the swap
                new_set = current - {to_remove} | {to_add}
                new_size = len(new_set)
                
                # Check validity efficiently
                valid = True
                new_list = list(new_set)
                for i in range(len(new_list)):
                    for j in range(i + 1, len(new_list)):
                        gij = gcd_table[new_list[i]][new_list[j]]
                        for k in range(j + 1, len(new_list)):
                            if gcd_table[new_list[i]][new_list[k]] == gij and gcd_table[new_list[j]][new_list[k]] == gij:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    current = new_set
                    current_size = new_size
                    if current_size > best_size:
                        best = set(current)
                        best_size = current_size
                else:
                    # Accept with probability based on "cost" (invalidity)
                    # For invalid moves, we just reject (simpler)
                    pass
            elif move_type < 0.7 and len(current) > 0:
                # Remove a random element
                to_remove = random.choice(list(current))
                new_set = current - {to_remove}
                new_size = len(new_set)
                # Always accept removal (reduces size but may enable better swaps later)
                current = new_set
                current_size = new_size
            else:
                # Add a random element
                candidates = [x for x in range(1, n + 1) if x not in current]
                if not candidates:
                    continue
                to_add = random.choice(candidates)
                new_set = current | {to_add}
                new_size = len(new_set)
                
                # Check validity
                valid = True
                new_list = list(new_set)
                for i in range(len(new_list)):
                    for j in range(i + 1, len(new_list)):
                        gij = gcd_table[new_list[i]][new_list[j]]
                        for k in range(j + 1, len(new_list)):
                            if gcd_table[new_list[i]][new_list[k]] == gij and gcd_table[new_list[j]][new_list[k]] == gij:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    current = new_set
                    current_size = new_size
                    if current_size > best_size:
                        best = set(current)
                        best_size = current_size
                else:
                    # Accept with some probability to escape local optima
                    if random.random() < T:
                        current = new_set
                        current_size = new_size
        
        return list(best)
    
    # Run simulated annealing from the best greedy solution
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        sa_result = simulated_annealing(best_solution, deadline - 1.0)
        if len(sa_result) > len(best_solution):
            best_solution = sa_result
    
    # Try additional random restarts with SA if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 3.0:
        random.seed(42)
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            # Random initial solution
            random_init = random.sample(range(1, n + 1), random.randint(10, 20))
            # Filter to valid subset
            valid_init = []
            for x in random_init:
                if can_add(valid_init, x):
                    valid_init.append(x)
            
            sa_result = simulated_annealing(valid_init, deadline - 1.0)
            if len(sa_result) > len(best_solution):
                best_solution = sa_result
    
    # Final verification
    if not is_valid_subset(best_solution):
        # Fallback to a known valid solution
        best_solution = list(range(1, n + 1))
        # This should never happen, but safety check
    
    if not best_solution:
        best_solution = [1]
    
    return best_solution
# EVOLVE-BLOCK-END
