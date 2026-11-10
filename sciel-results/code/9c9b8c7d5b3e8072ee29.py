import random
import time
import math
from math import gcd

def construct():
    start_time = time.time()
    max_time = 55.0  # Safety margin before 60s hard limit
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    
    # Precompute gcd table for fast lookup
    gcd_table = [[0] * (max_modulus + 1) for _ in range(max_modulus + 1)]
    for i in range(2, max_modulus + 1):
        for j in range(2, max_modulus + 1):
            gcd_table[i][j] = gcd(i, j)
    
    # Precompute number of divisors for sorting
    def num_divisors(n):
        count = 1
        i = 2
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i * i != n:
                    count += 1
            i += 1
        return count
    
    div_count = {m: num_divisors(m) for m in moduli}
    
    def check_valid(selected, r, m):
        """Check if (r, m) is compatible with all in selected."""
        for r_j, m_j in selected:
            g = gcd_table[m][m_j]
            if r % g == r_j % g:
                return False
        return True
    
    def find_valid_residues(selected, m):
        """Find all valid residues for m given selected classes."""
        valid = []
        for r in range(m):
            if check_valid(selected, r, m):
                valid.append(r)
        return valid
    
    def greedy(order, randomize=False):
        """Greedy: process moduli in order, add if possible."""
        selected = []
        selected_moduli = set()
        for m in order:
            if time.time() > start_time + max_time:
                break
            if m in selected_moduli:
                continue
            valid = find_valid_residues(selected, m)
            if valid:
                r = random.choice(valid) if randomize else valid[0]
                selected.append((r, m))
                selected_moduli.add(m)
        return selected
    
    def local_search_fill(selected, time_limit):
        """Try to add missing moduli in random order."""
        best = selected[:]
        selected_moduli = set(m for r, m in best)
        
        missing = [m for m in moduli if m not in selected_moduli]
        random.shuffle(missing)
        
        for m in missing:
            if time.time() > time_limit:
                break
            valid = find_valid_residues(best, m)
            if valid:
                r = random.choice(valid)
                best.append((r, m))
                selected_moduli.add(m)
        
        return best
    
    def remove_one_add_two(sol, time_limit):
        """Try to improve by removing one element and adding two new ones."""
        sol = [pair[:] for pair in sol]
        n = len(sol)
        improved = False
        
        for i in range(n):
            if time.time() > time_limit:
                break
            
            temp_sol = sol[:i] + sol[i+1:]
            temp_moduli = set(m for r, m in temp_sol)
            
            # Try to add two new elements
            new_elements = []
            for m in moduli:
                if m in temp_moduli or m in [p[1] for p in new_elements]:
                    continue
                valid = find_valid_residues(temp_sol + new_elements, m)
                if valid:
                    r = random.choice(valid)
                    new_elements.append([r, m])
                    if len(new_elements) == 2:
                        break
            
            if len(new_elements) == 2:
                sol = temp_sol + new_elements
                improved = True
                break
        
        return sol, improved
    
    # Initial greedy with different orderings to get baselines
    best_solution = []
    best_size = 0
    
    # Try several initial orderings
    orderings = [
        moduli[:],  # ascending
        moduli[::-1],  # descending
        sorted(moduli, key=lambda m: div_count[m], reverse=True),  # most divisors first
        sorted(moduli, key=lambda m: div_count[m]),  # fewest divisors first
    ]
    
    for order in orderings:
        if time.time() > start_time + max_time:
            break
        sol = greedy(order, randomize=False)
        sol = local_search_fill(sol, start_time + max_time - 10.0)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = [pair[:] for pair in sol]
    
    # Randomized local search / greedy with restarts
    while time.time() < start_time + max_time - 5.0:
        random.shuffle(moduli)
        current_set = greedy(moduli, randomize=True)
        current_set = local_search_fill(current_set, start_time + max_time - 5.0)
        
        # Apply remove-one-add-two improvement
        if time.time() < start_time + max_time - 10.0:
            current_set, improved = remove_one_add_two(current_set, start_time + max_time - 10.0)
            if improved:
                current_set = local_search_fill(current_set, start_time + max_time - 5.0)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_solution = [pair[:] for pair in current_set]
            
        # Also try local search on the best solution found so far
        if time.time() < start_time + max_time - 10.0:
            improved_best, improved_flag = remove_one_add_two(best_solution, start_time + max_time - 10.0)
            if improved_flag:
                improved_best = local_search_fill(improved_best, start_time + max_time - 5.0)
                if len(improved_best) > best_size:
                    best_size = len(improved_best)
                    best_solution = [pair[:] for pair in improved_best]
    
    if not best_solution:
        best_solution = [[0, 2]]
    
    return [[r, m] for r, m in best_solution]
