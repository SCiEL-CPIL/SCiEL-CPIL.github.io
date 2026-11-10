import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for i in range(2, max_mod + 1):
        for j in range(2, max_mod + 1):
            gcd_table[i][j] = gcd(i, j)
    
    def num_divisors(n):
        count = 0
        i = 2
        while i * i <= n:
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
            i += 1
        return count + 1
    
    # Sort moduli by descending divisor count (more constrained first)
    moduli_desc = sorted(moduli, key=num_divisors, reverse=True)
    moduli_asc = sorted(moduli, key=num_divisors)
    
    def check_valid(selected, r, m):
        """Check if (r, m) is compatible with all in selected."""
        for r_j, m_j in selected:
            g = gcd_table[m][m_j]
            if r % g == r_j % g:
                return False
        return True
    
    def find_valid_residue(selected, m):
        """Find any valid residue for m given selected."""
        for r in range(m):
            if check_valid(selected, r, m):
                return r
        return None
    
    def greedy(order):
        """Greedy: process moduli in order, add if possible."""
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(selected, m)
            if r is not None:
                selected.append((r, m))
        return selected
    
    def local_search(selected):
        """Improve solution by trying to add missing moduli or swap."""
        best = selected[:]
        best_len = len(selected)
        selected_moduli = set(m for r, m in best)
        
        # Phase 1: Try to add each missing modulus
        for m in moduli:
            if time.time() > deadline:
                break
            if m in selected_moduli:
                continue
            r = find_valid_residue(best, m)
            if r is not None:
                best.append((r, m))
                best_len += 1
                selected_moduli.add(m)
        
        # Phase 2: Try swaps - remove one, add two
        for i in range(len(best)):
            if time.time() > deadline:
                break
            removed = best[i]
            without_i = best[:i] + best[i+1:]
            without_i_moduli = set(m for r, m in without_i)
            
            # Try to add two moduli
            added = 0
            new_best = without_i[:]
            for m in moduli:
                if time.time() > deadline:
                    break
                if m in without_i_moduli or m == removed[1]:
                    continue
                r = find_valid_residue(new_best, m)
                if r is not None:
                    new_best.append((r, m))
                    without_i_moduli.add(m)
                    added += 1
                    if added >= 2:
                        break
            
            if len(new_best) > best_len:
                best = new_best
                best_len = len(new_best)
                selected_moduli = set(m for r, m in best)
        
        return best
    
    best_solution = []
    best_len = 0
    
    # Run multiple strategies
    strategies_done = 0
    while time.time() < deadline - 5:
        strategies_done += 1
        
        if strategies_done % 4 == 1:
            # Greedy with descending divisor order
            sol = greedy(moduli_desc)
        elif strategies_done % 4 == 2:
            # Greedy with ascending divisor order
            sol = greedy(moduli_asc)
        elif strategies_done % 4 == 3:
            # Greedy with random order
            random_order = moduli[:]
            random.shuffle(random_order)
            sol = greedy(random_order)
        else:
            # Perturb current best
            if best_solution:
                sol = best_solution[:]
                # Randomly remove 0-2 elements
                n_remove = random.randint(0, min(2, len(sol) - 1)) if len(sol) > 1 else 0
                for _ in range(n_remove):
                    idx = random.randint(0, len(sol) - 1)
                    sol.pop(idx)
                # Re-add missing moduli
                sol_moduli = set(m for r, m in sol)
                missing = [m for m in moduli if m not in sol_moduli]
                random.shuffle(missing)
                for m in missing:
                    if time.time() > deadline - 5:
                        break
                    r = find_valid_residue(sol, m)
                    if r is not None:
                        sol.append((r, m))
            else:
                sol = greedy(moduli_desc)
        
        # Apply local search
        sol = local_search(sol)
        
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
    
    if not best_solution:
        best_solution = [(0, 2)]
    
    return [[r, m] for r, m in best_solution]
