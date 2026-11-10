import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin for safety
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcd table for fast lookup
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
        """Find any valid residue for m given selected classes."""
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
        """Improve solution by trying to add missing moduli. Fast greedy fill."""
        best = selected[:]
        selected_moduli = set(m for r, m in best)
        
        # Try to add each missing modulus in random order to break ties
        missing = [m for m in moduli if m not in selected_moduli]
        random.shuffle(missing)
        
        for m in missing:
            if time.time() > deadline:
                break
            r = find_valid_residue(best, m)
            if r is not None:
                best.append((r, m))
                selected_moduli.add(m)
        
        return best
    
    best_solution = []
    best_len = 0
    
    # Run multiple strategies with local search
    strategies_done = 0
    while time.time() < deadline:
        strategies_done += 1
        
        # Use random order mostly, as it often finds good packing
        # Occasional structured orders
        if strategies_done % 5 == 1:
            order = moduli_desc
        elif strategies_done % 5 == 2:
            order = moduli_asc
        else:
            order = moduli[:]
            random.shuffle(order)
            
        sol = greedy(order)
        
        # Apply local search (fast fill)
        sol = local_search(sol)
        
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
            
        # Early exit if we hit a theoretical max or close to it? 
        # Max possible is bounded, but 127 is max moduli count. 
        # Just keep running until deadline.
    
    if not best_solution:
        best_solution = [(0, 2)]
    
    return [[r, m] for r, m in best_solution]
