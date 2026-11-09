import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcds
    gcd_matrix = {}
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            gcd_matrix[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
            
    def is_compatible(selected, m, r):
        """
        selected: list of (m', r')
        m, r: new candidate
        Returns True if (m, r) is disjoint from all in selected.
        """
        for m_p, r_p in selected:
            g = gcd_matrix.get((m, m_p), math.gcd(m, m_p))
            # Disjoint if r not congruent to r_p mod g
            if (r - r_p) % g == 0:
                return False
        return True

    def find_valid_residue(selected, m):
        """
        Find any r in [0, m-1] compatible with selected.
        Returns r or None.
        """
        for r in range(m):
            if is_compatible(selected, m, r):
                return r
        return None

    best_solution = []
    
    # Heuristic 1: Sort by number of divisors (ascending or descending)
    # Heuristic 2: Random shuffle
    # Heuristic 3: Sort by value
    
    # We will run multiple iterations of greedy with different orderings
    
    # Pre-compute divisors count for sorting heuristic
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    div_count = {m: count_divisors(m) for m in moduli}
    
    # Define several orderings
    orderings = []
    
    # 1. Random shuffles
    for _ in range(50):
        order = moduli.copy()
        random.shuffle(order)
        orderings.append(order)
        
    # 2. Sorted by divisors (fewer divisors first? more first?)
    # Usually, numbers with fewer constraints (fewer gcds > 1) are easier to place?
    # Or numbers with many gcds > 1 are "harder" to place?
    # Let's try both.
    orderings.append(sorted(moduli, key=lambda x: div_count[x]))
    orderings.append(sorted(moduli, key=lambda x: -div_count[x]))
    
    # 3. Sorted by value
    orderings.append(sorted(moduli))
    orderings.append(sorted(moduli, reverse=True))
    
    # 4. Greedy with "most constrained" first?
    # Hard to define without state.
    
    for order in orderings:
        if time.time() > deadline:
            break
            
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(selected, m)
            if r is not None:
                selected.append((m, r))
                
        if len(selected) > len(best_solution):
            best_solution = selected
            
    # Convert to required format: list of [residue, modulus]
    result = [[r, m] for m, r in best_solution]
    
    # If no solution found (should not happen, as single element is valid), return a valid single one
    if not result:
        return [[0, 2]]
        
    return result
