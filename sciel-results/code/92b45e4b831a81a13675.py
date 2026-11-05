import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Build all possible (m, r) pairs
    all_pairs = []
    for m in range(20, 201):
        for r in range(m):
            all_pairs.append((m, r))
    
    n = len(all_pairs)
    
    # Precompute gcd matrix
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # For each pair, compute incompatibility with others
    # But n is ~20000, so O(n^2) is too slow.
    
    # Better approach: Use the fact that we pick at most one r per m.
    # So we have 181 groups (m=20..200), each with m residues.
    
    # Let's use a greedy + local search approach.
    
    # First, let's find a good initial solution using greedy.
    # Sort moduli by 1/m (descending) and for each, pick the best compatible residue.
    
    moduli = list(range(20, 201))
    
    def is_compatible(selected, m_new, r_new):
        """Check if (m_new, r_new) is compatible with all in selected."""
        for (m_old, r_old) in selected:
            g = math.gcd(m_new, m_old)
            if r_new % g == r_old % g:
                return False
        return True
    
    def greedy_solution():
        selected = []
        remaining = list(range(20, 201))
        # Sort by 1/m descending
        remaining.sort(key=lambda m: -1.0/m)
        for m in remaining:
            # Try all residues, pick the first compatible one
            best_r = None
            for r in range(m):
                if is_compatible(selected, m, r):
                    best_r = r
                    break
            if best_r is not None:
                selected.append((m, best_r))
        return selected
    
    selected = greedy_solution()
    current_score = sum(1.0/m for m, r in selected)
    
    # Local search: try to swap residues or add/remove moduli
    def score_of(sol):
        return sum(1.0/m for m, r in sol)
    
    # Simulated annealing
    import random
    random.seed(42)
    
    T = 1.0
    cooling = 0.999
    min_T = 1e-6
    best_solution = list(selected)
    best_score = current_score
    
    while T > min_T and time.time() < deadline:
        # Generate a neighbor
        sol = list(selected)
        move_type = random.choice(['swap_r', 'add_m', 'remove_m', 'swap_m'])
        
        if move_type == 'swap_r' and sol:
            idx = random.randint(0, len(sol)-1)
            m, r = sol[idx]
            new_r = random.randint(0, m-1)
            if new_r != r:
                # Check compatibility
                sol[idx] = (m, new_r)
                compatible = True
                for j in range(len(sol)):
                    if j == idx:
                        continue
                    m2, r2 = sol[j]
                    g = math.gcd(m, m2)
                    if new_r % g == r2 % g:
                        compatible = False
                        break
                if not compatible:
                    sol[idx] = (m, r)
                    T *= cooling
                    continue
        elif move_type == 'add_m':
            # Try to add a new modulus
            current_mods = set(m for m, r in sol)
            candidates = [m for m in range(20, 201) if m not in current_mods]
            if candidates:
                m_new = random.choice(candidates)
                # Find a compatible residue
                found = False
                for r_new in range(m_new):
                    compatible = True
                    for (m_old, r_old) in sol:
                        g = math.gcd(m_new, m_old)
                        if r_new % g == r_old % g:
                            compatible = False
                            break
                    if compatible:
                        sol.append((m_new, r_new))
                        found = True
                        break
                if not found:
                    T *= cooling
                    continue
        elif move_type == 'remove_m' and len(sol) > 1:
            idx = random.randint(0, len(sol)-1)
            sol.pop(idx)
        elif move_type == 'swap_m' and sol:
            # Replace one modulus with another
            idx = random.randint(0, len(sol)-1)
            m_old, r_old = sol[idx]
            current_mods = set(m for m, r in sol)
            candidates = [m for m in range(20, 201) if m not in current_mods]
            if candidates:
                m_new = random.choice(candidates)
                # Find compatible residue for new modulus
                found = False
                for r_new in range(m_new):
                    compatible = True
                    for j in range(len(sol)):
                        if j == idx:
                            continue
                        m2, r2 = sol[j]
                        g = math.gcd(m_new, m2)
                        if r_new % g == r2 % g:
                            compatible = False
                            break
                    if compatible:
                        sol[idx] = (m_new, r_new)
                        found = True
                        break
                if not found:
                    T *= cooling
                    continue
        
        new_score = score_of(sol)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            selected = sol
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_solution = list(selected)
        
        T *= cooling
    
    return [[m, r] for m, r in best_solution]
