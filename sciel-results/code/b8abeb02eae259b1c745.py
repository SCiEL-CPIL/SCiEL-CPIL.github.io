import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s safety margin for 60s limit
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_moduli = len(moduli)
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    
    # Precompute gcd for all pairs
    gcd_cache = {}
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            gcd_cache[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
    
    def get_gcd(m1, m2):
        if m1 == m2:
            return m1
        key = (min(m1, m2), max(m1, m2))
        return gcd_cache.get(key, math.gcd(m1, m2))
    
    def is_compatible(r1, m1, r2, m2):
        """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
        g = get_gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def find_valid_residues(m, selected):
        """Find all residues r for modulus m that are compatible with all selected."""
        valid = []
        for r in range(m):
            compatible = True
            for m_p, r_p in selected:
                if not is_compatible(r, m, r_p, m_p):
                    compatible = False
                    break
            if compatible:
                valid.append(r)
        return valid

    # Phase 1: Dynamic Greedy with Most-Constrained-First Heuristic
    best_solution = []
    
    # We will run multiple iterations of the dynamic greedy with random tie-breaking
    # to explore the solution space.
    
    def dynamic_greedy(randomize_ties=False):
        selected = []
        remaining = set(moduli)
        
        while remaining:
            if time.time() > deadline:
                break
            
            # For each remaining modulus, find valid residues
            # and count them. We want to pick the one with the fewest valid residues
            # (Most Constrained First). If tie, prefer primes, then smaller moduli.
            
            candidates = []
            for m in remaining:
                valid_residues = find_valid_residues(m, selected)
                if valid_residues:
                    # Heuristic score: 
                    # 1. Minimize number of valid residues (most constrained)
                    # 2. Prefer primes (less likely to cause future conflicts? Actually primes are less constrained usually, 
                    #    but let's stick to the "most constrained" logic primarily)
                    #    Wait, if a modulus has only 1 valid residue, it's very constrained. 
                    #    If it has many, it's flexible. We should pick the constrained ones first to fail fast or lock in structure.
                    score = (len(valid_residues), 0 if m in primes else 1, m)
                    candidates.append((score, m, valid_residues))
            
            if not candidates:
                break
            
            # Sort candidates by score
            candidates.sort(key=lambda x: x[0])
            
            # Pick the best candidate
            # If randomize_ties is True, we might pick from top few? 
            # The directive says "dynamic ... selects the next modulus based on the minimum number of remaining valid residues".
            # So we strictly pick the min count. If ties, we can randomize among them to diversify.
            
            min_count = candidates[0][0][0]
            tied_candidates = [c for c in candidates if c[0][0] == min_count]
            
            if randomize_ties and len(tied_candidates) > 1:
                chosen = random.choice(tied_candidates)
            else:
                chosen = tied_candidates[0]
            
            _, m, valid_residues = chosen
            r = random.choice(valid_residues) if randomize_ties else valid_residues[0]
            
            selected.append((m, r))
            remaining.remove(m)
            
        return selected
    
    # Run dynamic greedy with various randomization settings
    # 1. Deterministic (pick first in sorted order, first residue)
    sol = dynamic_greedy(randomize_ties=False)
    if len(sol) > len(best_solution):
        best_solution = sol
        
    # 2. Randomized ties and residues
    for _ in range(50):
        if time.time() > deadline:
            break
        sol = dynamic_greedy(randomize_ties=True)
        if len(sol) > len(best_solution):
            best_solution = sol
            
    # Phase 2: Local search - try to add more moduli
    for _ in range(50):  # Iterations
        if time.time() > deadline:
            break
        
        improved = False
        selected_mods = set(m for m, r in best_solution)
        unselected_mods = [m for m in moduli if m not in selected_mods]
        
        # Try to add each unselected modulus
        for m in unselected_mods:
            if time.time() > deadline:
                break
            valid_residues = find_valid_residues(m, best_solution)
            if valid_residues:
                r = random.choice(valid_residues) if len(valid_residues) > 1 else valid_residues[0]
                best_solution.append((m, r))
                improved = True
                break
        
        # Try changing residues of selected moduli to allow adding more
        if not improved:
            for i in range(len(best_solution)):
                if time.time() > deadline:
                    break
                m, r = best_solution[i]
                for new_r in range(m):
                    if new_r == r:
                        continue
                    
                    # Check if new_r is compatible with all other selected
                    compatible = True
                    for j in range(len(best_solution)):
                        if j == i:
                            continue
                        m_p, r_p = best_solution[j]
                        if not is_compatible(new_r, m, r_p, m_p):
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    # Try adding an unselected modulus with this change
                    test_solution = best_solution.copy()
                    test_solution[i] = (m, new_r)
                    
                    for m_add in unselected_mods:
                        valid_add = find_valid_residues(m_add, test_solution)
                        if valid_add:
                            r_add = random.choice(valid_add) if len(valid_add) > 1 else valid_add[0]
                            best_solution = test_solution + [(m_add, r_add)]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    # Phase 3: More aggressive local search - try swapping residues
    for _ in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        selected_mods = set(m for m, r in best_solution)
        unselected_mods = [m for m in moduli if m not in selected_mods]
        
        if not unselected_mods:
            break
        
        # Try removing one selected modulus and adding two unselected
        for i in range(len(best_solution)):
            if time.time() > deadline:
                break
            
            m_remove, r_remove = best_solution[i]
            reduced_solution = best_solution[:i] + best_solution[i+1:]
            
            # Try adding two unselected moduli
            added_count = 0
            new_solution = reduced_solution.copy()
            for m_add in unselected_mods:
                if time.time() > deadline:
                    break
                if m_add in set(m for m, r in new_solution):
                    continue
                valid_add = find_valid_residues(m_add, new_solution)
                if valid_add:
                    r_add = random.choice(valid_add) if len(valid_add) > 1 else valid_add[0]
                    new_solution.append((m_add, r_add))
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if added_count > 1:  # Net gain
                best_solution = new_solution
                improved = True
                break
        
        if not improved:
            break
    
    result = [[r, m] for m, r in best_solution]
    
    if not result:
        return [[0, 2]]
        
    return result
