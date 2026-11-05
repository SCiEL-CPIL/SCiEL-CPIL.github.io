# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    moduli_set = set(moduli)
    
    # Precompute gcd for all pairs of moduli
    gcd_table = {}
    for i in range(n_moduli):
        for j in range(i+1, n_moduli):
            gcd_table[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
            
    def get_gcd(m1, m2):
        if m1 == m2:
            return m1
        a, b = min(m1, m2), max(m1, m2)
        return gcd_table.get((a, b), math.gcd(a, b))
    
    def score_solution(sol):
        return sum(Fraction(1, m) for m, r in sol)
    
    def is_compatible(m1, r1, m2, r2):
        """Check if (m1, r1) and (m2, r2) are disjoint."""
        if m1 == m2:
            return r1 != r2
        g = get_gcd(m1, m2)
        return r1 % g != r2 % g
        
    def greedy_construct(order, residue_strategy='min_damage'):
        selected = []
        selected_set = set()
        
        for m in order:
            if m in selected_set:
                continue
            
            valid_residues = []
            for r in range(m):
                ok = True
                for m_sel, r_sel in selected:
                    if not is_compatible(m, r, m_sel, r_sel):
                        ok = False
                        break
                if ok:
                    valid_residues.append(r)
            
            if not valid_residues:
                continue
            
            if len(valid_residues) == 1:
                r = valid_residues[0]
            elif residue_strategy == 'random':
                r = random.choice(valid_residues)
            else:  # 'min_damage'
                best_r = valid_residues[0]
                min_damage = float('inf')
                
                for r_candidate in valid_residues:
                    damage = 0
                    # Look ahead at unselected moduli
                    for n in order:
                        if n in selected_set or n == m:
                            continue
                        g = get_gcd(m, n)
                        r_mod = r_candidate % g
                        # Count how many residues of n are blocked by this choice
                        # Approximation: count total residues in that class
                        # This is an upper bound, but consistent for comparison
                        # Better: check if it blocks the *specific* residue we would likely pick?
                        # For now, count the size of the blocked set for n
                        ap_size = n // g + (1 if n % g > r_mod else 0)
                        damage += ap_size
                        if damage >= min_damage:
                            break
                    if damage < min_damage:
                        min_damage = damage
                        best_r = r_candidate
                        if min_damage == 0:
                            break
                r = best_r
            
            selected.append([m, r])
            selected_set.add(m)
        
        return selected

    def local_search(sol, max_iters=500):
        best = [x[:] for x in sol]
        best_score_val = score_solution(best)
        selected_moduli = set(m for m, r in best)
        unselected_moduli = [m for m in moduli if m not in selected_moduli]
        
        # Sort unselected by weight (1/m) descending to prioritize high value additions
        unselected_moduli.sort(key=lambda x: 1.0/x, reverse=True)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if not unselected_moduli:
                break
                
            # Randomly pick a selected modulus to potentially change or remove
            idx = random.randint(0, len(best) - 1)
            m, r = best[idx]
            
            # Try to change residue
            r_prime = random.randint(0, m - 1)
            if r_prime == r:
                continue
            
            compatible = True
            for i, (m2, r2) in enumerate(best):
                if i == idx:
                    continue
                if not is_compatible(m, r_prime, m2, r2):
                    compatible = False
                    break
            
            if compatible:
                new_sol = [x[:] for x in best]
                new_sol[idx] = [m, r_prime]
                
                # Try to add unselected moduli
                added_any = False
                for m_new in unselected_moduli:
                    if m_new in [mm for mm, _ in new_sol]:
                        continue
                    found_r = None
                    for r_new in range(m_new):
                        ok = True
                        for m_sel, r_sel in new_sol:
                            if not is_compatible(m_new, r_new, m_sel, r_sel):
                                ok = False
                                break
                        if ok:
                            found_r = r_new
                            break
                    if found_r is not None:
                        new_sol.append([m_new, found_r])
                        added_any = True
                        break # Only add one at a time for simplicity, or loop
                
                new_score = score_solution(new_sol)
                if new_score > best_score_val:
                    best = new_sol
                    best_score_val = new_score
                    selected_moduli = set(m for m, r in best)
                    unselected_moduli = [m for m in moduli if m not in selected_moduli]
                    unselected_moduli.sort(key=lambda x: 1.0/x, reverse=True)
                    
        return best

    # Strategy
    best_solution = []
    best_score = Fraction(0, 1)
    
    # Phase 1: Greedy with min_damage (sorted by increasing m, i.e., decreasing weight)
    order_sorted = sorted(moduli)
    sol_greedy = greedy_construct(order_sorted, residue_strategy='min_damage')
    score_greedy = score_solution(sol_greedy)
    if score_greedy > best_score:
        best_score = score_greedy
        best_solution = sol_greedy
    
    # Phase 2: Local search
    sol_ls = local_search(best_solution, max_iters=500)
    score_ls = score_solution(sol_ls)
    if score_ls > best_score:
        best_score = score_ls
        best_solution = sol_ls
        
    # Phase 3: Random restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        order_rand = moduli[:]
        random.shuffle(order_rand)
        sol_r = greedy_construct(order_rand, residue_strategy='random')
        sol_r_ls = local_search(sol_r, max_iters=100)
        score_r_ls = score_solution(sol_r_ls)
        if score_r_ls > best_score:
            best_score = score_r_ls
            best_solution = sol_r_ls

    return best_solution
# EVOLVE-BLOCK-END
