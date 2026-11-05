# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
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
        
        def try_add_moduli(sol, unselected_list):
            """Try to add as many unselected moduli as possible."""
            added = []
            for m_new in unselected_list:
                if m_new in [mm for mm, _ in sol]:
                    continue
                found_r = None
                for r_new in range(m_new):
                    ok = True
                    for m_sel, r_sel in sol:
                        if not is_compatible(m_new, r_new, m_sel, r_sel):
                            ok = False
                            break
                    if ok:
                        found_r = r_new
                        break
                if found_r is not None:
                    sol.append([m_new, found_r])
                    added.append(m_new)
            return added
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Choose move type: 60% residue change, 25% swap, 15% insert
            move_type = random.random()
            
            if len(best) < 2:
                # Can't do swap if fewer than 2 elements
                if move_type > 0.35:
                    move_type = 0.5  # Force 'change' branch
                else:
                    move_type = 0.9  # Force 'insert' branch
            
            if move_type <= 0.60:  # Residue change
                idx = random.randint(0, len(best) - 1)
                m, r = best[idx]
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
                    try_add_moduli(new_sol, unselected_moduli)
                    new_score = score_solution(new_sol)
                    if new_score > best_score_val:
                        best = new_sol
                        best_score_val = new_score
                        selected_moduli = set(m for m, r in best)
                        unselected_moduli = [m for m in moduli if m not in selected_moduli]
                        unselected_moduli.sort(key=lambda x: 1.0/x, reverse=True)
                        
            elif move_type <= 0.85:  # Swap residues between two selected moduli
                if len(best) < 2:
                    continue
                idx1, idx2 = random.sample(range(len(best)), 2)
                m1, r1 = best[idx1]
                m2, r2 = best[idx2]
                # Try swapping residues
                if r1 < m2 and r2 < m1:
                    # Check if swap is compatible
                    compatible = True
                    for i, (mi, ri) in enumerate(best):
                        if i == idx1 or i == idx2:
                            continue
                        if not is_compatible(m1, r2, mi, ri) or not is_compatible(m2, r1, mi, ri):
                            compatible = False
                            break
                    # Also check the two swapped ones against each other
                    if compatible and not is_compatible(m1, r2, m2, r1):
                        compatible = False
                    
                    if compatible:
                        new_sol = [x[:] for x in best]
                        new_sol[idx1] = [m1, r2]
                        new_sol[idx2] = [m2, r1]
                        try_add_moduli(new_sol, unselected_moduli)
                        new_score = score_solution(new_sol)
                        if new_score > best_score_val:
                            best = new_sol
                            best_score_val = new_score
                            selected_moduli = set(m for m, r in best)
                            unselected_moduli = [m for m in moduli if m not in selected_moduli]
                            unselected_moduli.sort(key=lambda x: 1.0/x, reverse=True)
                            
            else:  # Insert: try to add a missing high-value modulus
                if not unselected_moduli:
                    continue
                # Try to add the highest-value missing modulus
                m_new = unselected_moduli[0]
                found_r = None
                for r_new in range(m_new):
                    ok = True
                    for m_sel, r_sel in best:
                        if not is_compatible(m_new, r_new, m_sel, r_sel):
                            ok = False
                            break
                    if ok:
                        found_r = r_new
                        break
                if found_r is not None:
                    new_sol = [x[:] for x in best]
                    new_sol.append([m_new, found_r])
                    new_score = score_solution(new_sol)
                    if new_score > best_score_val:
                        best = new_sol
                        best_score_val = new_score
                        selected_moduli = set(m for m, r in best)
                        unselected_moduli = [m for m in moduli if m not in selected_moduli]
                        unselected_moduli.sort(key=lambda x: 1.0/x, reverse=True)
        
        return best
    
    def simulated_annealing(sol, max_iters=300):
        """Simulated annealing to escape local optima."""
        best = [x[:] for x in sol]
        best_score_val = score_solution(best)
        current = [x[:] for x in sol]
        current_score = score_solution(current)
        
        selected_moduli = set(m for m, r in current)
        unselected_moduli = [m for m in moduli if m not in selected_moduli]
        unselected_moduli.sort(key=lambda x: 1.0/x, reverse=True)
        
        T = 0.05  # Initial temperature
        T_min = 0.001
        cooling_rate = 0.995
        
        def try_add_moduli(sol, unselected_list):
            added = []
            for m_new in unselected_list:
                if m_new in [mm for mm, _ in sol]:
                    continue
                found_r = None
                for r_new in range(m_new):
                    ok = True
                    for m_sel, r_sel in sol:
                        if not is_compatible(m_new, r_new, m_sel, r_sel):
                            ok = False
                            break
                    if ok:
                        found_r = r_new
                        break
                if found_r is not None:
                    sol.append([m_new, found_r])
                    added.append(m_new)
            return added
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if T < T_min:
                break
            
            # Generate a random neighbor
            move_type = random.random()
            if len(current) < 2:
                if move_type > 0.35:
                    move_type = 0.5  # Force 'change' branch
                else:
                    move_type = 0.9  # Force 'insert' branch
            
            new_sol = [x[:] for x in current]
            
            if move_type <= 0.60:  # Residue change
                idx = random.randint(0, len(new_sol) - 1)
                m, r = new_sol[idx]
                r_prime = random.randint(0, m - 1)
                if r_prime == r:
                    continue
                
                compatible = True
                for i, (m2, r2) in enumerate(new_sol):
                    if i == idx:
                        continue
                    if not is_compatible(m, r_prime, m2, r2):
                        compatible = False
                        break
                
                if compatible:
                    new_sol[idx] = [m, r_prime]
                    try_add_moduli(new_sol, unselected_moduli)
                else:
                    continue
                    
            elif move_type <= 0.85:  # Swap
                if len(new_sol) < 2:
                    continue
                idx1, idx2 = random.sample(range(len(new_sol)), 2)
                m1, r1 = new_sol[idx1]
                m2, r2 = new_sol[idx2]
                if r1 < m2 and r2 < m1:
                    compatible = True
                    for i, (mi, ri) in enumerate(new_sol):
                        if i == idx1 or i == idx2:
                            continue
                        if not is_compatible(m1, r2, mi, ri) or not is_compatible(m2, r1, mi, ri):
                            compatible = False
                            break
                    if compatible and not is_compatible(m1, r2, m2, r1):
                        compatible = False
                    
                    if compatible:
                        new_sol[idx1] = [m1, r2]
                        new_sol[idx2] = [m2, r1]
                        try_add_moduli(new_sol, unselected_moduli)
                    else:
                        continue
                else:
                    continue
                    
            else:  # Insert
                if not unselected_moduli:
                    continue
                m_new = unselected_moduli[random.randint(0, min(5, len(unselected_moduli)-1))]
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
                else:
                    continue
            
            new_score = score_solution(new_sol)
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta >= 0 or random.random() < math.exp(-delta / T):
                current = new_sol
                current_score = new_score
                if current_score > best_score_val:
                    best = [x[:] for x in current]
                    best_score_val = current_score
                    selected_moduli = set(m for m, r in current)
                    unselected_moduli = [m for m in moduli if m not in selected_moduli]
                    unselected_moduli.sort(key=lambda x: 1.0/x, reverse=True)
            
            T *= cooling_rate
        
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
        
    # Phase 3: Simulated annealing on best solution
    sol_sa = simulated_annealing(best_solution, max_iters=200)
    score_sa = score_solution(sol_sa)
    if score_sa > best_score:
        best_score = score_sa
        best_solution = sol_sa
    
    # Phase 4: Random restarts
    for restart in range(15):
        if time.time() > deadline:
            break
        order_rand = moduli[:]
        random.shuffle(order_rand)
        sol_r = greedy_construct(order_rand, residue_strategy='random')
        sol_r_ls = local_search(sol_r, max_iters=150)
        score_r_ls = score_solution(sol_r_ls)
        if score_r_ls > best_score:
            best_score = score_r_ls
            best_solution = sol_r_ls
        
        # Also try SA on the locally-searched solution
        if time.time() < deadline - 20:
            sol_r_sa = simulated_annealing(sol_r_ls, max_iters=100)
            score_r_sa = score_solution(sol_r_sa)
            if score_r_sa > best_score:
                best_score = score_r_sa
                best_solution = sol_r_sa

    # Final polish: more local search iterations on best
    if time.time() < deadline - 5:
        sol_final = local_search(best_solution, max_iters=200)
        score_final = score_solution(sol_final)
        if score_final > best_score:
            best_score = score_final
            best_solution = sol_final

    return best_solution
# EVOLVE-BLOCK-END
