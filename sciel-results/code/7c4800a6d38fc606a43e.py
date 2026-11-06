import math
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find pairwise disjoint residue classes with distinct moduli in [20, 200]
    maximizing the sum of 1/m.
    
    Two classes (m1, r1) and (m2, r2) are disjoint iff r1 != r2 mod gcd(m1, m2).
    """
    deadline = time.time() + 115  # Safety margin before 150s limit
    rng = random.Random(42)
    
    min_m = 20
    max_m = 200
    all_moduli = list(range(min_m, max_m + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    def find_residues_fast(moduli_list, current_assignment=None, max_restarts=30):
        """
        Attempts to find valid residues for the given set of moduli.
        Returns a dict {m: r} if successful, None otherwise.
        Uses randomized greedy with MRV heuristic and optional warm start.
        """
        if not moduli_list:
            return {}
        
        # Sort by number of available residues (MRV) - smaller moduli first as proxy
        sorted_moduli = sorted(moduli_list)
        
        for restart in range(max_restarts):
            assignment = {}
            if current_assignment and restart == 0:
                # Warm start: copy current assignment for moduli that are still present
                for m in moduli_list:
                    if m in current_assignment:
                        assignment[m] = current_assignment[m]
            
            # Determine which moduli still need assignment
            unassigned = [m for m in sorted_moduli if m not in assignment]
            
            success = True
            for m in unassigned:
                valid_residues = []
                for r in range(m):
                    is_valid = True
                    for m2, r2 in assignment.items():
                        g = gcd_table[m][m2]
                        if r % g == r2 % g:
                            is_valid = False
                            break
                    if is_valid:
                        valid_residues.append(r)
                
                if not valid_residues:
                    success = False
                    break
                
                # Choose randomly to diversify
                r_choice = rng.choice(valid_residues)
                assignment[m] = r_choice
            
            if success:
                return assignment
        
        return None

    def calc_score(moduli_set):
        score = 0.0
        for m in moduli_set:
            score += 1.0 / m
        return score

    def greedy_construct(order_moduli, rng_local):
        """
        Build a solution by adding moduli in the given order,
        picking a random valid residue for each.
        """
        selected = []  # list of [m, r]
        assignment = {}
        for m in order_moduli:
            found = False
            residues = list(range(m))
            rng_local.shuffle(residues)
            for r in residues:
                valid = True
                for m2, r2 in selected:
                    g = gcd_table[m][m2]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    selected.append([m, r])
                    assignment[m] = r
                    found = True
                    break
        return selected, assignment

    # Phase 1: Multiple greedy constructions with different orderings
    best_sol = None
    best_assignment = {}
    best_score = 0.0
    
    start_time = time.time()
    
    # Try ascending order (small moduli first)
    for i in range(20):
        if time.time() - start_time > 5:
            break
        local_rng = random.Random(100 + i)
        order = list(range(min_m, max_m + 1))
        sol, assign = greedy_construct(order, local_rng)
        score = calc_score([m for m, _ in sol])
        if score > best_score:
            best_score = score
            best_sol = sol
            best_assignment = assign
    
    # Try descending order (large moduli first)
    for i in range(10):
        if time.time() - start_time > 10:
            break
        local_rng = random.Random(200 + i)
        order = list(range(max_m, min_m - 1, -1))
        sol, assign = greedy_construct(order, local_rng)
        score = calc_score([m for m, _ in sol])
        if score > best_score:
            best_score = score
            best_sol = sol
            best_assignment = assign
    
    # Try random orderings
    for i in range(10):
        if time.time() - start_time > 15:
            break
        local_rng = random.Random(300 + i)
        order = list(range(min_m, max_m + 1))
        local_rng.shuffle(order)
        sol, assign = greedy_construct(order, local_rng)
        score = calc_score([m for m, _ in sol])
        if score > best_score:
            best_score = score
            best_sol = sol
            best_assignment = assign
    
    # Phase 2: Local search - try to add moduli, with residue re-optimization
    current_moduli = set(best_assignment.keys())
    current_assignment = dict(best_assignment)
    current_score = best_score
    
    def try_add_modulus(m_add, current_moduli, current_assignment):
        """
        Tries to add m_add to the current solution.
        Returns (new_moduli, new_assignment, new_score) if successful, else None.
        """
        if m_add in current_moduli:
            return None
        
        # Step 1: Try to find r for m_add compatible with current_assignment
        valid_r = []
        for r in range(m_add):
            ok = True
            for m2, r2 in current_assignment.items():
                g = gcd_table[m_add][m2]
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                valid_r.append(r)
        
        if valid_r:
            new_assignment = dict(current_assignment)
            new_assignment[m_add] = rng.choice(valid_r)
            new_moduli = set(current_moduli)
            new_moduli.add(m_add)
            new_score = current_score + 1.0 / m_add
            return new_moduli, new_assignment, new_score
        
        # Step 2: If no direct fit, try to re-optimize residues of existing moduli
        # to make room for m_add. Use a limited solver.
        new_moduli = set(current_moduli)
        new_moduli.add(m_add)
        
        # Try with warm start from current_assignment
        new_assignment = find_residues_fast(list(new_moduli), current_assignment, max_restarts=15)
        if new_assignment:
            new_score = current_score + 1.0 / m_add
            return new_moduli, new_assignment, new_score
            
        return None

    # Local search loop
    sa_start = time.time()
    no_improve_count = 0
    
    while time.time() - sa_start < 95:
        # Try to add a modulus, biased towards smaller ones
        available_moduli = [m for m in all_moduli if m not in current_moduli]
        if not available_moduli:
            break
        
        # Sort by value descending, try top K
        available_moduli.sort(key=lambda m: 1.0 / m, reverse=True)
        k = min(15, len(available_moduli))
        
        improved = False
        for m_candidate in available_moduli[:k]:
            result = try_add_modulus(m_candidate, current_moduli, current_assignment)
            if result:
                current_moduli, current_assignment, current_score = result
                if current_score > best_score:
                    best_score = current_score
                    best_sol = [[m, r] for m, r in current_assignment.items()]
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            # If stuck, try a swap: remove one large modulus, try to add a smaller one
            if current_moduli and no_improve_count % 3 == 0:
                # Remove the largest modulus
                m_largest = max(current_moduli)
                trial_moduli = set(current_moduli)
                trial_moduli.remove(m_largest)
                trial_assignment = {m: r for m, r in current_assignment.items() if m != m_largest}
                
                # Try to add smaller moduli
                candidates = [m for m in all_moduli if m < m_largest and m not in trial_moduli]
                candidates.sort(key=lambda m: 1.0 / m, reverse=True)
                
                found_swap = False
                for m_try in candidates[:5]:
                    trial_moduli_with = set(trial_moduli)
                    trial_moduli_with.add(m_try)
                    new_assignment = find_residues_fast(list(trial_moduli_with), trial_assignment, max_restarts=10)
                    if new_assignment:
                        new_score = sum(1.0 / m for m in trial_moduli_with)
                        if new_score > current_score:
                            current_moduli = trial_moduli_with
                            current_assignment = new_assignment
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_sol = [[m, r] for m, r in current_assignment.items()]
                            found_swap = True
                            break
                
                if not found_swap:
                    # Try removing a medium-sized modulus instead
                    sorted_current = sorted(current_moduli, reverse=True)
                    for m_remove in sorted_current[1:3]:  # try 2nd and 3rd largest
                        trial_moduli = set(current_moduli)
                        trial_moduli.remove(m_remove)
                        trial_assignment = {m: r for m, r in current_assignment.items() if m != m_remove}
                        
                        candidates = [m for m in all_moduli if m < m_remove and m not in trial_moduli]
                        candidates.sort(key=lambda m: 1.0 / m, reverse=True)
                        
                        found_swap = False
                        for m_try in candidates[:5]:
                            trial_moduli_with = set(trial_moduli)
                            trial_moduli_with.add(m_try)
                            new_assignment = find_residues_fast(list(trial_moduli_with), trial_assignment, max_restarts=10)
                            if new_assignment:
                                new_score = sum(1.0 / m for m in trial_moduli_with)
                                if new_score > current_score:
                                    current_moduli = trial_moduli_with
                                    current_assignment = new_assignment
                                    current_score = new_score
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_sol = [[m, r] for m, r in current_assignment.items()]
                                    found_swap = True
                                    break
                        
                        if found_swap:
                            break

    # Final polish: try to re-optimize all residues for the current set
    if time.time() - start_time < 110 and current_moduli:
        final_assignment = find_residues_fast(list(current_moduli), current_assignment, max_restarts=50)
        if final_assignment:
            final_score = calc_score(list(current_moduli))
            if final_score > best_score:
                best_score = final_score
                best_sol = [[m, r] for m, r in final_assignment.items()]
    
    # Final result
    if best_sol:
        return [[m, r] for m, r in best_sol]
    else:
        return []
# EVOLVE-BLOCK-END
