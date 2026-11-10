import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    # Precompute GCDs for all pairs of moduli in [20, 200]
    moduli_range = list(range(20, 201))
    n_moduli = len(moduli_range)
    gcd_table = {}
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            m1, m2 = moduli_range[i], moduli_range[j]
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def is_valid_pair(m1, r1, m2, r2):
        key = (min(m1, m2), max(m1, m2))
        g = gcd_table.get(key, math.gcd(m1, m2))
        return (r1 - r2) % g != 0
    
    def find_valid_residue(m, selected):
        """Find a valid residue for m given selected moduli/residues."""
        if not selected:
            return 0
        
        # Try residues in order of increasing conflict potential
        # First try small residues, then random
        candidates = list(range(m))
        random.shuffle(candidates)
        
        for r in candidates:
            ok = True
            for m_prime, r_prime in selected:
                if not is_valid_pair(m, r, m_prime, r_prime):
                    ok = False
                    break
            if ok:
                return r
        return None

    def greedy_build(order):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])
        return selected

    def score_solution(solution):
        if not solution:
            return 0.0
        return sum(1.0 / m for m, r in solution)

    def local_search_improve(solution, max_iterations=50):
        """Local search: try to add missing moduli by swapping residues."""
        if not solution:
            return solution
        
        selected_moduli = set(m for m, r in solution)
        missing = [m for m in range(20, 201) if m not in selected_moduli]
        missing.sort()  # Small moduli first (higher 1/m value)
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1
            
            # Try to directly add each missing modulus
            for m in missing[:]:
                if time.time() > deadline:
                    break
                if m in selected_moduli:
                    continue
                
                r = find_valid_residue(m, solution)
                if r is not None:
                    solution.append([m, r])
                    selected_moduli.add(m)
                    missing.remove(m)
                    improved = True
            
            # Try to swap residues to free up space for smaller missing moduli
            n = len(solution)
            for i in range(n):
                if time.time() > deadline:
                    break
                m_i, r_i = solution[i]
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    m_j, r_j = solution[j]
                    
                    # Try swapping r_i and r_j
                    ok_swap = True
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        m_k, r_k = solution[k]
                        if not is_valid_pair(m_i, r_j, m_k, r_k) or not is_valid_pair(m_j, r_i, m_k, r_k):
                            ok_swap = False
                            break
                    
                    if not ok_swap:
                        continue
                    
                    # Check if we can add a missing modulus with swapped residues
                    for m_missing in missing[:5]:  # Only try top 5 missing
                        if m_missing in selected_moduli:
                            continue
                        
                        # Build the hypothetical solution with swapped residues
                        hypothetical = []
                        for k in range(n):
                            if k == i:
                                hypothetical.append([m_i, r_j])
                            elif k == j:
                                hypothetical.append([m_j, r_i])
                            else:
                                hypothetical.append(solution[k][:])
                        
                        r_new = find_valid_residue(m_missing, hypothetical)
                        if r_new is not None:
                            # Perform the swap
                            solution[i][1] = r_j
                            solution[j][1] = r_i
                            solution.append([m_missing, r_new])
                            selected_moduli.add(m_missing)
                            if m_missing in missing:
                                missing.remove(m_missing)
                            improved = True
                            break
        
        return solution

    best_result = []
    best_score = 0.0
    
    all_moduli = list(range(20, 201))
    
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        
        # Use ascending order for best reciprocal sum
        order = all_moduli[:]
        
        # Occasionally shuffle to explore different residue assignments
        if num_restarts % 3 == 0:
            random.shuffle(order)
        
        solution = greedy_build(order)
        
        # Apply local search to improve
        solution = local_search_improve(solution, max_iterations=30)
        
        current_score = score_solution(solution)
        if current_score > best_score:
            best_score = current_score
            best_result = [list(x) for x in solution]
            
        if time.time() > deadline - 1:
            break

    # Final validation
    if best_result:
        for i in range(len(best_result)):
            for j in range(i + 1, len(best_result)):
                m1, r1 = best_result[i]
                m2, r2 = best_result[j]
                if not is_valid_pair(m1, r1, m2, r2):
                    return [[20, 0]]
    
    return best_result
