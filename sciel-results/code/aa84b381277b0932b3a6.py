import time
import math
import random
from math import gcd

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 1.0

def _check_pair(r1, m1, r2, m2):
    """Check if two residue classes are disjoint."""
    g = gcd(m1, m2)
    return r1 % g != r2 % g

def _is_valid_solution(solution):
    """Check if a solution is valid: distinct moduli in [2,128], canonical residues, pairwise disjoint."""
    if not solution:
        return False
    seen_moduli = set()
    for r, m in solution:
        if m < 2 or m > MAXIMUM:
            return False
        if m in seen_moduli:
            return False
        seen_moduli.add(m)
        if r < 0 or r >= m:
            return False
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            r1, m1 = solution[i]
            r2, m2 = solution[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _greedy_construct(mod_order, deadline):
    """Greedy construction: for each modulus in order, pick the first valid residue."""
    solution = []
    used_moduli = set()
    
    for m in mod_order:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        
        found = False
        for r in range(m):
            compatible = True
            for (r_prev, m_prev) in solution:
                if not _check_pair(r, m, r_prev, m_prev):
                    compatible = False
                    break
            if compatible:
                solution.append([r, m])
                used_moduli.add(m)
                found = True
                break
    
    return solution

def _local_search_improve(solution, deadline):
    """Try to improve the solution by reassigning residues or swapping moduli."""
    if not solution:
        return solution
    
    best = [list(x) for x in solution]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(best)
        
        # Try reassigning residues for each element
        for i in range(n):
            if time.time() > deadline:
                break
            r_i, m_i = best[i]
            # Try other residues for this modulus
            for r in range(m_i):
                if r == r_i:
                    continue
                # Check if new residue works with all others
                compatible = True
                for j in range(n):
                    if i == j:
                        continue
                    r_j, m_j = best[j]
                    if not _check_pair(r, m_i, r_j, m_j):
                        compatible = False
                        break
                if compatible:
                    # Try this new residue - check if it allows adding a new modulus
                    new_solution = [list(x) for x in best]
                    new_solution[i] = [r, m_i]
                    
                    # Try to add a new modulus
                    used_moduli = set(m for _, m in new_solution)
                    for m_new in range(2, MAXIMUM + 1):
                        if m_new in used_moduli:
                            continue
                        for r_new in range(m_new):
                            compatible_new = True
                            for (r_prev, m_prev) in new_solution:
                                if not _check_pair(r_new, m_new, r_prev, m_prev):
                                    compatible_new = False
                                    break
                            if compatible_new:
                                new_solution.append([r_new, m_new])
                                if len(new_solution) > len(best):
                                    best = new_solution
                                    improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        # Try swapping moduli
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                r_i, m_i = best[i]
                r_j, m_j = best[j]
                # Try swapping: use m_j for class i and m_i for class j
                # Need to find new residues
                new_solution = [list(x) for x in best]
                # Try to assign residue for m_j in position i
                found_i = False
                for r in range(m_j):
                    compatible = True
                    for k in range(n):
                        if k == i:
                            continue
                        r_k, m_k = new_solution[k]
                        if k == j:
                            continue  # will be reassigned
                        if not _check_pair(r, m_j, r_k, m_k):
                            compatible = False
                            break
                    if compatible:
                        new_solution[i] = [r, m_j]
                        found_i = True
                        break
                if not found_i:
                    continue
                # Try to assign residue for m_i in position j
                found_j = False
                for r in range(m_i):
                    compatible = True
                    for k in range(n):
                        if k == j:
                            continue
                        r_k, m_k = new_solution[k]
                        if k == i:
                            continue  # will be reassigned
                        if not _check_pair(r, m_i, r_k, m_k):
                            compatible = False
                            break
                    if compatible:
                        new_solution[j] = [r, m_i]
                        found_j = True
                        break
                if not found_j:
                    continue
                
                # Check full validity
                if _is_valid_solution(new_solution):
                    # Try to add a new modulus
                    used_moduli = set(m for _, m in new_solution)
                    for m_new in range(2, MAXIMUM + 1):
                        if m_new in used_moduli:
                            continue
                        for r_new in range(m_new):
                            compatible_new = True
                            for (r_prev, m_prev) in new_solution:
                                if not _check_pair(r_new, m_new, r_prev, m_prev):
                                    compatible_new = False
                                    break
                            if compatible_new:
                                new_solution.append([r_new, m_new])
                                if len(new_solution) > len(best):
                                    best = new_solution
                                    improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    
    # Phase 1: Greedy with various orderings
    all_moduli = list(range(2, MAXIMUM + 1))
    even_moduli = [m for m in all_moduli if m % 2 == 0]
    odd_moduli = [m for m in all_moduli if m % 2 == 1]
    
    # Fixed orderings
    fixed_orderings = [
        all_moduli,
        list(reversed(all_moduli)),
        even_moduli,
        list(reversed(even_moduli)),
        sorted(all_moduli, key=lambda m: -m.bit_count()),
        sorted(all_moduli, key=lambda m: m.bit_count()),
    ]
    
    for mod_order in fixed_orderings:
        if time.time() > deadline:
            break
        solution = _greedy_construct(mod_order, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Phase 2: Local search improvement on best solution
    if best_solution:
        improved = _local_search_improve(best_solution, deadline)
        if len(improved) > len(best_solution):
            best_solution = improved
    
    # Phase 3: Randomized multi-start greedy with local search
    if time.time() < deadline:
        random.seed(42)
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Create a random ordering
            mod_order = all_moduli[:]
            random.shuffle(mod_order)
            
            solution = _greedy_construct(mod_order, deadline)
            if len(solution) > len(best_solution):
                best_solution = [list(x) for x in solution]
            
            # Try local search on this solution too
            if time.time() < deadline and len(solution) > 0:
                improved = _local_search_improve(solution, deadline)
                if len(improved) > len(best_solution):
                    best_solution = improved
    
    # Phase 4: Try to add more moduli to the best solution
    if best_solution and time.time() < deadline:
        final = _local_search_improve(best_solution, deadline)
        if len(final) > len(best_solution):
            best_solution = final
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
