import time
import math
import random
from math import gcd

MAXIMUM = 128
DEADLINE = 88.0

def _check_pair(r1, m1, r2, m2):
    """Check if two residue classes are disjoint."""
    g = gcd(m1, m2)
    return r1 % g != r2 % g

def _is_valid_solution(solution):
    """Check if a solution is valid."""
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

def _smarter_greedy(moduli_list, deadline):
    """Greedy with dynamic ordering: pick modulus with fewest valid residues (most constrained)."""
    solution = []
    remaining = list(moduli_list)
    
    while remaining and time.time() < deadline:
        best_m = None
        best_r = None
        best_score = float('inf')
        
        for m in remaining:
            valid_count = 0
            first_valid = None
            for r in range(m):
                compatible = True
                for (r_prev, m_prev) in solution:
                    if not _check_pair(r, m, r_prev, m_prev):
                        compatible = False
                        break
                if compatible:
                    valid_count += 1
                    if first_valid is None:
                        first_valid = r
            if valid_count > 0:
                score = valid_count
                if score < best_score:
                    best_score = score
                    best_m = m
                    best_r = first_valid
        
        if best_m is None:
            break
        
        solution.append([best_r, best_m])
        remaining.remove(best_m)
    
    return solution

def _local_search_improve(solution, deadline):
    """Try to improve by reassigning residues and adding new moduli."""
    if not solution:
        return solution
    
    best = [list(x) for x in solution]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(best)
        
        # Try reassigning residues to enable adding a new modulus
        for i in range(n):
            if time.time() > deadline:
                break
            r_i, m_i = best[i]
            for r in range(m_i):
                if r == r_i:
                    continue
                compatible = True
                for j in range(n):
                    if i == j:
                        continue
                    r_j, m_j = best[j]
                    if not _check_pair(r, m_i, r_j, m_j):
                        compatible = False
                        break
                if compatible:
                    new_solution = [list(x) for x in best]
                    new_solution[i] = [r, m_i]
                    
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
        
        # Try adding a new modulus directly
        used_moduli = set(m for _, m in best)
        for m_new in range(2, MAXIMUM + 1):
            if time.time() > deadline:
                break
            if m_new in used_moduli:
                continue
            for r_new in range(m_new):
                compatible_new = True
                for (r_prev, m_prev) in best:
                    if not _check_pair(r_new, m_new, r_prev, m_prev):
                        compatible_new = False
                        break
                if compatible_new:
                    new_solution = [list(x) for x in best]
                    new_solution.append([r_new, m_new])
                    if len(new_solution) > len(best):
                        best = new_solution
                        improved = True
                    break
            if improved:
                break
    
    return best

def _two_opt_swap(solution, deadline):
    """Try swapping residues of two elements to enable adding a new modulus."""
    if not solution:
        return solution
    
    best = [list(x) for x in solution]
    n = len(best)
    improved = False
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            r_i, m_i = best[i]
            r_j, m_j = best[j]
            
            # Try swapping residues
            for r_new_i in range(m_i):
                for r_new_j in range(m_j):
                    if r_new_i == r_i and r_new_j == r_j:
                        continue
                    
                    # Check compatibility of new residues with each other
                    if not _check_pair(r_new_i, m_i, r_new_j, m_j):
                        continue
                    
                    # Check compatibility with all other elements
                    compatible = True
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        r_k, m_k = best[k]
                        if not _check_pair(r_new_i, m_i, r_k, m_k) or not _check_pair(r_new_j, m_j, r_k, m_k):
                            compatible = False
                            break
                    
                    if compatible:
                        new_solution = [list(x) for x in best]
                        new_solution[i] = [r_new_i, m_i]
                        new_solution[j] = [r_new_j, m_j]
                        
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
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_solution = []
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Phase 1: Multiple greedy strategies with various orderings
    orderings = [
        all_moduli,
        list(reversed(all_moduli)),
        sorted(all_moduli, key=lambda m: m.bit_count()),
        sorted(all_moduli, key=lambda m: -m.bit_count()),
        sorted(all_moduli, key=lambda m: min(p for p in range(2, m+1) if m%p==0)),
        sorted(all_moduli, key=lambda m: max(p for p in range(2, m+1) if m%p==0)),
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        solution = _greedy_construct(mod_order, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Strategy: Smarter greedy with dynamic ordering
    if time.time() < deadline:
        solution = _smarter_greedy(all_moduli, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Phase 2: Local search improvement
    if best_solution and time.time() < deadline:
        improved = _local_search_improve(best_solution, deadline)
        if len(improved) > len(best_solution):
            best_solution = improved
    
    # Phase 3: Randomized multi-start with diverse orderings
    if time.time() < deadline:
        random.seed(12345)
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            mod_order = all_moduli[:]
            random.shuffle(mod_order)
            
            solution = _greedy_construct(mod_order, deadline)
            if len(solution) > len(best_solution):
                best_solution = [list(x) for x in solution]
            
            if time.time() < deadline and len(solution) > 0:
                improved = _local_search_improve(solution, deadline)
                if len(improved) > len(best_solution):
                    best_solution = improved
    
    # Phase 4: Two-opt swap improvement
    if best_solution and time.time() < deadline:
        swapped = _two_opt_swap(best_solution, deadline)
        if len(swapped) > len(best_solution):
            best_solution = swapped
    
    # Phase 5: Final local search
    if best_solution and time.time() < deadline:
        final = _local_search_improve(best_solution, deadline)
        if len(final) > len(best_solution):
            best_solution = final
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
