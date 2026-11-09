import time
import math
import random
from math import gcd

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 2.0

# Precompute gcd table for moduli 2..128
_gcd_table = {}
for m1 in range(2, MAXIMUM + 1):
    for m2 in range(m1, MAXIMUM + 1):
        g = gcd(m1, m2)
        _gcd_table[(m1, m2)] = g
        if m1 != m2:
            _gcd_table[(m2, m1)] = g

def _check_pair(r1, m1, r2, m2):
    """Check if two residue classes are disjoint."""
    g = _gcd_table.get((m1, m2), gcd(m1, m2))
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
        
        for r in range(m):
            compatible = True
            for (r_prev, m_prev) in solution:
                if not _check_pair(r, m, r_prev, m_prev):
                    compatible = False
                    break
            if compatible:
                solution.append([r, m])
                used_moduli.add(m)
                break
    
    return solution

def _local_search_improve(solution, deadline):
    """Try to improve the solution by adding new moduli with targeted residue search."""
    if not solution:
        return solution
    
    best = [list(x) for x in solution]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(best)
        used_moduli = set(m for _, m in best)
        
        # Try adding each missing modulus with the best residue
        for m_new in range(2, MAXIMUM + 1):
            if time.time() > deadline:
                break
            if m_new in used_moduli:
                continue
            
            # Find the residue that maximizes compatibility options for future additions
            best_r = None
            best_score = -1
            
            for r_new in range(m_new):
                # Check if this residue is compatible with all existing
                compatible = True
                for (r_prev, m_prev) in best:
                    if not _check_pair(r_new, m_new, r_prev, m_prev):
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Score: count how many other unused moduli would still have valid residues
                score = 0
                for m_other in range(2, MAXIMUM + 1):
                    if m_other in used_moduli or m_other == m_new:
                        continue
                    for r_other in range(m_other):
                        if _check_pair(r_other, m_other, r_new, m_new):
                            # Check against all current best
                            all_compat = True
                            for (r_prev, m_prev) in best:
                                if not _check_pair(r_other, m_other, r_prev, m_prev):
                                    all_compat = False
                                    break
                            if all_compat:
                                score += 1
                                break  # Just need one valid residue per modulus
                
                if score > best_score:
                    best_score = score
                    best_r = r_new
            
            if best_r is not None:
                new_solution = [list(x) for x in best]
                new_solution.append([best_r, m_new])
                best = new_solution
                used_moduli.add(m_new)
                improved = True
    
    return best

def _smarter_greedy(moduli_list, deadline):
    """Greedy with dynamic ordering: pick the modulus with fewest valid residues first."""
    solution = []
    remaining = list(moduli_list)
    
    while remaining and time.time() < deadline:
        best_m = None
        best_r = None
        best_score = -1
        
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
                if best_m is None or valid_count < best_score:
                    best_score = valid_count
                    best_m = m
                    best_r = first_valid
        
        if best_m is None:
            break
        
        solution.append([best_r, best_m])
        remaining.remove(best_m)
    
    return solution

def _targeted_greedy_with_lookahead(moduli_list, deadline):
    """Greedy that at each step picks the modulus+residue pair that allows the most future additions."""
    solution = []
    remaining = list(moduli_list)
    
    while remaining and time.time() < deadline:
        best_pair = None
        best_future_score = -1
        
        for m in remaining:
            for r in range(m):
                # Check compatibility with current solution
                compatible = True
                for (r_prev, m_prev) in solution:
                    if not _check_pair(r, m, r_prev, m_prev):
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Count how many other moduli would still have at least one valid residue
                future_score = 0
                for m_other in remaining:
                    if m_other == m:
                        continue
                    for r_other in range(m_other):
                        if _check_pair(r_other, m_other, r, m):
                            all_compat = True
                            for (r_prev, m_prev) in solution:
                                if not _check_pair(r_other, m_other, r_prev, m_prev):
                                    all_compat = False
                                    break
                            if all_compat:
                                future_score += 1
                                break
        
        if best_pair is None:
            break
        
        solution.append([best_pair[0], best_pair[1]])
        remaining.remove(best_pair[1])
    
    return solution

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Phase 1: Multiple greedy strategies with different orderings
    orderings = [
        all_moduli,
        list(reversed(all_moduli)),
        sorted(all_moduli, key=lambda m: m.bit_count()),
        sorted(all_moduli, key=lambda m: -m.bit_count()),
        sorted(all_moduli, key=lambda m: min(p for p in range(2, m+1) if m%p==0)),
        sorted(all_moduli, key=lambda m: m.bit_length()),
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        solution = _greedy_construct(mod_order, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Strategy 2: Smarter greedy with dynamic ordering
    if time.time() < deadline:
        solution = _smarter_greedy(all_moduli, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Phase 2: Local search improvement (focused on adding moduli)
    if best_solution and time.time() < deadline:
        improved = _local_search_improve(best_solution, deadline)
        if len(improved) > len(best_solution):
            best_solution = improved
    
    # Phase 3: Randomized multi-start with local improvement
    if time.time() < deadline:
        random.seed(42)
        num_attempts = 500
        for attempt in range(num_attempts):
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
    
    # Phase 4: Final local search on best solution
    if best_solution and time.time() < deadline:
        final = _local_search_improve(best_solution, deadline)
        if len(final) > len(best_solution):
            best_solution = final
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
