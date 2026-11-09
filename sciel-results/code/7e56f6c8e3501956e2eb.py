import time
import math
from math import gcd
from itertools import combinations

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 1.0

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
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                return False
    return True

def _greedy_construct(mod_order, deadline):
    """Greedy construction: for each modulus in order, pick the first compatible residue."""
    solution = []
    used_moduli = set()
    
    for m in mod_order:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        
        # Try to find a residue for modulus m that works with all existing classes
        found = False
        for r in range(m):
            compatible = True
            for (r_prev, m_prev) in solution:
                g = gcd(m, m_prev)
                if r % g == r_prev % g:
                    compatible = False
                    break
            if compatible:
                solution.append([r, m])
                used_moduli.add(m)
                found = True
                break
        
        if not found:
            continue
    
    return solution

def _local_search_improve(solution, deadline):
    """Try to improve a solution by swapping moduli or trying different residues."""
    if not solution:
        return solution
    
    current = [list(x) for x in solution]
    current_len = len(current)
    best = [list(x) for x in current]
    
    # Try to add more moduli
    all_moduli = list(range(2, MAXIMUM + 1))
    used_moduli = set(m for _, m in current)
    
    # Try adding each unused modulus
    for m in all_moduli:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        
        # Try all residues for this modulus
        for r in range(m):
            candidate = current + [[r, m]]
            if _is_valid_solution(candidate):
                best = [list(x) for x in candidate]
                used_moduli.add(m)
                break
    
    # Try to swap residues of existing moduli to free up space for more
    for i in range(len(best)):
        if time.time() > deadline:
            break
        m_i = best[i][1]
        # Try different residues for this modulus
        for r_new in range(m_i):
            if r_new == best[i][0]:
                continue
            candidate = [list(x) for x in best]
            candidate[i][0] = r_new
            if _is_valid_solution(candidate):
                # Try to add more moduli with this new configuration
                used_m = set(x[1] for x in candidate)
                for m_add in all_moduli:
                    if m_add in used_m:
                        continue
                    for r_add in range(m_add):
                        candidate2 = candidate + [[r_add, m_add]]
                        if _is_valid_solution(candidate2):
                            if len(candidate2) > len(best):
                                best = [list(x) for x in candidate2]
                            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    
    # Strategy 1: Use the seed's approach - powers of 2 with residues 2^k - 1
    # Seed: [(2^k - 1, 2^(k+1)) for k in range(7)] gives moduli 4,8,16,32,64,128,256
    # But 256 > 128, so we can only use k=0..5: moduli 4,8,16,32,64,128 (6 classes)
    seed_solution = [[(1 << k) - 1, 1 << (k + 1)] for k in range(6)]
    # Verify: moduli are 4,8,16,32,64,128 - all in [2,128], distinct
    if _is_valid_solution(seed_solution):
        best_solution = [list(x) for x in seed_solution]
    
    # Strategy 2: Greedy with various orderings
    all_moduli = list(range(2, MAXIMUM + 1))
    even_moduli = [m for m in all_moduli if m % 2 == 0]
    
    orderings = [
        even_moduli,
        list(reversed(even_moduli)),
        all_moduli,
        list(reversed(all_moduli)),
        sorted(all_moduli, key=lambda m: (m % 2, m)),
        sorted(all_moduli, key=lambda m: (m % 2 == 0, -m)),
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        
        solution = _greedy_construct(mod_order, deadline)
        if len(solution) > len(best_solution):
            best_solution = solution
    
    # Strategy 3: Local search improvement
    if len(best_solution) > 0:
        improved = _local_search_improve(best_solution, deadline)
        if len(improved) > len(best_solution):
            best_solution = improved
    
    # Strategy 4: Try a more systematic approach - group moduli by their gcd structure
    # Focus on moduli that share common factors to allow disjointness
    
    # Try all even moduli with a smarter residue assignment
    # For even moduli, gcd is at least 2, so we need r_i mod g != r_j mod g
    # Let's try to use residues that are all odd for even moduli (since gcd >= 2)
    # Actually, if all residues are odd and all moduli are even, then for any pair,
    # gcd(m_i, m_j) >= 2, and r_i mod gcd and r_j mod gcd could still be equal.
    
    # Let's try a different approach: use moduli that are multiples of a fixed number
    # For example, all multiples of 4: gcd >= 4, so we need r_i mod 4 != r_j mod 4
    # There are 4 residue classes mod 4, so we can have at most 4 such classes
    # But we can mix different base factors.
    
    # Try a hybrid: some multiples of 2, some of 3, etc.
    # For now, let's do a more exhaustive greedy with backtracking for small cases
    
    # Try to build a solution using moduli sorted by descending value (larger moduli first)
    # This might allow more flexibility in residue assignment
    
    solution = []
    for m in sorted(all_moduli, reverse=True):
        if time.time() > deadline:
            break
        found = False
        for r in range(m):
            compatible = True
            for (r_prev, m_prev) in solution:
                g = gcd(m, m_prev)
                if r % g == r_prev % g:
                    compatible = False
                    break
            if compatible:
                solution.append([r, m])
                found = True
                break
        if not found:
            continue
    
    if len(solution) > len(best_solution):
        best_solution = solution
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
