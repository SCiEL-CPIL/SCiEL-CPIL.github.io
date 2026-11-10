import math
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 110
    moduli = list(range(20, 201))
    
    # Precompute gcds and weights
    gcd_cache = {}
    for i in range(20, 201):
        for j in range(i, 201):
            gcd_cache[(i, j)] = math.gcd(i, j)
            
    def get_gcd(m1, m2):
        if m1 == m2: return m1
        if m1 > m2: m1, m2 = m2, m1
        return gcd_cache.get((m1, m2), math.gcd(m1, m2))

    def is_compatible(m1, r1, m2, r2):
        g = get_gcd(m1, m2)
        if g == 1:
            return False # Always intersect
        return (r1 % g) != (r2 % g)

    def can_add(m, selected_moduli):
        """Check if m is compatible with all moduli in selected_moduli (ignoring residues for now, just checking gcd>1)"""
        for m2 in selected_moduli:
            if get_gcd(m, m2) == 1:
                return False
        return True

    def find_residues(moduli_list):
        """
        Backtracking search for residues.
        Returns list of [m, r] or None.
        """
        # Order by most constrained (smallest m? or largest gcd constraints?)
        # Let's just try in order of increasing m
        sorted_mods = sorted(moduli_list)
        
        def backtrack(idx, current_assignments):
            if idx == len(sorted_mods):
                return current_assignments
            
            m = sorted_mods[idx]
            # Try all residues
            for r in range(m):
                compatible = True
                for j in range(idx):
                    m2 = sorted_mods[j]
                    r2 = current_assignments[j]
                    if not is_compatible(m, r, m2, r2):
                        compatible = False
                        break
                if compatible:
                    res = backtrack(idx + 1, current_assignments + [r])
                    if res is not None:
                        return res
            return None

        res = backtrack(0, [])
        if res is None:
            return None
        return [[sorted_mods[i], res[i]] for i in range(len(sorted_mods))]

    def score(moduli_list):
        return sum(1.0 / m for m in moduli_list)

    # Phase 1: Greedy Construction
    # Sort by 1/m descending (smallest m first)
    sorted_moduli = sorted(moduli, key=lambda x: x)
    
    best_set = []
    for m in sorted_moduli:
        if time.time() > deadline: break
        if can_add(m, best_set):
            best_set.append(m)
            
    # Verify residues exist for best_set
    # If not, we might need to remove some
    # But greedy on non-coprime graph usually yields valid CSP instances
    
    # Phase 2: Local Search on the Set of Moduli
    # We want to maximize sum(1/m).
    # Moves:
    # 1. Add a modulus if it fits.
    # 2. Swap a modulus with another if the new one is better (smaller) and fits.
    # 3. 2-Swap: Remove 2, Add 2.
    
    current_set = list(best_set)
    current_score = score(current_set)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Try adding any missing modulus
        current_set_set = set(current_set)
        for m in sorted_moduli:
            if time.time() > deadline - 10: break
            if m in current_set_set: continue
            if can_add(m, current_set):
                # Check if residues can be assigned
                # Optimization: Only check if adding m doesn't break the clique property?
                # can_add already ensures clique property.
                # Residue assignment might still fail, but rare.
                new_set = current_set + [m]
                # Skip residue check in loop for speed, do it at the end?
                # No, if residue check fails, the set is invalid.
                # Let's assume if clique holds, residues exist.
                # If not, we fix it later.
                current_set = new_set
                current_set_set.add(m)
                current_score = score(current_set)
                improved = True
                break # Restart loop to allow more additions
                
        # Try swapping
        for i in range(len(current_set)):
            if time.time() > deadline - 10: break
            m_old = current_set[i]
            # Try replacing with a smaller modulus
            for m_new in sorted_moduli:
                if m_new >= m_old: break # We want smaller m (higher weight)
                if m_new in current_set_set: continue
                # Check if m_new is compatible with all others
                others = [m for j, m in enumerate(current_set) if j != i]
                if can_add(m_new, others):
                    # Swap
                    current_set[i] = m_new
                    current_set_set.discard(m_old)
                    current_set_set.add(m_new)
                    current_score = score(current_set)
                    improved = True
                    break
            if improved: break
            
    # Final Verification and Residue Assignment
    # Sort current_set for deterministic output
    current_set.sort()
    
    # Try to find residues
    result = find_residues(current_set)
    
    if result is None:
        # If residue assignment fails, we need to drop some moduli
        # Simple heuristic: drop the largest moduli (lowest weight) until it works
        # Or use a more robust CSP solver
        # For now, try dropping largest
        while current_set and result is None:
            if time.time() > deadline - 2: break
            # Remove largest m
            current_set.pop()
            current_set.sort()
            result = find_residues(current_set)
            
    if result is None:
        # Fallback: return empty or single element
        return [[20, 0]]
        
    return result
