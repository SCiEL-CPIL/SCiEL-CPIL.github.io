import math

def construct():
    max_mod = 128
    
    # The optimal set of moduli is all even numbers from 2 to 128.
    # This forms a clique in the "gcd > 1" graph.
    # Size is 64.
    moduli = list(range(2, max_mod + 1, 2))
    
    # We need to assign residues r_m such that for all i != j:
    # r_i != r_j (mod gcd(m_i, m_j))
    
    # We use backtracking to find a valid assignment for this specific set.
    # Sorting by ascending modulus helps because smaller moduli have fewer residue options.
    # However, moduli with many constraints (high degree) should be prioritized.
    # In this set, all moduli are connected.
    # Let's sort by descending "tightness" or just use a heuristic.
    # Actually, sorting by ascending value is a good start.
    
    sorted_moduli = sorted(moduli)
    assignment = {}
    
    def is_compatible(r, m, assigned):
        for m2, r2 in assigned.items():
            g = math.gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True

    def backtrack(idx):
        if idx == len(sorted_moduli):
            return True
        
        m = sorted_moduli[idx]
        # Try residues
        # Optimization: We can try residues that are unlikely to conflict.
        # For now, iterate 0..m-1.
        for r in range(m):
            if is_compatible(r, m, assignment):
                assignment[m] = r
                if backtrack(idx + 1):
                    return True
                del assignment[m]
        
        return False

    if backtrack(0):
        result = [[assignment[m], m] for m in moduli]
        return result
    else:
        # Fallback: If backtracking fails (unlikely for this structure),
        # try a greedy approach or return a smaller valid set.
        # But 64 should be solvable.
        # Let's try a simple greedy just in case.
        current_moduli = []
        current_assignment = {}
        for m in moduli:
            for r in range(m):
                if is_compatible(r, m, current_assignment):
                    current_assignment[m] = r
                    current_moduli.append(m)
                    break
        return [[current_assignment[m], m] for m in current_moduli]
