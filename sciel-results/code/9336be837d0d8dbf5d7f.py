import math
import time

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    max_m = 200
    
    # Precompute gcd table
    gcd_table = [[0]*(max_m+1) for _ in range(max_m+1)]
    for i in range(20, max_m+1):
        for j in range(i, max_m+1):
            gcd_table[i][j] = math.gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    def is_compatible(m1, r1, m2, r2):
        """Check if (m1, r1) and (m2, r2) are compatible (disjoint)."""
        g = gcd_table[m1][m2]
        return (r1 % g) != (r2 % g)
    
    def score(selected):
        """Compute score of a solution (dict: m -> r)."""
        return sum(1.0/m for m in selected)
    
    # Phase 1: Greedy construction - pick first compatible residue
    selected = {}
    
    for m in range(20, max_m+1):
        if time.time() > deadline - 30:
            break
        
        for r in range(m):
            compatible = True
            for m2, r2 in selected.items():
                if not is_compatible(m, r, m2, r2):
                    compatible = False
                    break
            if compatible:
                selected[m] = r
                break
    
    current_score = score(selected)
    
    # Phase 2: Try to add missing moduli
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        
        improved = False
        for m in range(20, max_m+1):
            if m in selected:
                continue
            for r in range(m):
                compatible = True
                for m2, r2 in selected.items():
                    if not is_compatible(m, r, m2, r2):
                        compatible = False
                        break
                if compatible:
                    selected[m] = r
                    improved = True
                    break
        
        if not improved:
            break
    
    current_score = score(selected)
    
    # Phase 3: Local search with swaps
    for iteration in range(30):
        if time.time() > deadline - 10:
            break
        
        improved = False
        
        for m_add in range(20, max_m+1):
            if m_add in selected:
                continue
            
            for r_add in range(m_add):
                # Find which selected moduli conflict with (m_add, r_add)
                conflicts = []
                for m2, r2 in selected.items():
                    if not is_compatible(m_add, r_add, m2, r2):
                        conflicts.append(m2)
                
                if len(conflicts) == 0:
                    # Can add directly
                    selected[m_add] = r_add
                    improved = True
                    break
                
                if len(conflicts) > 1:
                    continue
                
                # Exactly one conflict: try swapping
                m_remove = conflicts[0]
                r_remove = selected[m_remove]
                
                old_score = score(selected)
                del selected[m_remove]
                selected[m_add] = r_add
                new_score = score(selected)
                
                if new_score > old_score:
                    improved = True
                    break
                else:
                    # Revert
                    del selected[m_add]
                    selected[m_remove] = r_remove
        
        if not improved:
            break
    
    return [[m, selected[m]] for m in sorted(selected.keys())]
