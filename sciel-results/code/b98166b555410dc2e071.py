import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_mods = len(moduli)
    
    # Precompute gcds
    gcd_matrix = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            g = math.gcd(i, j)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    # Order moduli: Smaller moduli are more constrained (fewer residues)
    # So process them first to fail fast?
    # Or process them last?
    # If we process small ones first, we fix their residues early.
    # Then larger ones have more freedom.
    # This seems good.
    ordered_moduli = sorted(moduli)
    
    # valid_residues[m] is a set of valid residues for m
    valid_residues = [set() for _ in range(max_mod + 1)]
    for m in moduli:
        valid_residues[m] = set(range(m))
        
    # Assignment: assignment[m] = r
    assignment = [None] * (max_mod + 1)
    
    # Forward checking helper
    # When we assign m -> r, we update valid_residues for all m' > m (in ordered list)
    # Actually, we need to update for all unassigned m'.
    
    # To make backtracking efficient, we need to restore valid_residues.
    # We can store a stack of changes.
    
    def dfs(idx):
        if time.time() > deadline:
            return False, "timeout"
            
        if idx == n_mods:
            return True, []
            
        m = ordered_moduli[idx]
        
        # Get valid residues for m
        # We need to iterate over them.
        # To improve, we can pick the residue that leaves the most options for others?
        # For now, just iterate.
        
        for r in valid_residues[m]:
            # Try assigning m -> r
            assignment[m] = r
            
            # Record changes for rollback
            changes = []
            valid = True
            
            # Update future moduli
            for j in range(idx + 1, n_mods):
                m2 = ordered_moduli[j]
                g = gcd_matrix[m][m2]
                if g == 1:
                    continue
                # Remove residues r2 in valid_residues[m2] such that r2 % g == r % g
                r_mod = r % g
                to_remove = [r2 for r2 in valid_residues[m2] if r2 % g == r_mod]
                if to_remove:
                    # Store changes
                    for r2 in to_remove:
                        changes.append((m2, r2))
                        valid_residues[m2].remove(r2)
                    if not valid_residues[m2]:
                        valid = False
                        break
                        
            if valid:
                success, _ = dfs(idx + 1)
                if success:
                    return True, []
            
            # Rollback
            assignment[m] = None
            for m2, r2 in changes:
                valid_residues[m2].add(r2)
                
        return False, []

    # Run DFS
    success, _ = dfs(0)
    
    if success:
        result = [[assignment[m], m] for m in ordered_moduli]
        return result
    else:
        # If DFS failed or timed out, we need a fallback.
        # We can return a greedy solution.
        # Or try to find a large subset.
        # Given the likely success, this might not be needed.
        # But to be safe, let's implement a quick greedy fallback.
        
        # Greedy: Pick moduli in order, find any valid residue.
        selected = []
        for m in ordered_moduli:
            # Check against selected
            valid_r = None
            for r in range(m):
                ok = True
                for m_p, r_p in selected:
                    g = gcd_matrix[m][m_p]
                    if (r - r_p) % g == 0:
                        ok = False
                        break
                if ok:
                    valid_r = r
                    break
            if valid_r is not None:
                selected.append((m, valid_r))
                
        if selected:
            return [[r, m] for m, r in selected]
        else:
            return [[0, 2]]
