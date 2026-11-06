import math
import time

def construct():
    START = time.time()
    DEADLINE = START + 80
    
    MAX_MOD = 128
    
    # Candidate moduli: All numbers 2..128
    # But we know pairwise non-coprime is required.
    # We will build the set greedily.
    
    # Precompute gcds for all pairs 2..128
    gcds = [[0]*(MAX_MOD+1) for _ in range(MAX_MOD+1)]
    for i in range(2, MAX_MOD+1):
        for j in range(i+1, MAX_MOD+1):
            g = math.gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # List of moduli to consider.
    # We want to maximize count.
    # Start with a good heuristic order.
    # Moduli with many divisors are "rich" in constraints?
    # Or moduli with small values are "tight"?
    # Let's try sorting by value.
    all_mods = list(range(2, MAX_MOD+1))
    
    # We will maintain a list of selected moduli and their residues.
    # selected = [(r1, m1), (r2, m2), ...]
    
    # To speed up, we will use a CSP solver that takes a list of moduli
    # and returns a valid assignment or None.
    
    def solve_csp(moduli_list):
        """
        Returns a list of residues for the given moduli_list if a valid assignment exists,
        else None.
        """
        n = len(moduli_list)
        if n == 0:
            return []
        if n == 1:
            return [0]
            
        # Map moduli to indices
        # We need to check constraints between all pairs.
        
        # Domains as bitmasks
        # m is in moduli_list[i]
        # domain[i] has bits 0..m-1 set
        domains = [0] * n
        for i, m in enumerate(moduli_list):
            # (1 << m) - 1 sets bits 0 to m-1
            domains[i] = (1 << m) - 1
            
        # Order variables by MRV (smallest domain size)
        # Initial sizes are m. So sort by m.
        # But we can just use the order given, or sort by m.
        # Let's sort by m ascending.
        order = sorted(range(n), key=lambda i: moduli_list[i])
        
        # Precompute constraint masks
        # For each pair (i, j), we need to check r_i != r_j mod gcd
        # This is hard to precompute as a simple mask because it depends on r_i and r_j.
        # We will do it in the loop.
        
        def is_consistent(i, r_i, assignment):
            # Check if r_i is consistent with already assigned variables
            # assignment is a dict {var_index: residue}
            for j, r_j in assignment.items():
                if j == i:
                    continue
                g = gcds[moduli_list[i]][moduli_list[j]]
                if (r_i - r_j) % g == 0:
                    return False
            return True
            
        def backtrack(idx, assignment):
            if time.time() > DEADLINE:
                return None
            if idx == n:
                return assignment
                
            i = order[idx]
            
            # Get possible residues for i
            # We can iterate through bits in domains[i]
            # But we should also check consistency with assigned variables
            # And update domains of unassigned variables?
            # For simplicity, just check consistency.
            # MRV is handled by the fixed order.
            
            # Optimization: Try residues that are "least constraining"?
            # For now, just iterate 0 to m-1.
            m_i = moduli_list[i]
            
            for r in range(m_i):
                if not (domains[i] & (1 << r)):
                    continue
                if is_consistent(i, r, assignment):
                    assignment[i] = r
                    
                    # Forward Checking
                    valid = True
                    saved = {}
                    for k in order[idx+1:]:
                        if k in assignment:
                            continue
                        # Check if domain of k becomes empty
                        new_domain = 0
                        m_k = moduli_list[k]
                        g_ik = gcds[moduli_list[i]][moduli_list[k]]
                        
                        # Filter domain of k
                        # r_k must not be congruent to r_i mod g_ik
                        # And must not be congruent to any other assigned r_j mod g_jk
                        
                        # This is O(m_k * assigned_count)
                        # m_k <= 128, assigned_count <= 64.
                        # 128 * 64 = 8192 ops.
                        # This might be slow if done for every k.
                        
                        # Let's just check if ANY value in domain[k] is valid.
                        # If yes, keep domain[k] as is? No, we should prune.
                        # But pruning is expensive.
                        # Let's just check consistency for the specific value we pick?
                        # No, we need to know if a solution exists.
                        
                        # Simple FC: Check if there exists at least one valid value in domain[k].
                        # If not, backtrack.
                        found_valid = False
                        # Iterate through bits in domain[k]
                        dom_k = domains[k]
                        while dom_k:
                            lsb = dom_k & -dom_k
                            r_k = lsb.bit_length() - 1
                            dom_k -= lsb
                            
                            ok = True
                            # Check against i
                            if (r_k - r) % g_ik == 0:
                                ok = False
                            else:
                                # Check against other assigned
                                for j, r_j in assignment.items():
                                    if j == k or j == i:
                                        continue
                                    g_jk = gcds[moduli_list[j]][moduli_list[k]]
                                    if (r_k - r_j) % g_jk == 0:
                                        ok = False
                                        break
                            if ok:
                                found_valid = True
                                break
                                
                        if not found_valid:
                            valid = False
                            break
                            
                    if valid:
                        result = backtrack(idx + 1, assignment)
                        if result is not None:
                            return result
                            
                    # Undo
                    del assignment[i]
                    
            return None

        return backtrack(0, {})

    # Greedy construction
    selected_mods = []
    best_solution = None
    
    # Sort moduli. Which order?
    # Let's try descending. Larger moduli have more flexibility.
    # Or ascending?
    # Let's try a mix: sort by number of divisors?
    # Let's just try ascending first.
    
    for m in sorted(all_mods):
        if time.time() > DEADLINE - 5:
            break
        # Try to add m
        new_mods = selected_mods + [m]
        sol = solve_csp(new_mods)
        if sol is not None:
            selected_mods = new_mods
            best_solution = sol
        # Else skip m
            
    # If we didn't find anything, return empty? No, must be nonempty.
    # The first modulus will always work.
    
    if not selected_mods:
        return [[0, 2]]
        
    # Convert to [residue, modulus] pairs
    result = []
    for i, m in enumerate(selected_mods):
        result.append([best_solution[i], m])
        
    return result
