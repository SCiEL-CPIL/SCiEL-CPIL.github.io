import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_mods = len(moduli)
    
    # Precompute gcds
    gcds = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for i in range(2, max_mod + 1):
        for j in range(2, max_mod + 1):
            gcds[i][j] = gcd(i, j)
            
    # Map modulus to index in the list for easy access
    # We will work with the list of moduli directly.
    
    # CSP Solver
    # Variables: indices 0..126 corresponding to moduli[0..126]
    # Domains: list of lists, initially range(moduli[i])
    
    def solve_csp(order, time_limit=5.0):
        """
        order: list of indices into moduli, defining the variable order.
        Returns (success, solution_dict)
        """
        n = len(order)
        # domains[i] is a set of valid residues for variable at position i in 'order'
        # We use lists for domains, and check validity on the fly or maintain a boolean array.
        # Given small domains, a list of booleans or a set is fine.
        
        # To speed up, we can precompute the constraint masks?
        # For variable i (modulus m_i) and value v_i, which values v_j are forbidden for variable j (modulus m_j)?
        # Forbidden if v_j % gcd(m_i, m_j) == v_i % gcd(m_i, m_j).
        
        # We will use a simple recursive backtracker with forward checking.
        
        assignment = [None] * n
        domains = [set(range(moduli[order[i]])) for i in range(n)]
        
        best_depth = 0
        best_solution = {}
        
        def backtrack(idx):
            nonlocal best_depth, best_solution
            if time.time() > start_time + time_limit:
                return False
            
            if idx == n:
                best_depth = n
                best_solution = {moduli[order[i]]: assignment[i] for i in range(n)}
                return True
                
            if idx > best_depth:
                best_depth = idx
                best_solution = {moduli[order[i]]: assignment[i] for i in range(idx)}
            
            var_idx = order[idx]
            m_i = moduli[var_idx]
            
            # Get valid values for m_i from current domain
            # We maintain domains[idx] as the set of values consistent with previous assignments.
            # But we also need to check consistency with assigned variables.
            # Forward checking: when we assign a value, we prune domains of future variables.
            
            for v in list(domains[idx]):
                # Check if v is consistent with all assigned variables (0..idx-1)
                # Actually, if we prune domains correctly, domains[idx] should only contain valid values.
                # Let's verify.
                
                # Assign
                assignment[idx] = v
                
                # Forward checking: prune domains for idx+1..n-1
                saved_domains = []
                consistent = True
                for j in range(idx + 1, n):
                    var_j = order[j]
                    m_j = moduli[var_j]
                    g = gcds[m_i][m_j]
                    r_i_mod_g = v % g
                    
                    # Filter domain of j
                    new_domain = set()
                    for v_j in domains[j]:
                        if v_j % g != r_i_mod_g:
                            new_domain.add(v_j)
                    
                    saved_domains.append((j, domains[j]))
                    domains[j] = new_domain
                    if not new_domain:
                        consistent = False
                        break
                
                if consistent:
                    if backtrack(idx + 1):
                        return True
                
                # Undo forward checking
                for j, orig_domain in saved_domains:
                    domains[j] = orig_domain
                
                # Unassign
                assignment[idx] = None
                
            return False
            
        backtrack(0)
        
        if best_depth == n:
            return True, best_solution
        else:
            return False, best_solution

    # Try different orders
    # 1. Natural order
    # 2. Reverse natural order
    # 3. Random orders
    
    best_solution = {}
    
    # Try natural order first with a short time limit
    order1 = list(range(n_mods))
    success, sol1 = solve_csp(order1, 2.0)
    if success:
        return [[sol1[m], m] for m in moduli]
    else:
        best_solution = sol1
        
    # Try reverse order
    order2 = list(range(n_mods - 1, -1, -1))
    success, sol2 = solve_csp(order2, 2.0)
    if success:
        return [[sol2[m], m] for m in moduli]
    else:
        if len(sol2) > len(best_solution):
            best_solution = sol2
            
    # Try a few random orders
    import random
    for _ in range(10):
        if time.time() > start_time + 80.0:
            break
        order3 = list(range(n_mods))
        random.shuffle(order3)
        success, sol3 = solve_csp(order3, 2.0)
        if success:
            return [[sol3[m], m] for m in moduli]
        else:
            if len(sol3) > len(best_solution):
                best_solution = sol3
                
    # Fallback to greedy if CSP didn't find a full solution and partials are small
    # But CSP partials are likely better than greedy?
    # Greedy from previous code was ~124.
    # If CSP found < 124, we should probably run the old greedy to be safe.
    
    # Let's implement a fast greedy as fallback
    def fast_greedy():
        selected = []
        moduli_set = set(moduli)
        # Sort by divisibility or something?
        # Just try all moduli in random order, pick first valid residue
        random.shuffle(moduli)
        for m in moduli:
            # Find valid residue
            for r in range(m):
                valid = True
                for r_j, m_j in selected:
                    g = gcds[m][m_j]
                    if r % g == r_j % g:
                        valid = False
                        break
                if valid:
                    selected.append((r, m))
                    break
        return selected
        
    greedy_sol = fast_greedy()
    
    if len(greedy_sol) > len(best_solution):
        best_solution = {m: r for r, m in greedy_sol}
        
    # Construct result from best_solution
    # Ensure all moduli in best_solution are valid
    result = []
    for m, r in best_solution.items():
        result.append([r, m])
        
    if not result:
        result = [[0, 2]]
        
    return result
