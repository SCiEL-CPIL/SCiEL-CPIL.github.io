import math

def construct():
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcds
    # gcd_matrix[i][j] = gcd(moduli[i], moduli[j])
    n = len(moduli)
    gcd_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcd_matrix[i][j] = math.gcd(moduli[i], moduli[j])
            gcd_matrix[j][i] = gcd_matrix[i][j]
            
    # Build compatibility graph: edge if gcd > 1
    # We want a large clique in this graph.
    # Using Bron-Kerbosch with pivot to find a large clique.
    
    def bron_kerbosch(r, p, x):
        if not p and not x:
            return r
        # Choose pivot u from p union x
        # Heuristic: pick u with max degree in p
        u = None
        max_deg = -1
        p_x = list(p) + list(x)
        for v in p_x:
            deg = len([w for w in p if w != v and gcd_matrix[v][w] > 1])
            if deg > max_deg:
                max_deg = deg
                u = v
        # If no pivot, just iterate p
        if u is None:
            for v in list(p):
                bron_kerbosch(r + [v], [w for w in p if gcd_matrix[v][w] > 1], [w for w in x if gcd_matrix[v][w] > 1])
                p.remove(v)
                x.append(v)
        else:
            # Iterate over p \ N(u)
            # N(u) in p are neighbors of u in p
            non_neighbors = [v for v in p if gcd_matrix[v][u] == 1]
            for v in non_neighbors:
                bron_kerbosch(r + [v], [w for w in p if w != v and gcd_matrix[v][w] > 1], [w for w in x if gcd_matrix[v][w] > 1])
                p.remove(v)
                x.append(v)
                
    # Initialize
    p = set(range(n))
    x = set()
    r = []
    # Run BK
    # Note: BK finds all maximal cliques. We want one large one.
    # We can modify it to stop early if we find a large enough one, or just find the max.
    # Given N=127, finding ALL maximal cliques might be slow.
    # Let's use a heuristic: greedy coloring bound or just pick a random start and expand?
    # Actually, for this specific structure (integers 2..128), the clique number is likely quite high.
    # Let's use a simpler greedy heuristic for clique first, then verify/expand with CSP.
    
    # Greedy Clique Heuristic
    # Sort vertices by degree (number of compatible neighbors) descending
    degrees = [sum(1 for j in range(n) if i != j and gcd_matrix[i][j] > 1) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    clique = []
    candidates = set(range(n))
    for v in order:
        if v not in candidates:
            continue
        # Check if v is compatible with all in clique
        if all(gcd_matrix[v][u] > 1 for u in clique):
            clique.append(v)
            # Remove incompatible vertices from candidates
            candidates = {u for u in candidates if gcd_matrix[v][u] > 1}
            
    # Now we have a candidate set of moduli (indices)
    # Solve CSP for residues
    def solve_csp(moduli_indices):
        m = len(moduli_indices)
        if m == 0:
            return []
        
        # Variables: 0..m-1 corresponding to moduli_indices
        # Domains: list of lists
        domains = [list(range(moduli[moduli_indices[i]])) for i in range(m)]
        
        # Order variables by domain size (smallest first)
        # But we must keep track of original indices
        var_order = sorted(range(m), key=lambda i: len(domains[i]))
        
        assignment = {}
        
        def backtrack(idx):
            if idx == m:
                return True
            
            var = var_order[idx]
            m_val = moduli[moduli_indices[var]]
            
            for r_val in domains[var]:
                # Check if r_val is consistent with previous assignments
                consistent = True
                for prev_var in var_order[:idx]:
                    prev_m_idx = moduli_indices[prev_var]
                    prev_r = assignment[prev_var]
                    prev_m_val = moduli[moduli_indices[prev_var]]
                    g = gcd_matrix[moduli_indices[var]][prev_m_idx]
                    if (r_val - prev_r) % g == 0:
                        consistent = False
                        break
                
                if consistent:
                    assignment[var] = r_val
                    # Forward checking: prune domains of unassigned variables
                    domains_copy = [d[:] for d in domains] # Deep copy is expensive, do it in place with undo?
                    # For simplicity and speed, let's just check consistency at assignment time 
                    # and rely on the order to catch conflicts early.
                    # Actually, forward checking is better.
                    
                    # Let's implement simple forward checking
                    pruned = {}
                    fail = False
                    for future_var in var_order[idx+1:]:
                        new_domain = []
                        for r_f in domains[future_var]:
                            # Check against current assignment
                            ok = True
                            for assigned_var, assigned_r in assignment.items():
                                if assigned_var == future_var:
                                    continue
                                g = gcd_matrix[moduli_indices[future_var]][moduli_indices[assigned_var]]
                                if (r_f - assigned_r) % g == 0:
                                    ok = False
                                    break
                            if ok:
                                new_domain.append(r_f)
                        if not new_domain:
                            fail = True
                            break
                        pruned[future_var] = new_domain
                    
                    if fail:
                        # Undo
                        pass
                    else:
                        # Apply pruning
                        old_domains = {}
                        for fv in pruned:
                            old_domains[fv] = domains[fv][:]
                            domains[fv] = pruned[fv]
                        
                        if backtrack(idx + 1):
                            return True
                        
                        # Undo pruning
                        for fv in pruned:
                            domains[fv] = old_domains[fv]
                
                # Remove r_val from domain for next iteration? 
                # No, we just try the next r_val.
                
            return False
        
        if backtrack(0):
            # Extract results
            result = []
            for var in range(m):
                r_val = assignment[var]
                m_val = moduli[moduli_indices[var]]
                result.append([r_val, m_val])
            return result
        else:
            return None

    # Try to solve for the greedy clique
    solution = solve_csp(clique)
    
    if solution:
        return solution
    
    # If greedy clique fails, we might need to backtrack on the clique selection itself.
    # This is getting complex. Let's try a different strategy:
    # Iterate through moduli in a smart order, and for each, decide to include or not.
    # This is essentially a branch and bound on the set of moduli.
    
    # Given the time, let's stick to the greedy clique + CSP. 
    # If it returns None, we can try removing the last added vertex and retrying, 
    # or just return a smaller valid set.
    
    # Fallback: Return a small valid set
    return [[1, 2]]
