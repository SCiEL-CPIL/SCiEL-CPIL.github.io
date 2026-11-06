import math

def construct():
    max_mod = 128
    
    def prime_factors(n):
        pf = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                pf.add(d)
                n //= d
            d += 1
        if n > 1:
            pf.add(n)
        return pf

    moduli = list(range(2, max_mod + 1))
    pf_map = {m: prime_factors(m) for m in moduli}
    
    # Score moduli: prefer those with more prime factors and smaller primes
    # This tends to create moduli with large GCDs, which are easier to satisfy
    def score(m):
        pfs = pf_map[m]
        if not pfs:
            return 0
        # Weight: more factors is better, smaller primes is better
        s = len(pfs) * 10 - sum(pfs)
        return s

    sorted_moduli = sorted(moduli, key=score, reverse=True)
    
    # We will try to select a subset of moduli and solve the CSP.
    # Start with a large candidate set, reduce if CSP fails.
    
    # Heuristic: The max clique size is likely around 20-40.
    # Let's try to solve for the top K moduli.
    # If it fails, we reduce K.
    
    best_solution = None
    
    # Try different subset sizes
    # We know pairwise gcd must be > 1.
    # So first, filter sorted_moduli to ensure pairwise gcd > 1?
    # No, the CSP solver will handle it, but we can prune the candidate list.
    
    # Let's build a candidate list where each modulus is compatible with the previous ones in the list?
    # That's the greedy clique construction.
    
    # Greedy Clique Construction
    candidate_moduli = []
    for m in sorted_moduli:
        compatible = True
        for m_c in candidate_moduli:
            if math.gcd(m, m_c) == 1:
                compatible = False
                break
        if compatible:
            candidate_moduli.append(m)
    
    # Now we have a list of moduli that form a clique.
    # We want to assign residues to a subset of these to maximize count.
    # We'll try to solve CSP for the entire candidate list.
    # If it fails, we remove the last one and retry.
    
    # To make CSP efficient, we order variables by "most constrained".
    # In this clique, all moduli are connected.
    # Moduli with small values have smaller domains.
    # So we sort candidate_moduli by ascending value for the CSP solver.
    # However, we want to maximize the number of selected moduli.
    # If we remove a modulus, which one?
    # Probably the one that causes the most conflict.
    # Simple approach: Start with all, remove the last added (lowest score) if it fails.
    
    # Let's try solving for the full candidate list first.
    # If N is too large, CSP might be slow.
    # Limit candidate list size to, say, 30 to be safe?
    # The greedy clique might be larger.
    # Let's cap the search to the top 30 candidates.
    
    search_moduli = candidate_moduli[:30]
    
    # If search_moduli is empty, return fallback
    if not search_moduli:
        return [[1, 2]]
    
    # CSP Solver
    def solve_csp(moduli_list):
        # Variables: indices in moduli_list
        # Domains: range(m)
        # Constraints: pairwise
        
        n = len(moduli_list)
        if n == 0:
            return []
        
        # Precompute gcds
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                gcds[i][j] = math.gcd(moduli_list[i], moduli_list[j])
                gcds[j][i] = gcds[i][j]
        
        # Order variables by domain size (smallest m first)
        # This is a heuristic.
        # Actually, let's just use the order given, but maybe sort by m?
        # Sorting by m (ascending) is good for forward checking.
        
        # Let's try a simple backtracking with forward checking.
        
        assignment = [None] * n
        domains = [list(range(m)) for m in moduli_list]
        
        # Heuristic: pick variable with smallest domain
        # We'll implement a recursive solver
        
        def is_consistent(i, r):
            for j in range(i):
                if assignment[j] is not None:
                    g = gcds[i][j]
                    if (r - assignment[j]) % g == 0:
                        return False
            return True
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # Find unassigned variable with smallest domain (MRV)
            # For simplicity, we'll just iterate 0..n-1 and find the first unassigned with smallest domain?
            # Or just use fixed order?
            # Fixed order by increasing m is usually good.
            
            # Let's try to pick the next variable dynamically.
            # But that's complex. Let's stick to fixed order: sorted by m.
            
            # Wait, the input moduli_list is not sorted by m.
            # Let's sort it inside the solver?
            # No, we need to map back.
            
            # Let's just use the order of moduli_list.
            # We'll pick the first unassigned variable.
            
            # Actually, for speed, let's assume moduli_list is sorted by m ascending.
            # We will sort it before calling.
            
            # Get current variable
            var = idx
            
            # Try values in domain
            # Shuffle? No, deterministic is fine.
            
            for r in domains[var]:
                if not is_consistent(var, r):
                    continue
                
                # Check forward consistency with unassigned neighbors
                # For each unassigned j > var, check if domain[j] is not empty after removing r
                # This is expensive. Let's just do simple backtracking first.
                # If it's too slow, we can add AC-3.
                
                assignment[var] = r
                if backtrack(idx + 1):
                    return True
                assignment[var] = None
            
            return False
        
        # Sort moduli_list by value for better pruning
        sorted_indices = sorted(range(n), key=lambda i: moduli_list[i])
        sorted_mods = [moduli_list[i] for i in sorted_indices]
        
        # Re-map gcds? No, we can just use the original indices in the solver if we pass the list.
        # Let's refactor: The solver takes a list of moduli.
        
        # To keep it simple and fast, we'll use the provided list order.
        # We will sort the input list before calling solve_csp.
        
        # Start backtracking
        if backtrack(0):
            # Extract solution
            sol = []
            for i in range(n):
                sol.append([assignment[i], moduli_list[i]])
            return sol
        return None

    # Sort search_moduli by value to help CSP
    search_moduli_sorted = sorted(search_moduli)
    
    # Try to solve for the full set
    # If it fails, we can try removing elements.
    # But for now, let's just try the full set.
    # If it times out, we need to be smarter.
    # 30 variables might be too many for naive backtracking in Python.
    
    # Let's limit to 20 moduli to be safe within 60s.
    # The greedy clique of "rich" moduli might be smaller.
    
    # Let's try a set of size 20.
    target_size = 20
    current_moduli = search_moduli_sorted[:target_size]
    
    # We can also try to add more if 20 works easily?
    # Let's just solve for the top 20.
    
    sol = solve_csp(current_moduli)
    
    if sol:
        return sol
    else:
        # If CSP fails, try a smaller set or fallback
        # Try size 10
        current_moduli = search_moduli_sorted[:10]
        sol = solve_csp(current_moduli)
        if sol:
            return sol
        
        # Fallback: Greedy residue assignment on the clique
        selected = []
        for m in search_moduli:
            found = False
            for r in range(m):
                valid = True
                for (r_s, m_s) in selected:
                    if math.gcd(m, m_s) == 1:
                        valid = False
                        break
                    g = math.gcd(m, m_s)
                    if (r - r_s) % g == 0:
                        valid = False
                        break
                if valid:
                    selected.append([r, m])
                    found = True
                    break
            if not found:
                continue
        if selected:
            return selected
        return [[1, 2]]
