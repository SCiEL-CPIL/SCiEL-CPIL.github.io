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
    
    # Heuristic: Sort moduli by number of prime factors (descending) and value (ascending)
    def score(m):
        pfs = prime_factors(m)
        return (-len(pfs), m)
        
    sorted_moduli = sorted(moduli, key=score)
    
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
            
    # CSP Solver
    def solve_csp(moduli_list):
        n = len(moduli_list)
        if n == 0:
            return []
        
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                gcds[i][j] = math.gcd(moduli_list[i], moduli_list[j])
                gcds[j][i] = gcds[i][j]
                
        domains = [set(range(m)) for m in moduli_list]
        
        def ac3(domains, gcds, n, initial_queue=None):
            if initial_queue is None:
                queue = [(i, j) for i in range(n) for j in range(n) if i != j]
            else:
                queue = list(initial_queue)
                
            while queue:
                i, j = queue.pop(0)
                g = gcds[i][j]
                if g == 1:
                    domains[i].clear()
                    if not domains[i]:
                        return False
                    continue
                    
                new_domain_i = set()
                for r_i in domains[i]:
                    has_support = False
                    for r_j in domains[j]:
                        if (r_i - r_j) % g != 0:
                            has_support = True
                            break
                    if has_support:
                        new_domain_i.add(r_i)
                
                if new_domain_i != domains[i]:
                    domains[i] = new_domain_i
                    if not domains[i]:
                        return False
                    for k in range(n):
                        if k != i and k != j:
                            queue.append((k, i))
            return True

        if not ac3(domains, gcds, n):
            return None
            
        assignment = [None] * n
        
        def backtrack():
            # MRV
            min_var = -1
            min_size = float('inf')
            for v in range(n):
                if assignment[v] is None:
                    if len(domains[v]) < min_size:
                        min_size = len(domains[v])
                        min_var = v
                        
            if min_var == -1:
                return True
                
            values = list(domains[min_var])
            for r in values:
                consistent = True
                for j in range(n):
                    if assignment[j] is not None:
                        g = gcds[min_var][j]
                        if (r - assignment[j]) % g == 0:
                            consistent = False
                            break
                            
                if not consistent:
                    continue
                
                old_domains = [d.copy() for d in domains]
                assignment[min_var] = r
                domains[min_var] = {r}
                
                # Determine which arcs to check
                # Neighbors of min_var whose domains might be pruned
                # And arcs from those neighbors to others
                # Simple approach: Run AC-3 on all arcs involving changed domains
                # domains[min_var] changed. So arcs (min_var, j) and (j, min_var) are affected.
                # But AC-3 propagates.
                # Let's just run AC-3 with a queue of (k, min_var) and (min_var, k) for all k?
                # Actually, since we only assigned one variable, the only domains that *can* change are neighbors.
                # So queue = [(j, k) for j in range(n) if j != min_var for k in range(n) if k != j and k != min_var]
                # This is still O(N^2).
                
                # Optimization: Only check arcs (j, min_var) to prune j, then arcs (k, j) if j changed.
                # Let's use a simpler queue: all arcs involving min_var?
                # No, standard AC-3: queue all arcs (i, j) where i's domain might change.
                # i's domain changes if j is assigned or j's domain changed.
                # Here min_var is assigned. So all j != min_var might change.
                # So queue = [(j, k) for j in range(n) if j != min_var for k in range(n) if k != j]
                
                q = []
                for j in range(n):
                    if j != min_var:
                        for k in range(n):
                            if k != j and k != min_var:
                                q.append((j, k))
                # Also need to check (min_var, j)? No, min_var is fixed.
                # We need to prune domains of j based on min_var.
                # So we should include (j, min_var) in the queue?
                # AC-3 arc (i, j) checks if i has support in j.
                # We want to prune j based on min_var. So we need arc (j, min_var).
                # So add (j, min_var) for all j != min_var.
                for j in range(n):
                    if j != min_var:
                        q.append((j, min_var))
                        
                if ac3(domains, gcds, n, q):
                    if backtrack():
                        return True
                        
                assignment[min_var] = None
                domains = old_domains
                
            return False
            
        if backtrack():
            return [[assignment[i], moduli_list[i]] for i in range(n)]
        return None

    current_set = candidate_moduli[:]
    
    for i in range(len(current_set), 0, -1):
        subset = current_set[:i]
        sol = solve_csp(subset)
        if sol:
            return sol
            
    return [[1, 2]]
