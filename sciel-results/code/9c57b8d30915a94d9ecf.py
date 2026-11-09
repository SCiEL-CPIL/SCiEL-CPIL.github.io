import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        # Returns list of maximal cliques (sets of vertices)
        cliques = []
        
        # Precompute neighbor sets for faster intersection
        neighbors = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if adj[u][v]:
                    neighbors[u].add(v)
        
        def expand(R, P, X):
            if not P and not X:
                cliques.append(frozenset(R))
                return
            # Pivot selection: choose u in P union X that maximizes |P ∩ N(u)|
            union = P | X
            if union:
                # Find pivot that maximizes |P ∩ N(u)|
                best_u = None
                best_count = -1
                for u in union:
                    count = len(P & neighbors[u])
                    if count > best_count:
                        best_count = count
                        best_u = u
                u = best_u
            else:
                u = None
                
            # Iterate over P \ N(u)
            if u is not None:
                candidates = P - neighbors[u]
            else:
                candidates = P
                
            for v in candidates:
                new_R = R | {v}
                new_P = P & neighbors[v]
                new_X = X & neighbors[v]
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
                
        expand(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        # Filter cliques with size >= 2
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []
        
        # Formulate ILP
        # Minimize sum(x_i)
        # s.t. sum_{j in C} x_j >= 1 for all C in relevant_cliques
        # 0 <= x_i <= 1, x_i integer
        
        m = len(relevant_cliques)
        # A is m x n - use sparse representation for efficiency
        from scipy.sparse import csr_matrix
        row_idx = []
        col_idx = []
        data = []
        for i, clique in enumerate(relevant_cliques):
            for v in clique:
                row_idx.append(i)
                col_idx.append(v)
                data.append(1)
        
        A = csr_matrix((data, (row_idx, col_idx)), shape=(m, n), dtype=np.int8)
                
        c = np.ones(n, dtype=np.float64)
        
        # Bounds: (0, 1)
        bounds = [(0, 1)] * n
        
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            constraints = LinearConstraint(A, 1, np.inf)
            bounds_obj = Bounds(0, 1)
            
            # Add objective bound to help presolve
            # The minimum possible is 1 if there's any clique, max is n
            # This helps the solver prune
            
            # Set time limit per instance to 30s
            # We have 3 instances, so 90s total. 
            # Check global deadline
            elapsed = time.time() - start_time
            time_left = deadline - elapsed
            if time_left <= 5:
                # Not enough time, return greedy fallback
                return greedy_hit_set(relevant_cliques, n)
            
            instance_time_limit = max(5, (time_left - 5) / 3)
            
            res = milp(c=c, constraints=constraints, integrality=np.ones(n, dtype=np.int8), 
                       bounds=bounds_obj, 
                       options={"time_limit": instance_time_limit, "presolve": True, 
                                "mip_rel_gap": 0.0, "mip_abs_gap": 0.0})
            
            # Check multiple possible success conditions
            success = False
            if hasattr(res, 'success') and res.success:
                success = True
            elif hasattr(res, 'status') and res.status in (0, 1):  # 0=optimal, 1=optimal (tolerances)
                success = True
                
            if success and res.x is not None:
                sol = [i for i in range(n) if res.x[i] > 0.5]
                if is_valid_sol(sol, relevant_cliques):
                    return sol
            
            # If not successful or invalid, try incumbent from res.x if available
            if res.x is not None:
                sol = [i for i in range(n) if res.x[i] > 0.5]
                if is_valid_sol(sol, relevant_cliques):
                    return sol
        except Exception:
            pass
            
        return greedy_hit_set(relevant_cliques, n)

    def is_valid_sol(sol_set, cliques):
        sol_set = set(sol_set)
        for c in cliques:
            if not sol_set.intersection(c):
                return False
        return True

    def greedy_hit_set(cliques, n):
        # Improved greedy set cover with better tie-breaking
        remaining_cliques = [set(c) for c in cliques]
        covered_cliques = [False] * len(remaining_cliques)
        selected = []
        
        while True:
            # Check if all covered
            if all(covered_cliques):
                break
            
            # Count how many uncovered cliques each vertex hits
            hits = [0] * n
            for i, c in enumerate(remaining_cliques):
                if not covered_cliques[i]:
                    for v in c:
                        hits[v] += 1
            
            max_hits = max(hits)
            if max_hits == 0:
                break # Should not happen if input is valid
                
            # Pick vertex with max hits; tie-break by minimum degree in remaining graph
            # (prefer vertices that appear in fewer cliques to avoid wasting coverage)
            candidates = [v for v in range(n) if hits[v] == max_hits]
            
            # Count total appearances in all cliques (not just uncovered) for tie-breaking
            total_appearances = [0] * n
            for c in remaining_cliques:
                for v in c:
                    total_appearances[v] += 1
            
            # Pick candidate with minimum total appearances (more specialized)
            v = min(candidates, key=lambda x: (total_appearances[x], x))
            selected.append(v)
            
            # Update coverage
            for i, c in enumerate(remaining_cliques):
                if not covered_cliques[i] and v in c:
                    covered_cliques[i] = True
                    
        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sorted(sol))
        
        if time.time() > deadline:
             # Fill remaining with greedy quickly if we run out of time
             # But we should have enough time.
             break
             
    # If loop broke early, ensure we have 3 results
    # This shouldn't happen with the time allocation above, but safe.
    while len(results) < 3:
         results.append([])
         
    return results
