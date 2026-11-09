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
        # Bitmask-based Bron-Kerbosch for speed
        neighbors = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if u != v and adj[u][v]:
                    mask |= (1 << v)
            neighbors[u] = mask
        
        cliques = []
        
        def expand(R, P, X):
            if P == 0 and X == 0:
                # Convert bitmask R to list
                verts = [i for i in range(n) if (R >> i) & 1]
                if len(verts) >= 2:
                    cliques.append(verts)
                return
            
            # Pivot selection: pick u in P|X maximizing popcount(P & N(u))
            union = P | X
            if union:
                best_u = -1
                best_count = -1
                tmp = union
                while tmp:
                    lsb = tmp & (-tmp)
                    u = lsb.bit_length() - 1
                    tmp ^= lsb
                    cnt = (P & neighbors[u]).bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_u = u
                candidates = P & ~neighbors[best_u]
            else:
                candidates = P
            
            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb
                new_R = R | lsb
                new_P = P & neighbors[v]
                new_X = X & neighbors[v]
                expand(new_R, new_P, new_X)
                P &= ~lsb
                X |= lsb
        
        expand(0, (1 << n) - 1, 0)
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
                    # Polish with local search
                    elapsed = time.time() - start_time
                    remaining_time = deadline - elapsed - 5
                    if remaining_time > 2:
                        sol = local_search_2for1(sol, relevant_cliques, n, time_budget=min(remaining_time, 10))
                    return sol
            
            # If not successful or invalid, try incumbent from res.x if available
            if res.x is not None:
                sol = [i for i in range(n) if res.x[i] > 0.5]
                if is_valid_sol(sol, relevant_cliques):
                    elapsed = time.time() - start_time
                    remaining_time = deadline - elapsed - 5
                    if remaining_time > 2:
                        sol = local_search_2for1(sol, relevant_cliques, n, time_budget=min(remaining_time, 10))
                    return sol
        except Exception:
            pass
            
        sol = greedy_hit_set(relevant_cliques, n)
        # Polish greedy solution with local search
        elapsed = time.time() - start_time
        remaining_time = deadline - elapsed - 5
        if remaining_time > 2:
            sol = local_search_2for1(sol, relevant_cliques, n, time_budget=min(remaining_time, 10))
        return sol

    def is_valid_sol(sol_set, cliques):
        sol_set = set(sol_set)
        for c in cliques:
            if not sol_set.intersection(c):
                return False
        return True

    def greedy_hit_set(cliques, n):
        # Greedy set cover
        clique_sets = [set(c) for c in cliques]
        covered = [False] * len(clique_sets)
        selected = []
        
        while True:
            if all(covered):
                break
            
            hits = [0] * n
            for i, c in enumerate(clique_sets):
                if not covered[i]:
                    for v in c:
                        hits[v] += 1
            
            max_hits = max(hits)
            if max_hits == 0:
                break
            
            candidates = [v for v in range(n) if hits[v] == max_hits]
            
            # Tie-break: prefer vertices in fewer total cliques
            total_appearances = [0] * n
            for c in clique_sets:
                for v in c:
                    total_appearances[v] += 1
            
            v = min(candidates, key=lambda x: (total_appearances[x], x))
            selected.append(v)
            
            for i, c in enumerate(clique_sets):
                if not covered[i] and v in c:
                    covered[i] = True
                    
        return selected

    def local_search_2for1(sol, cliques, n, time_budget=5.0):
        """Exhaustive 2-for-1 swap: try replacing two selected vertices with one
        that covers all cliques that were only covered by those two."""
        start = time.time()
        sol_set = set(sol)
        m = len(cliques)
        
        # For each clique, track which sol vertices cover it
        clique_coverers = [[] for _ in range(m)]
        for i, c in enumerate(cliques):
            cset = set(c)
            for v in sol_set:
                if v in cset:
                    clique_coverers[i].append(v)
        
        improved = True
        while improved and (time.time() - start) < time_budget:
            improved = False
            sol_list = sorted(sol_set)
            for a_idx in range(len(sol_list)):
                if (time.time() - start) > time_budget:
                    break
                a = sol_list[a_idx]
                for b_idx in range(a_idx + 1, len(sol_list)):
                    b = sol_list[b_idx]
                    # Find cliques that are covered ONLY by a and/or b
                    # i.e., cliques where the set of sol_coverers is subset of {a, b}
                    # and at least one of a,b is in the clique
                    critical = []
                    for i in range(m):
                        cov = clique_coverers[i]
                        # Check if all coverers are in {a, b}
                        all_in = all(v in (a, b) for v in cov)
                        if all_in and len(cov) > 0:
                            critical.append(i)
                    
                    if not critical:
                        continue
                    
                    # We need a single vertex w (not in sol_set, or could be a or b)
                    # that covers all critical cliques
                    # Actually w must be in the intersection of all critical cliques
                    # and w must not be a or b (otherwise we'd just keep it)
                    # Wait: if w == a, then removing b and keeping a still covers
                    # cliques that had a as coverer. But cliques that only had b
                    # would lose coverage. So w must cover ALL critical cliques.
                    
                    # The vertex w must be in every critical clique
                    # Find intersection of critical cliques
                    if not critical:
                        continue
                    inter = set(cliques[critical[0]])
                    for i in critical[1:]:
                        inter &= set(cliques[i])
                        if not inter:
                            break
                    
                    if not inter:
                        continue
                    
                    # Try each candidate w in inter
                    for w in inter:
                        if w == a or w == b:
                            continue
                        # Check: after removing a and b, adding w, are all cliques still covered?
                        # A clique is covered if it has at least one vertex in (sol_set - {a,b}) ∪ {w}
                        new_sol = (sol_set - {a, b}) | {w}
                        valid = True
                        for i in range(m):
                            cset = set(cliques[i])
                            if not (new_sol & cset):
                                valid = False
                                break
                        if valid:
                            sol_set = new_sol
                            # Rebuild clique_coverers
                            for i in range(m):
                                cset = set(cliques[i])
                                clique_coverers[i] = [v for v in sol_set if v in cset]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return sorted(sol_set)

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
