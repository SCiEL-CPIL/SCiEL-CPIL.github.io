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
        
        # Preprocessing: identify vertices that must be in any solution
        # If a vertex v is in a clique C and removing v from C leaves a sub-clique
        # that is also maximal, then v might be forced... but this is complex.
        # Instead, use a simpler reduction: for each edge (u,v), we need at least one.
        
        m = len(relevant_cliques)
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
        
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            constraints = LinearConstraint(A, 1, np.inf)
            bounds_obj = Bounds(0, 1)
            
            elapsed = time.time() - start_time
            time_left = deadline - elapsed
            if time_left <= 5:
                return greedy_hit_set(relevant_cliques, n)
            
            instance_time_limit = max(5, (time_left - 5) / 3)
            
            res = milp(c=c, constraints=constraints, integrality=np.ones(n, dtype=np.int8), 
                       bounds=bounds_obj, 
                       options={"time_limit": instance_time_limit, "presolve": True, 
                                "mip_rel_gap": 0.0, "mip_abs_gap": 0.0})
            
            success = False
            if hasattr(res, 'success') and res.success:
                success = True
            elif hasattr(res, 'status') and res.status in (0, 1):
                success = True
                
            if success and res.x is not None:
                sol = [i for i in range(n) if res.x[i] > 0.5]
                if is_valid_sol(sol, relevant_cliques):
                    elapsed = time.time() - start_time
                    remaining_time = deadline - elapsed - 5
                    if remaining_time > 2:
                        sol = local_search_2for1(sol, relevant_cliques, n, time_budget=min(remaining_time, 10))
                    return sol
            
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
        # Greedy set cover with forced vertex identification
        clique_sets = [set(c) for c in cliques]
        m = len(clique_sets)
        
        # Precompute total appearances for tie-breaking
        total_appearances = [0] * n
        for c in clique_sets:
            for v in c:
                total_appearances[v] += 1
        
        # Identify forced vertices: if a vertex appears in all cliques of size 2
        # that contain it, and removing it would leave some edge uncovered...
        # Actually, let's use a different approach: iterative refinement
        
        covered = [False] * m
        selected = []
        
        # First pass: handle edges (cliques of size 2) greedily by matching
        edge_cliques = [(i, c) for i, c in enumerate(clique_sets) if len(c) == 2]
        
        # For edges, we can use a greedy matching approach
        edge_covered = [False] * len(edge_cliques)
        
        # Count how many edges each vertex is in
        edge_degree = [0] * n
        for i, c in edge_cliques:
            for v in c:
                edge_degree[v] += 1
        
        # Greedy: pick vertices that cover the most uncovered edges
        while True:
            if all(edge_covered):
                break
            
            best_v = -1
            best_count = 0
            for v in range(n):
                if v in selected:
                    continue
                count = 0
                for j, (idx, c) in enumerate(edge_cliques):
                    if not edge_covered[j] and v in c:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1:
                break
            
            selected.append(best_v)
            for j, (idx, c) in enumerate(edge_cliques):
                if not edge_covered[j] and best_v in c:
                    edge_covered[j] = True
            
            # Also mark non-edge cliques as covered if best_v is in them
            for i in range(m):
                if not covered[i] and best_v in clique_sets[i]:
                    covered[i] = True
        
        # Second pass: handle remaining cliques (size >= 3)
        remaining_cliques = [(i, c) for i, c in enumerate(clique_sets) if not covered[i]]
        
        while remaining_cliques:
            # Pick vertex that covers the most remaining cliques
            hits = [0] * n
            for i, c in remaining_cliques:
                for v in c:
                    hits[v] += 1
            
            max_hits = max(hits)
            if max_hits == 0:
                break
            
            candidates = [v for v in range(n) if hits[v] == max_hits and v not in selected]
            
            if not candidates:
                break
            
            # Tie-break: prefer vertices with higher edge_degree (they're more "central")
            v = max(candidates, key=lambda x: (total_appearances[x], edge_degree[x], -x))
            selected.append(v)
            
            # Update covered status
            new_remaining = []
            for i, c in remaining_cliques:
                if v in c:
                    covered[i] = True
                else:
                    new_remaining.append((i, c))
            remaining_cliques = new_remaining
        
        return selected

    def local_search_2for1(sol, cliques, n, time_budget=5.0):
        """Exhaustive 2-for-1 swap with efficient bookkeeping"""
        start = time.time()
        sol_set = set(sol)
        m = len(cliques)
        
        # For each clique, track which sol vertices cover it
        clique_coverers = [[] for _ in range(m)]
        clique_sets = [set(c) for c in cliques]
        for i, cset in enumerate(clique_sets):
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
                    critical = []
                    for i in range(m):
                        cov = clique_coverers[i]
                        if len(cov) > 0 and all(v in (a, b) for v in cov):
                            critical.append(i)
                    
                    if not critical:
                        continue
                    
                    # Find intersection of critical cliques
                    inter = clique_sets[critical[0]]
                    for i in critical[1:]:
                        inter = inter & clique_sets[i]
                        if not inter:
                            break
                    
                    if not inter:
                        continue
                    
                    # Try each candidate w in inter
                    for w in inter:
                        if w == a or w == b:
                            continue
                        # Check: after removing a and b, adding w, are all cliques still covered?
                        valid = True
                        for i in range(m):
                            cset = clique_sets[i]
                            if not (cset & (sol_set - {a, b})) and w not in cset:
                                valid = False
                                break
                        if valid:
                            sol_set = (sol_set - {a, b}) | {w}
                            # Rebuild clique_coverers efficiently
                            for i in range(m):
                                cset = clique_sets[i]
                                clique_coverers[i] = [v for v in sol_set if v in cset]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Also try 1-for-1 swaps: remove one vertex, add another
        sol_list = sorted(sol_set)
        for a in sol_list:
            if (time.time() - start) > time_budget:
                break
            # Find cliques that are covered ONLY by a
            critical = []
            for i in range(m):
                cov = clique_coverers[i]
                if len(cov) == 1 and cov[0] == a:
                    critical.append(i)
            
            if not critical:
                continue
            
            # Find intersection of critical cliques
            inter = clique_sets[critical[0]]
            for i in critical[1:]:
                inter = inter & clique_sets[i]
                if not inter:
                    break
            
            if not inter:
                continue
            
            # Try each candidate w in inter that's not in sol_set
            for w in inter:
                if w in sol_set:
                    continue
                new_sol = (sol_set - {a}) | {w}
                valid = True
                for i in range(m):
                    cset = clique_sets[i]
                    if not (cset & new_sol):
                        valid = False
                        break
                if valid:
                    sol_set = new_sol
                    for i in range(m):
                        cset = clique_sets[i]
                        clique_coverers[i] = [v for v in sol_set if v in cset]
                    improved = True
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
