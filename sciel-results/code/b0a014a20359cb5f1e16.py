import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin (150s limit)
    
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
        """
        Enumerate all maximal cliques using Bron-Kerbosch with pivot.
        Uses bitmasks for speed.
        """
        neigh = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if adj[u][v]:
                    mask |= (1 << v)
            neigh[u] = mask
        
        cliques = []
        all_mask = (1 << n) - 1
        
        def expand(R, P, X):
            if not P and not X:
                cliques.append(R)
                return
            
            union = P | X
            if union:
                best_u = -1
                best_count = -1
                u = union
                while u:
                    lsb = u & (-u)
                    v = lsb.bit_length() - 1
                    u ^= lsb
                    count = bin(P & neigh[v]).count('1')
                    if count > best_count:
                        best_count = count
                        best_u = v
                u = best_u
            else:
                u = None
                
            if u is not None:
                candidates = P & ~neigh[u]
            else:
                candidates = P
                
            while candidates:
                lsb = candidates & (-candidates)
                v = lsb.bit_length() - 1
                candidates ^= lsb
                
                new_R = R | lsb
                new_P = P & neigh[v]
                new_X = X & neigh[v]
                expand(new_R, new_P, new_X)
                
                P &= ~lsb
                X |= lsb
                
        expand(0, all_mask, 0)
        return cliques

    def is_valid_sol(sol_set, cliques):
        sol_set = set(sol_set)
        for c in cliques:
            if not sol_set.intersection(c):
                return False
        return True

    def greedy_hit_set(cliques, n, weights=None):
        """
        Greedy set cover: pick vertex hitting the most uncovered cliques.
        If weights provided, use weighted count.
        """
        remaining_cliques = [set(c) for c in cliques]
        covered_cliques = [False] * len(remaining_cliques)
        selected = []
        
        if weights is None:
            weights = [1] * n
            
        while True:
            if all(covered_cliques):
                break
            
            # Count how many uncovered cliques each vertex hits
            hits = [0.0] * n
            for i, c in enumerate(remaining_cliques):
                if not covered_cliques[i]:
                    for v in c:
                        hits[v] += 1.0 / weights[v]  # Weighted by inverse weight
            
            max_hits = max(hits)
            if max_hits == 0:
                break
                
            # Pick vertex with max hits (break ties by lower index)
            v = hits.index(max_hits)
            selected.append(v)
            
            for i, c in enumerate(remaining_cliques):
                if not covered_cliques[i] and v in c:
                    covered_cliques[i] = True
                    
        return selected

    def local_search_improve(sol, cliques, n, max_iters=5000):
        """
        Try to reduce solution size by removing vertices or swapping.
        """
        current_sol = set(sol)
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try removing each vertex
            for v in list(current_sol):
                candidate = current_sol - {v}
                if is_valid_sol(candidate, cliques):
                    current_sol = candidate
                    improved = True
                    break
            
            # If no removal worked, try swapping: remove one, add another
            if not improved:
                for v_remove in list(current_sol):
                    for v_add in range(n):
                        if v_add in current_sol:
                            continue
                        candidate = (current_sol - {v_remove}) | {v_add}
                        if is_valid_sol(candidate, cliques):
                            current_sol = candidate
                            improved = True
                            break
                    if improved:
                        break
        
        return list(current_sol)

    def solve_instance(n, p, seed, time_limit=30):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        relevant_cliques = [c for c in all_cliques if bin(c).count('1') >= 2]
        
        if not relevant_cliques:
            return []
        
        cliques_as_sets = []
        for c in relevant_cliques:
            s = set()
            temp = c
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                s.add(v)
                temp ^= lsb
            cliques_as_sets.append(s)
        
        # Strategy 1: Greedy hit set
        best_sol = greedy_hit_set(cliques_as_sets, n)
        best_size = len(best_sol)
        
        # Strategy 2: Local search improvement
        improved_sol = local_search_improve(best_sol, cliques_as_sets, n)
        if len(improved_sol) < best_size:
            best_sol = improved_sol
            best_size = len(improved_sol)
        
        # Strategy 3: Try multiple greedy starts with different weightings
        for trial in range(3):
            # Randomize weights slightly
            weights = [1.0 + random.random() * 0.5 for _ in range(n)]
            sol = greedy_hit_set(cliques_as_sets, n, weights)
            sol = local_search_improve(sol, cliques_as_sets, n)
            if len(sol) < best_size:
                best_sol = sol
                best_size = len(sol)
        
        # Strategy 4: Use scipy MILP if time permits
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left > 15 and best_size > 0:
            try:
                m = len(cliques_as_sets)
                A = np.zeros((m, n), dtype=np.float64)
                for i, clique in enumerate(cliques_as_sets):
                    for v in clique:
                        A[i, v] = 1
                
                c = np.ones(n, dtype=np.float64)
                constraints = LinearConstraint(A, 1, np.inf)
                bounds_obj = Bounds(0, 1)
                
                instance_time_limit = min(time_limit, max(5, time_left - 5))
                
                res = milp(c=c, constraints=constraints, integrality=np.ones(n, dtype=np.int8), 
                           bounds=bounds_obj, 
                           options={"time_limit": instance_time_limit, "presolve": True,
                                    "mip_rel_gap": 1e-4})
                
                if res.x is not None and res.status in [1, 2]:  # Optimal or feasible
                    sol = [i for i in range(n) if res.x[i] > 0.5]
                    if is_valid_sol(sol, cliques_as_sets) and len(sol) < best_size:
                        best_sol = sol
                        best_size = len(sol)
            except Exception:
                pass
        
        return best_sol

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 10:
            break
        sol = solve_instance(n, p, seed, time_limit=min(30, max(5, time_left - 10)))
        results.append(sorted(sol))
        
        if time.time() > deadline:
             break
             
    while len(results) < 3:
         results.append([])
         
    return results
