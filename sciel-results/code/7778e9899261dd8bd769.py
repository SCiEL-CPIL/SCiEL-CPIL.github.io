import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    time_limit = 290  # Safety margin
    
    # 1. Generate Seed: Paley Graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n_seed = q
    
    # Build adjacency as bitmasks
    adj = [0] * n_seed
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
                
    def get_k4_and_indep_k4(adj, n):
        """Return list of K4s and list of independent 4-sets as tuples of vertices."""
        k4s = []
        indep_k4s = []
        for combo in itertools.combinations(range(n), 4):
            v0, v1, v2, v3 = combo
            mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3)
            
            # Check Clique
            is_clique = True
            if (adj[v0] & (mask ^ (1 << v0))) != (mask ^ (1 << v0)): is_clique = False
            elif (adj[v1] & (mask ^ (1 << v1))) != (mask ^ (1 << v1)): is_clique = False
            elif (adj[v2] & (mask ^ (1 << v2))) != (mask ^ (1 << v2)): is_clique = False
            elif (adj[v3] & (mask ^ (1 << v3))) != (mask ^ (1 << v3)): is_clique = False
            
            if is_clique:
                k4s.append(combo)
                
            # Check Independent
            is_indep = True
            if (adj[v0] & (mask ^ (1 << v0))) != 0: is_indep = False
            elif (adj[v1] & (mask ^ (1 << v1))) != 0: is_indep = False
            elif (adj[v2] & (mask ^ (1 << v2))) != 0: is_indep = False
            elif (adj[v3] & (mask ^ (1 << v3))) != 0: is_indep = False
            
            if is_indep:
                indep_k4s.append(combo)
                
        return k4s, indep_k4s

    def solve_new_vertex(k4s, indep_k4s, n_curr, time_budget):
        """
        Find adjacency vector x (list of 0/1) of length n_curr for a new vertex.
        """
        # Pre-compute incidence lists
        k4_inc = [[] for _ in range(n_curr)]
        for idx, k4 in enumerate(k4s):
            for v in k4:
                k4_inc[v].append(idx)
                
        ik4_inc = [[] for _ in range(n_curr)]
        for idx, ik4 in enumerate(indep_k4s):
            for v in ik4:
                ik4_inc[v].append(idx)
                
        def get_initial_cost(x):
            cost = 0
            for k4 in k4s:
                if x[k4[0]] and x[k4[1]] and x[k4[2]] and x[k4[3]]:
                    cost += 1
            for ik4 in indep_k4s:
                if not x[ik4[0]] and not x[ik4[1]] and not x[ik4[2]] and not x[ik4[3]]:
                    cost += 1
            return cost

        best_solution = None
        best_cost_global = float('inf')
        
        num_restarts = 100
        restart_time = time_budget / num_restarts
        
        for r in range(num_restarts):
            if time.time() > start_time + time_limit:
                break
                
            x = [random.randint(0, 1) for _ in range(n_curr)]
            current_cost = get_initial_cost(x)
            
            if current_cost == 0:
                return x
                
            best_x_local = x[:]
            best_cost_local = current_cost
            
            t_start = time.time()
            
            while time.time() - t_start < restart_time:
                if best_cost_local == 0:
                    return best_x_local
                    
                # Pick random variable to flip
                i = random.randint(0, n_curr - 1)
                old_val = x[i]
                x[i] = 1 - x[i]
                
                # Calculate delta cost
                delta_cost = 0
                
                # Check K4s involving i
                for idx in k4_inc[i]:
                    k4 = k4s[idx]
                    others = [k4[j] for j in range(4) if k4[j] != i]
                    all_others_1 = all(x[v] == 1 for v in others)
                    
                    if old_val == 0 and x[i] == 1:
                        if all_others_1:
                            delta_cost += 1
                    elif old_val == 1 and x[i] == 0:
                        if all_others_1:
                            delta_cost -= 1
                            
                # Check Indep K4s involving i
                for idx in ik4_inc[i]:
                    ik4 = indep_k4s[idx]
                    others = [ik4[j] for j in range(4) if ik4[j] != i]
                    all_others_0 = all(x[v] == 0 for v in others)
                    
                    if old_val == 0 and x[i] == 1:
                        if all_others_0:
                            delta_cost -= 1
                    elif old_val == 1 and x[i] == 0:
                        if all_others_0:
                            delta_cost += 1

                new_cost = current_cost + delta_cost
                
                # Accept if better, or with noise
                if new_cost <= current_cost or random.random() < 0.05:
                    current_cost = new_cost
                    if new_cost < best_cost_local:
                        best_cost_local = new_cost
                        best_x_local = x[:]
                else:
                    x[i] = old_val # Revert
                    
            if best_cost_local < best_cost_global:
                best_cost_global = best_cost_local
                best_solution = best_x_local
                
            if best_cost_global == 0:
                return best_solution
                
        return None

    # Try to grow the graph
    current_adj = adj[:]
    current_n = n_seed
    
    max_target = 43 # Try to reach the known best
    
    for new_v in range(n_seed, max_target):
        if time.time() - start_time > time_limit - 20:
            break
            
        # Get constraints from current graph
        k4s, indep_k4s = get_k4_and_independent_k4(current_adj, current_n)
        
        # Time budget for this step
        budget = 40.0
        
        sol = solve_new_vertex(k4s, indep_k4s, current_n, budget)
        
        if sol is not None:
            # Add new vertex
            new_n = current_n + 1
            current_adj.append(0)
            new_mask = 0
            for i in range(current_n):
                if sol[i]:
                    current_adj[i] |= (1 << new_n - 1)
                    new_mask |= (1 << i)
            current_adj[new_n - 1] = new_mask
            
            current_n = new_n
        else:
            break
            
    # Verify final graph
    def is_valid_graph(adj, n):
        for combo in itertools.combinations(range(n), 5):
            v0, v1, v2, v3, v4 = combo
            mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
            
            # Check Clique
            is_clique = True
            if (adj[v0] & (mask ^ (1 << v0))) != (mask ^ (1 << v0)): is_clique = False
            elif (adj[v1] & (mask ^ (1 << v1))) != (mask ^ (1 << v1)): is_clique = False
            elif (adj[v2] & (mask ^ (1 << v2))) != (mask ^ (1 << v2)): is_clique = False
            elif (adj[v3] & (mask ^ (1 << v3))) != (mask ^ (1 << v3)): is_clique = False
            elif (adj[v4] & (mask ^ (1 << v4))) != (mask ^ (1 << v4)): is_clique = False
            
            if is_clique:
                return False
                
            # Check Independent
            is_indep = True
            if (adj[v0] & (mask ^ (1 << v0))) != 0: is_indep = False
            elif (adj[v1] & (mask ^ (1 << v1))) != 0: is_indep = False
            elif (adj[v2] & (mask ^ (1 << v2))) != 0: is_indep = False
            elif (adj[v3] & (mask ^ (1 << v3))) != 0: is_indep = False
            elif (adj[v4] & (mask ^ (1 << v4))) != 0: is_indep = False
            
            if is_indep:
                return False
                
        return True

    if is_valid_graph(current_adj, current_n):
        final_edges = []
        for i in range(current_n):
            for j in range(i + 1, current_n):
                if current_adj[i] & (1 << j):
                    final_edges.append((i, j))
        return current_n, final_edges
    else:
        return n_seed, edges

if __name__ == "__main__":
    n, e = construct()
    print(f"Constructed graph with {n} vertices and {len(e)} edges")
