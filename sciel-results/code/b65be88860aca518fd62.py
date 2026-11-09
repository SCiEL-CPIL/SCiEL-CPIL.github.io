import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    def get_paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        n = q
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return n, edges

    def solve_for_n(n, time_budget):
        if n < 5:
            return n, []
        
        # Precompute 5-sets and pair mappings
        # This is expensive, so we do it once per n
        five_sets = list(combinations(range(n), 5))
        num_sets = len(five_sets)
        
        # Map pair (u, v) to list of indices of 5-sets containing u, v
        # Use a dictionary of lists, then convert to numpy arrays
        pair_to_indices = {}
        for idx, s in enumerate(five_sets):
            # s is a tuple of 5 vertices
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = s[i], s[j]
                    if u > v: u, v = v, u
                    key = (u, v)
                    if key not in pair_to_indices:
                        pair_to_indices[key] = []
                    pair_to_indices[key].append(idx)
        
        # Convert to numpy arrays for speed
        pair_to_arr = {}
        for k, v in pair_to_indices.items():
            pair_to_arr[k] = np.array(v, dtype=np.int32)
            
        # Initialize State
        # adj: n x n matrix
        adj = np.zeros((n, n), dtype=np.int8)
        
        # If n >= 37, use Paley on first 37
        if n >= 37:
            q = 37
            residues = {(x * x) % q for x in range(1, q)}
            for i in range(37):
                for j in range(i + 1, 37):
                    if (j - i) % q in residues:
                        adj[i, j] = 1
                        adj[j, i] = 1
                        
        # Fill rest randomly
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0 and not (i < 37 and j < 37):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
                        
        # Initialize counts
        # counts[s] = number of edges in 5-set s
        counts = np.zeros(num_sets, dtype=np.int8)
        clique_count = 0
        ind_count = 0
        
        for idx, s in enumerate(five_sets):
            c = 0
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = s[i], s[j]
                    if adj[u, v]:
                        c += 1
            counts[idx] = c
            if c == 10:
                clique_count += 1
            elif c == 0:
                ind_count += 1
                
        current_cost = clique_count + ind_count
        best_cost = current_cost
        best_adj = adj.copy()
        
        # SA Parameters
        T = 10.0
        cooling = 0.999
        steps = 0
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            # Pick random pair
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            key = (u, v)
            if key not in pair_to_arr:
                continue # Should not happen for n>=5
            
            indices = pair_to_arr[key]
            c_old = counts[indices]
            
            # Determine move: add or remove edge
            if adj[u, v] == 1:
                # Remove edge
                c_new = c_old - 1
                # Update counts
                counts[indices] = c_new
                adj[u, v] = 0
                adj[v, u] = 0
                
                # Update costs
                was_clique = (c_old == 10)
                is_clique = (c_new == 10)
                delta_clique = np.sum(is_clique) - np.sum(was_clique)
                
                was_ind = (c_old == 0)
                is_ind = (c_new == 0)
                delta_ind = np.sum(is_ind) - np.sum(was_ind)
                
                delta_cost = delta_clique + delta_ind
            else:
                # Add edge
                c_new = c_old + 1
                counts[indices] = c_new
                adj[u, v] = 1
                adj[v, u] = 1
                
                was_clique = (c_old == 10)
                is_clique = (c_new == 10)
                delta_clique = np.sum(is_clique) - np.sum(was_clique)
                
                was_ind = (c_old == 0)
                is_ind = (c_new == 0)
                delta_ind = np.sum(is_ind) - np.sum(was_ind)
                
                delta_cost = delta_clique + delta_ind
                
            new_cost = current_cost + delta_cost
            
            # Accept/Reject
            if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj.copy()
                    
                if best_cost == 0:
                    # Found valid graph
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j] == 1:
                                edges.append((i, j))
                    return n, edges
            else:
                # Revert
                if adj[u, v] == 1:
                    # We removed it, so put it back
                    adj[u, v] = 1
                    adj[v, u] = 1
                    counts[indices] = c_old
                else:
                    # We added it, so remove it
                    adj[u, v] = 0
                    adj[v, u] = 0
                    counts[indices] = c_old
                    
            T *= cooling
            steps += 1
            
        # If we didn't find 0, return None to try smaller n
        return None

    # Try 42 first
    res = solve_for_n(42, 100)
    if res:
        return res
        
    # Try 41
    res = solve_for_n(41, 50)
    if res:
        return res
        
    # Try 40
    res = solve_for_n(40, 30)
    if res:
        return res
        
    # Try 39
    res = solve_for_n(39, 20)
    if res:
        return res
        
    # Try 38
    res = solve_for_n(38, 10)
    if res:
        return res
        
    # Fallback to 37
    return get_paley_37()

# EVOLVE-BLOCK-END
