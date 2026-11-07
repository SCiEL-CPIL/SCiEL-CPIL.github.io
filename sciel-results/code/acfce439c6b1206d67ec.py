"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    n = P['n']
    base_edges_set = set()
    for u, v in P['edges']:
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # Candidate edges: must connect opposite parity to maintain bipartiteness (triangle-free)
    # and must not be in base.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 == 1 and (u, v) not in base_edges_set:
                candidates.append((u, v))
                
    def get_all_distances(adj):
        dist = [[-1] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[i][v] == -1:
                        dist[i][v] = dist[i][u] + 1
                        q.append(v)
        return dist

    def get_diameter(adj):
        max_d = 0
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                        if dist[v] > max_d:
                            max_d = dist[v]
        return max_d

    # Initial graph: just the path
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # 1. Find a valid solution using Greedy
    current_edges = set()
    current_adj = [set(s) for s in adj]
    current_diam = get_diameter(current_adj)
    
    while current_diam > 4:
        if time.time() > deadline:
            break
        best_edge = None
        best_diam = current_diam + 1
        
        for u, v in candidates:
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
            current_adj[u].add(v)
            current_adj[v].add(u)
            new_diam = get_diameter(current_adj)
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
            if new_diam <= 4:
                break
                
        if best_edge is None:
            break
            
        u, v = best_edge
        current_adj[u].add(v)
        current_adj[v].add(u)
        current_edges.add((u, v))
        current_edges.add((v, u))
        current_diam = best_diam
        
    best_solution = set(current_edges)
    best_count = len([e for e in best_solution if e[0] < e[1]])
    
    # 2. Simulated Annealing to minimize edge count
    # State is a set of edges. We want to minimize size, subject to diam <= 4.
    # We start from the greedy solution.
    
    def is_valid(edge_set):
        test_adj = [set(s) for s in adj]
        for u, v in edge_set:
            if u < v:
                test_adj[u].add(v)
                test_adj[v].add(u)
            else:
                test_adj[v].add(u)
                test_adj[u].add(v)
        return get_diameter(test_adj) <= 4

    # Normalize best_solution to canonical form (u < v)
    best_solution = set((min(u,v), max(u,v)) for u,v in best_solution)
    
    current_set = set(best_solution)
    current_diam = get_diameter([set(s) for s in adj] if not current_set else None) # Placeholder
    # Recalculate current diam properly
    temp_adj = [set(s) for s in adj]
    for u,v in current_set:
        temp_adj[u].add(v)
        temp_adj[v].add(u)
    current_diam = get_diameter(temp_adj)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Pre-compute candidate list for random selection
    cand_list = list(candidates)
    
    while time.time() < deadline and T > T_min:
        # Generate neighbor
        move_type = random.random()
        new_set = set(current_set)
        
        if move_type < 0.4:
            # Remove an edge
            if not current_set: continue
            edge_to_remove = random.choice(list(current_set))
            new_set.remove(edge_to_remove)
        elif move_type < 0.8:
            # Add an edge
            edge_to_add = random.choice(cand_list)
            if edge_to_add not in new_set:
                new_set.add(edge_to_add)
        else:
            # Swap
            if not current_set: continue
            edge_to_remove = random.choice(list(current_set))
            edge_to_add = random.choice(cand_list)
            if edge_to_add not in new_set:
                new_set.remove(edge_to_remove)
                new_set.add(edge_to_add)
        
        # Check validity of new_set
        test_adj = [set(s) for s in adj]
        for u, v in new_set:
            test_adj[u].add(v)
            test_adj[v].add(u)
        
        new_diam = get_diameter(test_adj)
        
        if new_diam <= 4:
            new_cost = len(new_set)
            old_cost = len(current_set)
            delta_cost = new_cost - old_cost
            
            if delta_cost <= 0:
                # Accept if better or equal
                current_set = new_set
                current_diam = new_diam
                if len(current_set) < best_count:
                    best_solution = set(current_set)
                    best_count = len(current_set)
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < pow(sys.e, -delta_cost / T):
                    current_set = new_set
                    current_diam = new_diam
        else:
            # Invalid state, reject (or penalize heavily, but simpler to reject)
            pass
            
        T *= cooling
        
    # Convert best_solution to list of lists
    final_added = [[int(u), int(v)] for u, v in best_solution]
    return final_added

# EVOLVE-BLOCK-END
