import time
import math
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 256
    nodes = list(range(1, N + 1))
    
    # 1. Identify forbidden nodes (diagonal constraint a+b square with a=b)
    # 2a is a perfect square
    forbidden = set()
    for a in nodes:
        s = 2 * a
        r = int(math.isqrt(s))
        if r * r == s:
            forbidden.add(a)
            
    valid_nodes = [a for a in nodes if a not in forbidden]
    
    if not valid_nodes:
        return []
        
    # 2. Build graph
    # Map valid nodes to indices 0..M-1 for efficiency
    node_map = {v: i for i, v in enumerate(valid_nodes)}
    M = len(valid_nodes)
    
    # Adjacency list
    adj = [set() for _ in range(M)]
    
    # Squares up to 2*N
    squares = [i*i for i in range(1, int(math.isqrt(2*N)) + 2)]
    square_set = set(squares)
    
    for i in range(M):
        a = valid_nodes[i]
        for j in range(i + 1, M):
            b = valid_nodes[j]
            if a + b in square_set:
                adj[i].add(j)
                adj[j].add(i)
                
    # Initial solution via Greedy
    # Simple greedy: pick node, remove neighbors
    def greedy_mis(nodes_set):
        nodes_list = list(nodes_set)
        # Sort by degree descending
        nodes_list.sort(key=lambda x: len(adj[x]), reverse=True)
        chosen = set()
        available = set(nodes_set)
        for v in nodes_list:
            if v in available:
                chosen.add(v)
                available.remove(v)
                for u in adj[v]:
                    if u in available:
                        available.remove(u)
        return chosen

    current_best_set = greedy_mis(set(range(M)))
    current_best_size = len(current_best_set)
    
    # Branch and Bound
    best_size = current_best_size
    best_set = current_best_set
    
    def get_bound(remaining_nodes):
        if not remaining_nodes:
            return 0
        # LP Relaxation
        # Maximize sum(x_i)
        # Subject to x_i + x_j <= 1 for edges
        # 0 <= x_i <= 1
        
        c = np.ones(len(remaining_nodes)) * -1 # Minimize -sum
        
        # Constraints
        # We need to map remaining nodes to local indices
        rem_list = list(remaining_nodes)
        rem_idx = {v: i for i, v in enumerate(rem_list)}
        
        A_ub = []
        b_ub = []
        
        for v in rem_list:
            for u in adj[v]:
                if u in rem_idx:
                    # Ensure we don't add duplicate constraints (i,j) and (j,i)
                    # Since we iterate all v, we will see (v,u) and (u,v).
                    # We can just add them all, or optimize.
                    # For simplicity, add all.
                    row = np.zeros(len(rem_list))
                    row[rem_idx[v]] = 1
                    row[rem_idx[u]] = 1
                    A_ub.append(row)
                    b_ub.append(1)
                    
        if not A_ub:
            return len(rem_list)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * len(rem_list), method='highs')
            if res.status == 0:
                return -res.fun
            else:
                return len(rem_list) # Fallback
        except Exception:
            return len(rem_list)
            
    def branch_and_bound(remaining_nodes, current_size, current_set):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        
        if not remaining_nodes:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
            
        # Bound
        bound = get_bound(remaining_nodes)
        if current_size + bound <= best_size:
            return
            
        # Select branch node
        # Heuristic: highest degree in remaining subgraph
        max_deg = -1
        branch_node = -1
        for v in remaining_nodes:
            deg = len([u for u in adj[v] if u in remaining_nodes])
            if deg > max_deg:
                max_deg = deg
                branch_node = v
                
        if branch_node == -1:
             # No edges, take all
             new_size = current_size + len(remaining_nodes)
             new_set = set(current_set)
             new_set.update(remaining_nodes)
             if new_size > best_size:
                 best_size = new_size
                 best_set = new_set
             return

        # Branch 1: Include branch_node
        # Remove branch_node and its neighbors from remaining
        neighbors = set(adj[branch_node]) & remaining_nodes
        new_remaining = remaining_nodes - {branch_node} - neighbors
        new_set = set(current_set)
        new_set.add(branch_node)
        branch_and_bound(new_remaining, current_size + 1, new_set)
        
        if time.time() > deadline:
            return

        # Branch 2: Exclude branch_node
        new_remaining = remaining_nodes - {branch_node}
        branch_and_bound(new_remaining, current_size, current_set)
        
    # Start B&B
    # To make it faster, we can order the initial set
    branch_and_bound(set(range(M)), 0, set())
    
    # Map back to original values
    result = [valid_nodes[i] for i in sorted(best_set)]
    
    # Verify
    res_set = set(result)
    for a in result:
        for b in result:
            if a <= b:
                s = a + b
                r = int(math.isqrt(s))
                if r * r == s:
                    # Invalid
                    return list(range(1, 257, 3))
                    
    return result
