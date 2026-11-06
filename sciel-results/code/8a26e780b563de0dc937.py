import numpy as np
import time
import random

def construct():
    n = 26
    # Edges are (i, j) with i < j
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Map edge to index
    edge_idx = {}
    for k, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = k
    
    def get_edge_idx(i, j):
        if i > j:
            i, j = j, i
        return edge_idx[(i, j)]

    # Initialize colors randomly
    colors = np.random.randint(0, 5, size=num_edges)
    
    # Precompute edge endpoints as arrays for fast access
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    def calculate_cost(colors):
        # colors is a numpy array of shape (num_edges,)
        total_cost = 0
        for c in range(5):
            # Create adjacency matrix for color c
            A = np.zeros((n, n), dtype=np.int8)
            mask = (colors == c)
            # Set A[u,v] and A[v,u] to 1 for edges in this color
            A[u_arr[mask], v_arr[mask]] = 1
            A[v_arr[mask], u_arr[mask]] = 1
            
            # Compute A^2
            A2 = A @ A
            
            # Count C4s: for each u < v, count common neighbors
            # (A2)_{u,v} is the number of common neighbors
            # We need sum over u < v of comb(A2[u,v], 2)
            # Extract upper triangle
            triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            common_neighbors = A2[triu_mask]
            
            # Sum of comb(k, 2) = k*(k-1)//2
            # k is integer
            cost_c = np.sum(common_neighbors * (common_neighbors - 1) // 2)
            total_cost += cost_c
        return int(total_cost)

    current_cost = calculate_cost(colors)
    
    # Simulated Annealing
    start_time = time.time()
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while current_cost > 0 and time.time() - start_time < 110:
        # Pick a random edge
        k = random.randint(0, num_edges - 1)
        old_color = colors[k]
        
        # Pick a random new color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        colors[k] = new_color
        new_cost = calculate_cost(colors)
        
        delta = new_cost - current_cost
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_cost = new_cost
        else:
            # Revert
            colors[k] = old_color
            
        T *= cooling_rate
        if T < T_min:
            T = T_min

    if current_cost == 0:
        return {'n': n, 'colors': colors.tolist()}
    
    # Fallback
    n_seed = 11
    return {'n': n_seed, 'colors': [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]}
