import numpy as np
import itertools
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

# Precomputation
edges = []
edge_map = {}
idx = 0
for u in range(N):
    for v in range(u + 1, N):
        edges.append((u, v))
        edge_map[(u, v)] = idx
        idx += 1

# Precompute subsets
subset_data = []
for combo in itertools.combinations(range(N), 8):
    subset_data.append(combo)

# Precompute local bipartitions for 8 vertices (indices 0-7)
local_bipartitions = []
for a_mask in range(1, 256):
    if bin(a_mask).count('1') == 4:
        b_mask = 255 ^ a_mask
        if a_mask < b_mask:
            a_idx = [i for i in range(8) if (a_mask >> i) & 1]
            b_idx = [i for i in range(8) if (b_mask >> i) & 1]
            local_bipartitions.append((a_idx, b_idx))

def contains_q3(adj_matrix):
    """
    adj_matrix: 16x16 numpy array (int8).
    Returns True if Q3 is found.
    """
    for combo in subset_data:
        sub_mat = adj_matrix[np.ix_(combo, combo)]
        # Count edges
        if np.sum(sub_mat) // 2 < 12:
            continue
        
        # Check bipartitions
        for a_idx, b_idx in local_bipartitions:
            degs = np.sum(sub_mat, axis=1)
            if np.min(degs[a_idx]) >= 3 and np.min(degs[b_idx]) >= 3:
                return True
    return False

def construct():
    start_time = time.time()
    timeout = 115
    
    best_adj = np.zeros((N, N), dtype=np.int8)
    best_count = 0
    
    # Helper to update best
    def update_best(adj, count):
        nonlocal best_adj, best_count
        if count > best_count:
            best_adj = adj.copy()
            best_count = count

    # Local Search
    def local_search(start_adj):
        current_adj = start_adj.copy()
        current_count = np.sum(current_adj) // 2
        
        # Ensure valid
        while contains_q3(current_adj):
            # Remove a random edge
            # Find a Q3? Or just remove random edge until valid.
            # To be efficient, we should find the Q3 and remove an edge from it.
            # But finding the Q3 is as expensive as checking.
            # Let's just remove random edges.
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u != v and current_adj[u, v] == 1:
                current_adj[u, v] = 0
                current_adj[v, u] = 0
                current_count -= 1
                
        update_best(current_adj, current_count)
        
        no_improve = 0
        while time.time() - start_time < timeout:
            # Try to add a random edge
            if random.random() < 0.8:
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v or current_adj[u, v] == 1:
                    continue
                    
                new_adj = current_adj.copy()
                new_adj[u, v] = 1
                new_adj[v, u] = 1
                new_count = current_count + 1
                
                if not contains_q3(new_adj):
                    current_adj = new_adj
                    current_count = new_count
                    no_improve = 0
                    update_best(current_adj, current_count)
                else:
                    no_improve += 1
            else:
                # Remove a random edge to escape local optimum
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u != v and current_adj[u, v] == 1:
                    current_adj[u, v] = 0
                    current_adj[v, u] = 0
                    current_count -= 1
                    no_improve += 1
            
            if no_improve > 500:
                break
                
        return current_adj, current_count

    # Run multiple restarts
    for i in range(20):
        if time.time() - start_time > timeout:
            break
            
        # Start with random graph
        start_adj = np.zeros((N, N), dtype=np.int8)
        for u in range(N):
            for v in range(u + 1, N):
                if random.random() < 0.5:
                    start_adj[u, v] = 1
                    start_adj[v, u] = 1
                    
        local_search(start_adj)

    # Convert best_adj to list of edges
    result = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u, v] == 1:
                result.append([u, v])
                
    return result
