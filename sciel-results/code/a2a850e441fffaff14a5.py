"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    """
    Solve Max-Cut using Cerny tabu search with multiple restarts.
    Returns a 0/1 label array of length n.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int64)
    
    # Convert to numpy bool array for fast operations
    adj = np.asarray(adjacency, dtype=np.uint8)
    # Precompute degree vector
    degree = adj.sum(axis=1)
    
    # Time budget: 120s wall clock, use 110s to be safe
    start_time = time.time()
    time_budget = 110.0
    
    # Tabu list length
    tabu_len = 10
    # Number of iterations per run
    max_iter = 50000
    
    best_labels = None
    best_cut = -1
    
    def compute_cut(labels):
        """Compute cut size efficiently."""
        # For each vertex v, count neighbors in opposite partition
        # cut = sum_v sum_u adj[v,u] * (labels[v] != labels[u])
        # Using: cut = sum_v (degree[v] - sum_u adj[v,u] * (labels[v] == labels[u]))
        # Or: cut = 0.5 * sum_v sum_u adj[v,u] * (1 - (2*labels[v]-1)*(2*labels[u]-1))
        # Let's use the spin formulation: let s[v] = 2*labels[v] - 1 (i.e., ±1)
        # cut = 0.5 * n * sum_edges ... no, let's just do it directly
        # cut = sum_v (adj[v] @ (1 - labels)).sum() / 2? No.
        # For each v, edges from v to opposite side = adj[v] @ (1 - labels)
        # But this counts each edge twice (once from each endpoint)
        # So cut = 0.5 * sum_v (adj[v] @ (1 - labels))
        # Wait, that's not right either. Let me think again.
        # Edge (v,u) is in cut iff labels[v] != labels[u]
        # cut = sum_{v<u} adj[v,u] * (labels[v] != labels[u])
        # Using matrix: cut = 0.5 * (labels.T @ adj @ (1-labels) + (1-labels).T @ adj @ labels)
        # Simpler: cut = 0.5 * sum_v sum_u adj[v,u] * (labels[v] != labels[u])
        # = 0.5 * sum_v sum_u adj[v,u] * (1 - (1 - 2*labels[v])*(1 - 2*labels[u]))
        # Let's just compute it directly with numpy
        s = 2 * labels - 1  # ±1 spin
        # cut = 0.5 * (n * sum_degrees/2 ... no)
        # Actually: sum_v s[v] * sum_u adj[v,u] s[u] = sum_v sum_u adj[v,u] s[v] s[u]
        # = 2 * sum_{v<u} adj[v,u] s[v] s[u]
        # For edges in cut: s[v] != s[u], so s[v]s[u] = -1
        # For edges not in cut: s[v] = s[u], so s[v]s[u] = 1
        # sum_v sum_u adj[v,u] s[v] s[u] = (non-cut edges) * 1 + (cut edges) * (-1)
        # = (E - cut) - cut = E - 2*cut
        # So cut = (E - sum_v sum_u adj[v,u] s[v] s[u]) / 2
        # where E = total edges = degree.sum() / 2
        s_adj_s = s @ (adj @ s)  # sum_v sum_u adj[v,u] s[v] s[u]
        E = degree.sum() / 2
        cut = (E - s_adj_s) / 2
        return cut
    
    def tabu_search(labels, tabu_len, max_iter, deadline):
        """Run Cerny tabu search from initial labels."""
        labels = labels.copy()
        s = 2 * labels - 1  # ±1 spins
        E = degree.sum() / 2
        
        # Compute current cut
        s_adj_s = s @ (adj @ s)
        current_cut = (E - s_adj_s) / 2
        
        best_local_cut = current_cut
        best_local_labels = labels.copy()
        
        # Tabu list: store last flip time for each vertex
        last_flipped = np.full(n, -1, dtype=np.int64)
        tabu_counter = 0
        
        # Precompute row sums for fast delta calculation
        # When flipping vertex v:
        # Delta in cut = -2 * sum_u adj[v,u] s[v] s[u]
        # = -2 * s[v] * (adj[v] @ s)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find best move (including tabu consideration)
            # Delta for flipping vertex v:
            # new_s = s.copy(); new_s[v] = -new_s[v]
            # new_cut - old_cut = -2 * s[v] * (adj[v] @ s)
            # Let's compute delta for all vertices
            adj_s = adj @ s  # adj_s[v] = sum_u adj[v,u] s[u]
            delta = -2 * s * adj_s
            
            # Apply tabu: disallow flipping if last_flipped[v] >= tabu_counter - tabu_len
            # unless it's a "aspiration" move (improves best known)
            tabu_mask = last_flipped >= (tabu_counter - tabu_len)
            
            # Best non-tabu move, or best move if it improves global best
            best_v = -1
            best_delta = -np.inf
            
            # For efficiency, find best delta among non-tabu vertices
            # If all are tabu, pick the best tabu vertex
            non_tabu_delta = delta.copy()
            non_tabu_delta[tabu_mask] = -np.inf
            
            # Also consider: if a tabu move improves best_local_cut, allow it (aspiration)
            # For simplicity, let's just pick best non-tabu, and if none, best overall
            if np.any(~tabu_mask):
                best_v = np.argmax(non_tabu_delta)
                best_delta = non_tabu_delta[best_v]
            else:
                best_v = np.argmax(delta)
                best_delta = delta[best_v]
            
            if best_v < 0:
                break
            
            # Perform the flip
            v = best_v
            s[v] = -s[v]
            labels[v] = 1 - labels[v]
            
            # Update cut
            current_cut += best_delta
            
            # Update tabu list
            last_flipped[v] = tabu_counter
            tabu_counter += 1
            
            # Update best
            if current_cut > best_local_cut:
                best_local_cut = current_cut
                best_local_labels = labels.copy()
        
        return best_local_labels, best_local_cut
    
    # Run multiple restarts with different initializations
    num_restarts = 0
    deadline = start_time + time_budget
    
    # First, try a few deterministic initializations
    # 1. Alternating
    labels_init = np.arange(n) % 2
    best_labels, best_cut = tabu_search(labels_init, tabu_len, max_iter, deadline)
    num_restarts += 1
    
    # 2. Random initializations
    while time.time() < deadline - 5.0:  # Leave 5s margin
        labels_init = np.random.randint(0, 2, size=n)
        res_labels, res_cut = tabu_search(labels_init, tabu_len, max_iter, deadline)
        num_restarts += 1
        if res_cut > best_cut:
            best_cut = res_cut
            best_labels = res_labels
    
    return best_labels.astype(np.int64)

# EVOLVE-BLOCK-END
