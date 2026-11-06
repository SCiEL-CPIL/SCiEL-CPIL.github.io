"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Maximum Cut via Local Search with Random Restarts.
    
    Approach:
    1. Start with a random partition.
    2. Iteratively find the vertex whose flip increases the cut size the most.
    3. Flip that vertex and update the state.
    4. Repeat until no flip improves the cut (local optimum).
    5. Restart with a new random partition multiple times to escape local optima.
    6. Return the best partition found.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.float64)
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0)
    
    total_edges = 0.5 * np.sum(adj)
    
    best_cut = -1
    best_labels = np.zeros(n, dtype=np.int8)
    
    # Time limit management
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    # Number of restarts
    # For n=100, local search is very fast. We can do many restarts.
    max_restarts = 1000
    
    for r in range(max_restarts):
        if time.time() - start_time > max_time:
            break
            
        # 1. Random Initialization
        labels = np.random.randint(0, 2, size=n).astype(np.int8)
        s = 2 * labels - 1 # Signs: -1 or 1
        
        # 2. Compute initial cut and neighbor sums
        # As[i] = sum_j adj[i, j] * s[j]
        As = adj @ s
        # Cut size = 0.5 * sum_{i,j} adj[i,j] (1 - s[i]s[j]) = total_edges - 0.5 * s^T A s
        # s^T A s = sum_i s[i] As[i]
        current_cut = total_edges - 0.5 * np.dot(s, As)
        
        # 3. Local Search (Best Improvement)
        improved = True
        while improved:
            improved = False
            
            # Calculate delta for all vertices
            # delta[v] = 2 * s[v] * As[v]
            # If we flip v, new s[v] = -s[v].
            # Change in s^T A s:
            # Old term involving v: s[v] * As[v]
            # New term: -s[v] * (As[v] - 2 * adj[v, v] * s[v] + ... wait, simpler derivation)
            # Let's use the standard formula:
            # Delta in cut = 2 * s[v] * As[v]
            # Proof:
            # Cut = 0.5 * sum_{i,j} A_{ij} (1 - s_i s_j)
            # Flip v: s_v -> -s_v
            # Terms involving v: (v, u) and (u, v).
            # Old: A_{vu}(1 - s_v s_u) + A_{uv}(1 - s_u s_v) = 2 A_{vu}(1 - s_v s_u)
            # New: A_{vu}(1 - (-s_v) s_u) + A_{uv}(1 - s_u (-s_v)) = 2 A_{vu}(1 + s_v s_u)
            # Diff = 2 A_{vu}(1 + s_v s_u) - 2 A_{vu}(1 - s_v s_u) = 4 A_{vu} s_v s_u
            # Sum over u: 4 s_v sum_u A_{vu} s_u = 4 s_v As[v]
            # Cut change = 0.5 * (Sum of diffs) ... wait.
            # The sum over i,j includes both (v,u) and (u,v).
            # My previous derivation in the seed code said:
            # Total sum change (double sum) = 4 s_v (A @ s)_v
            # Cut size change = 0.5 * (Double Sum Change) = 2 s_v (A @ s)_v
            # Yes, delta = 2 * s[v] * As[v].
            
            deltas = 2 * s * As
            
            # Find the best improvement
            max_delta = np.max(deltas)
            
            if max_delta <= 1e-9:
                # No improvement possible
                break
            
            # Find the vertex with max delta
            # If there are ties, pick one randomly or the first one.
            # Using argmax picks the first one.
            v = np.argmax(deltas)
            
            # Apply flip
            old_s_v = s[v]
            s[v] = -old_s_v
            labels[v] = 1 - labels[v]
            
            # Update As
            # As[u] = sum_k A[u, k] s[k]
            # Only term k=v changes.
            # Change in As[u] = A[u, v] * (new_s_v - old_s_v) = A[u, v] * (-2 * old_s_v)
            As -= 2 * old_s_v * adj[v]
            
            # Update cut
            current_cut += max_delta
            improved = True
            
            # Check time inside loop? It's a tight loop, but n=100 is small.
            # The while loop runs at most O(E) times usually, but can be more.
            # For safety, we can check time every few iterations, but likely not needed for n=100.
            
        # Update global best
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
