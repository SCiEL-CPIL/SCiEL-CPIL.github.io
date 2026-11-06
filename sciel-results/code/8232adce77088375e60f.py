"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    # Convert to float for linear algebra operations
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    # Safety deadline: 150s limit, use 140s to be safe
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_labels = None
    best_cut = -1
    
    # Number of restarts based on available time
    # Each restart involves:
    # 1. Spectral init (eigenvector)
    # 2. Local search (multiple passes)
    
    # Estimate cost per restart:
    # - eigsh/eigh on 100x100 is fast (~1ms)
    # - Local search pass: n iterations, each O(n) if not vectorized, or O(n^2) if vectorized
    # We can vectorize the local search step significantly.
    
    # Strategy:
    # 1. Generate spectral initialization
    # 2. Run local search until no improvement or time limit
    # 3. Repeat with random perturbations if time permits
    
    # Pre-compute degree vector for faster gain calculation
    # Gain for flipping vertex v:
    # If v is in 0: gain = 2 * (edges to 1) - (total degree)
    # If v is in 1: gain = 2 * (edges to 0) - (total degree)
    
    for restart in range(50):  # Max 50 restarts, but we'll break on time
        if time.time() > deadline:
            break
            
        # --- Initialization ---
        # Spectral initialization: use the leading eigenvector of A
        # For a graph, the leading eigenvector of the adjacency matrix 
        # often correlates with the optimal cut structure.
        try:
            # Use numpy's eigh for symmetric matrices (faster and more stable than eig)
            # We only need the largest eigenvalue's eigenvector
            # But eigh returns all. For 100x100, this is very fast.
            vals, vecs = np.linalg.eigh(A)
            # The largest eigenvalue is the last one in ascending order
            leading_vec = vecs[:, -1]
            
            # Create labels based on sign of the leading eigenvector
            # To handle the zero-crossing, we can use the median or simply sign
            # Using sign: positive -> 1, negative -> 0
            # If there are zeros, assign randomly or by a tie-breaker
            labels = (leading_vec > 0).astype(np.int8)
            
            # Ensure both partitions are non-empty (usually true for connected graphs)
            # If all 0s or all 1s, flip a random vertex
            if labels.sum() == 0 or labels.sum() == n:
                labels[np.random.randint(n)] = 1 - labels[np.random.randint(n)]
                
        except Exception:
            # Fallback to random initialization if spectral fails
            labels = np.random.randint(0, 2, n).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels[np.random.randint(n)] = 1 - labels[np.random.randint(n)]

        # --- Local Search ---
        # We perform 1-flip local search.
        # To speed up, we can vectorize the gain calculation for all vertices at once.
        # However, flipping one vertex changes the gains of its neighbors.
        # A common optimization is to maintain the gain array and update it incrementally.
        
        # Let's implement a vectorized pass where we calculate the gain for all vertices
        # given the current labels, then flip the vertex with the maximum positive gain.
        # Repeat until no positive gain exists.
        
        # Precompute:
        # Let S0 be the set of vertices with label 0, S1 with label 1.
        # For vertex v in S0:
        #   cut_contribution = sum(A[v, S1])
        #   if we flip v to 1:
        #     new_cut_contribution = sum(A[v, S0]) = degree[v] - sum(A[v, S1])
        #     delta = (degree[v] - sum(A[v, S1])) - sum(A[v, S1]) = degree[v] - 2*sum(A[v, S1])
        # For vertex v in S1:
        #   cut_contribution = sum(A[v, S0])
        #   if we flip v to 0:
        #     new_cut_contribution = sum(A[v, S1]) = degree[v] - sum(A[v, S0])
        #     delta = (degree[v] - sum(A[v, S0])) - sum(A[v, S0]) = degree[v] - 2*sum(A[v, S0])
        
        # We can maintain:
        # cross_sum[v] = sum of A[v, u] for u in the opposite partition to v.
        # Initially:
        #   For v in S0: cross_sum[v] = A[v, S1].sum()
        #   For v in S1: cross_sum[v] = A[v, S0].sum()
        
        # Let's compute initial cross_sum vectorized.
        # labels: 0/1 array
        # opposite = 1 - labels
        # cross_sum = A @ opposite  (This gives sum of A[v,u] for u where opposite[u]=1, i.e., u in opposite partition)
        # Wait, if v is 0, opposite is 1. So cross_sum[v] = sum(A[v, u] for u in S1). Correct.
        # If v is 1, opposite is 0. So cross_sum[v] = sum(A[v, u] for u in S0). Correct.
        
        # Degree vector
        degree = A.sum(axis=1)
        
        # Current cut size
        # Cut size = sum over v in S0 of A[v, S1] = 0.5 * sum(cross_sum)
        # Because each edge is counted twice (once for each endpoint).
        # Actually, let's just calculate it directly for the best cut tracking.
        
        # Local Search Loop
        improved = True
        while improved:
            if time.time() > deadline:
                break
                
            # Calculate gain for flipping each vertex
            # gain[v] = degree[v] - 2 * cross_sum[v]
            gains = degree - 2 * (A @ (1 - labels))
            
            # Find the vertex with the maximum gain
            max_gain = gains.max()
            
            if max_gain <= 0:
                improved = False
                break
            
            # Flip the vertex with max gain
            # In case of ties, pick the first one
            v = np.argmax(gains)
            
            # Update labels
            labels[v] = 1 - labels[v]
            
            # Update cross_sum for neighbors of v
            # When v flips, its "opposite" partition changes.
            # For any neighbor u of v:
            #   If u was in the same partition as v, now u is in the opposite partition.
            #   If u was in the opposite partition as v, now u is in the same partition.
            
            # Let's re-calculate cross_sum efficiently?
            # Or just update the gains?
            # Actually, re-calculating `A @ (1 - labels)` is O(n^2).
            # For n=100, O(10000) is very fast.
            # Number of flips in local search is usually small (tens to hundreds).
            # So O(n^2 * flips) is acceptable.
            
            # Let's just re-calculate cross_sum in the next iteration of the loop.
            # This is simpler and less error-prone.
            # The loop condition `if time.time() > deadline` ensures we don't run away.
            
            # To be slightly more efficient, we can track the cut size.
            # But for now, simplicity is key.
            
        # End of local search for this restart
        
        # Calculate final cut size for this restart
        # Cut size = sum of A[i,j] where labels[i] != labels[j]
        # Vectorized:
        # diff = labels[:, None] != labels[None, :]
        # cut = (A * diff).sum() / 2  (since symmetric)
        
        # More efficient:
        # cross_sum_final = A @ (1 - labels)
        # cut_size = cross_sum_final.sum() / 2
        
        cross_sum_final = A @ (1 - labels)
        current_cut = cross_sum_final.sum() / 2
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
        # If we found a perfect cut (all edges crossing? No, max cut is at most E), 
        # or if no improvement was possible in the first pass, we might stop early?
        # No, different restarts might yield different local optima.
        
        # Check time again
        if time.time() > deadline - 5.0:
            # Reserve 5 seconds for the final return and any cleanup
            break

    if best_labels is None:
        # Fallback: return alternating pattern
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
