"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Cerny tabu search for maximum cut.
    adjacency: bool matrix (n x n)
    returns: 0/1 label array of length n
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array of 0/1 for fast arithmetic
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 150s wall-clock limit, use 140s to be safe
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Tabu list length
    tabu_len = max(5, n // 10)
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different random initializations
    rng = np.random.RandomState(42)
    
    num_restarts = 50  # We'll check time and break early
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        labels = rng.randint(0, 2, size=n).astype(np.int8)
        
        # Compute initial cut size
        # cut = sum over v of sum over u where labels[u] != labels[v] of adj[v,u] / 2
        # More efficiently: for each v, count neighbors in opposite partition
        # Use matrix multiplication: labels @ adj gives for each v, sum of adj[v,u]*labels[u]
        # degree[v] = sum of adj[v,u]
        # same_side_sum[v] = sum of adj[v,u]*labels[u] where labels[u] == labels[v]
        # Actually let's compute cut directly:
        # cut = 0.5 * sum_{v,u} adj[v,u] * (labels[v] != labels[u])
        # = 0.5 * (sum_{v,u} adj[v,u] - sum_{v,u} adj[v,u] * (labels[v] == labels[u]))
        # For 0/1 labels: (labels[v] != labels[u]) = labels[v] + labels[u] - 2*labels[v]*labels[u]
        # So cut = 0.5 * sum_{v,u} adj[v,u] * (labels[v] + labels[u] - 2*labels[v]*labels[u])
        # = 0.5 * (sum_v labels[v] * sum_u adj[v,u] + sum_u labels[u] * sum_v adj[v,u] - 2 * sum_{v,u} adj[v,u]*labels[v]*labels[u])
        # = sum_v labels[v] * degree[v] - labels^T @ adj @ labels
        
        def compute_cut(labels):
            # labels is 0/1 int array
            deg = adj.sum(axis=1)  # degree of each vertex
            # labels^T @ adj @ labels
            al = adj @ labels
            ll = labels @ al
            return int(labels @ deg - ll)
        
        current_cut = compute_cut(labels)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
        
        # Tabu search
        # For each vertex v, the gain of flipping v is:
        # If v is in partition 0, flipping to 1:
        #   gain = (num neighbors in 1) - (num neighbors in 0)
        #   = (deg[v] - same_side_count) - same_side_count = deg[v] - 2*same_side_count
        # More precisely:
        #   Let S0 = {u : labels[u] == 0}, S1 = {u : labels[u] == 1}
        #   If labels[v] == 0:
        #     neighbors in S0 (excluding v itself, but adj[v,v]=0 typically): n0 = sum_{u in S0} adj[v,u]
        #     neighbors in S1: n1 = sum_{u in S1} adj[v,u]
        #     Flipping v from 0 to 1: edges from v to S0 become crossing, edges from v to S1 stop being crossing
        #     gain = n0 - n1
        #   If labels[v] == 1:
        #     gain = n1 - n0 (where n0 = neighbors in S0, n1 = neighbors in S1)
        # 
        # In general: gain[v] = (deg[v] - 2 * same_side_neighbors[v]) if we define same_side_neighbors appropriately
        # Actually: gain[v] = 2 * (num_neighbors_in_opposite) - deg[v]
        # Wait let me re-derive:
        # Current contribution of v to cut: number of edges from v to opposite side
        # After flip: number of edges from v to same side (which was same side, now opposite)
        # So gain = (edges to same side) - (edges to opposite side)
        # = same_side_count - opposite_side_count
        # = same_side_count - (deg[v] - same_side_count)
        # = 2 * same_side_count - deg[v]
        
        # Let's precompute for efficiency
        # same_side_count[v] = sum_{u : labels[u] == labels[v]} adj[v,u]
        # We can compute: for each v, adj[v] @ labels gives sum of adj[v,u]*labels[u]
        # If labels[v] == 1: same_side_count = adj[v] @ labels (since labels[u]=1 for same side)
        # If labels[v] == 0: same_side_count = adj[v] @ (1 - labels) = deg[v] - adj[v] @ labels
        
        def compute_gains(labels):
            # Returns array where gains[v] = change in cut if we flip vertex v
            al = adj @ labels  # al[v] = sum_u adj[v,u] * labels[u]
            deg = adj.sum(axis=1)
            # If labels[v] == 1: same = al[v], opp = deg[v] - al[v], gain = same - opp = 2*al[v] - deg[v]
            # If labels[v] == 0: same = deg[v] - al[v], opp = al[v], gain = same - opp = 2*(deg[v]-al[v]) - al[v] = 2*deg[v] - 3*al[v]
            # Wait, let me redo:
            # For v with labels[v]=1:
            #   neighbors with label 1: al[v]
            #   neighbors with label 0: deg[v] - al[v]
            #   gain = (neighbors with label 1) - (neighbors with label 0) = al[v] - (deg[v] - al[v]) = 2*al[v] - deg[v]
            # For v with labels[v]=0:
            #   neighbors with label 0: deg[v] - al[v]
            #   neighbors with label 1: al[v]
            #   gain = (neighbors with label 0) - (neighbors with label 1) = (deg[v] - al[v]) - al[v] = deg[v] - 2*al[v]
            
            gains = np.empty(n, dtype=np.int32)
            mask1 = (labels == 1)
            mask0 = (labels == 0)
            gains[mask1] = 2 * al[mask1] - deg[mask1]
            gains[mask0] = deg[mask0] - 2 * al[mask0]
            return gains
        
        gains = compute_gains(labels)
        
        # Tabu list: store (vertex, iteration) for last tabu_len flips
        tabu_list = {}  # vertex -> iteration when it was last flipped
        current_iter = 0
        
        # Run tabu search for a number of iterations
        max_iters = 2000
        no_improve_count = 0
        best_in_run = current_cut
        best_labels_in_run = labels.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            current_iter += 1
            
            # Find best non-tabu move
            best_gain = -np.inf
            best_v = -1
            
            for v in range(n):
                if v in tabu_list and (current_iter - tabu_list[v]) < tabu_len:
                    continue
                if gains[v] > best_gain:
                    best_gain = gains[v]
                    best_v = v
            
            if best_v == -1:
                # All moves are tabu, pick best move overall
                best_v = int(np.argmax(gains))
                best_gain = gains[best_v]
            
            # Apply the flip
            labels[best_v] = 1 - labels[best_v]
            tabu_list[best_v] = current_iter
            
            # Update gains efficiently
            # Flipping best_v changes gains for best_v and all its neighbors
            # For best_v: its gain changes sign (roughly)
            # For neighbor u of best_v:
            #   If u and best_v were on same side, now they're on opposite sides
            #   The gain for u changes by: 
            #     If u was on same side as best_v before flip:
            #       Before: u's same_side included best_v, now it doesn't
            #       If labels[u] == old_labels[best_v]: same_side_count decreases by 1 (if adj[u,best_v]=1)
            #       This changes gain by -2 (since gain = 2*same - deg or deg - 2*same)
            #     Let me think more carefully...
            
            # Actually, let's just recompute gains for simplicity since n=100
            gains = compute_gains(labels)
            current_cut += best_gain
            
            if current_cut > best_in_run:
                best_in_run = current_cut
                best_labels_in_run = labels.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Diversification: if stuck for too long, randomize
            if no_improve_count > 100:
                # Random restart within this run
                labels = rng.randint(0, 2, size=n).astype(np.int8)
                current_cut = compute_cut(labels)
                gains = compute_gains(labels)
                no_improve_count = 0
                best_in_run = current_cut
                best_labels_in_run = labels.copy()
                tabu_list = {}
        
        if best_in_run > best_cut:
            best_cut = best_in_run
            best_labels = best_labels_in_run.copy()
    
    # Final local search polish on best solution
    if best_labels is not None and time.time() < deadline:
        labels = best_labels.copy()
        current_cut = compute_cut(labels)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            gains = compute_gains(labels)
            for v in range(n):
                if gains[v] > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += gains[v]
                    improved = True
                    gains = compute_gains(labels)
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
