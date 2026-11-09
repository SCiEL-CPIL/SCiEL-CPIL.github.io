"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast operations
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Deadline guard: 110s limit to be safe within 120s
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Helper: Calculate cut size
    def cut_size(labels):
        # Cut size = sum of edges where labels differ
        # Efficient way: sum(adj * (labels[:, None] != labels[None, :])) / 2
        # But we can just count 1s in adj where labels differ
        # Using matrix multiplication trick:
        # Let L be vector of labels. Cut = 0.5 * (sum(L) - L^T A L) ... no that's for Ising.
        # For Max Cut: Cut = 0.5 * (||L||^2 - L^T A L) if L is +/-1?
        # If L is 0/1:
        # Let s = 2*L - 1 (so -1, 1)
        # Cut = 0.5 * (n^2 - s^T A s) / 2? 
        # Actually, standard formula: Cut = 1/4 * (sum(s_i) - s^T A s) is for +/-1?
        # Let's stick to the simple vectorized counting which is fast enough for n=100.
        # diff = (labels[:, None] != labels[None, :])
        # return np.sum(adj * diff)
        
        # Faster:
        # Cross = sum_{i,j: l_i != l_j} A_ij
        # = sum_{i,j} A_ij - sum_{i,j: l_i == l_j} A_ij
        # = TotalEdges - SameEdges
        # SameEdges = 0.5 * (L^T A L + (sum L)^2? No.)
        
        # Let's use the property:
        # Cut = 0.5 * (sum_{i,j} A_ij (1 - (2l_i-1)(2l_j-1)) ) / 2 ?
        # Let s_i = 1 if l_i=1, -1 if l_i=0.
        # Edge (i,j) is in cut if s_i != s_j.
        # Indicator = (1 - s_i s_j) / 2.
        # Cut = sum_{i<j} A_ij * (1 - s_i s_j) / 2
        #     = 0.5 * (TotalEdges - sum_{i<j} A_ij s_i s_j)
        #     = 0.5 * (TotalEdges - 0.5 * s^T A s)
        #     = 0.5 * TotalEdges - 0.25 * s^T A s
        
        s = 2.0 * labels - 1.0
        total_edges = 0.5 * np.sum(adj)
        s_as = s @ (adj @ s)
        return 0.5 * total_edges - 0.25 * s_as

    # Helper: Local Search (Greedy Flip)
    def local_search(labels):
        # Vectorized greedy pass
        # Gain from flipping v:
        # If v is in group 0, moving to 1 changes cut by:
        # (neighbors in 1) - (neighbors in 0)
        # Let's compute for all v at once.
        
        changed = True
        while changed:
            changed = False
            # For each vertex v, compute gain of flipping
            # Current contribution of v to cut:
            # If label[v] == 0: contrib = sum(A[v, labels==1])
            # If label[v] == 1: contrib = sum(A[v, labels==0])
            
            # Let's compute potential cut contribution if flipped.
            # This is O(n^2) per pass. With n=100, it's very fast.
            
            # Vectorized approach for single pass:
            # For v with label 0:
            #   Current cut edges from v: adj[v, labels==1].sum()
            #   New cut edges from v if flipped to 1: adj[v, labels==0].sum()
            #   Gain = New - Current = adj[v, labels==0].sum() - adj[v, labels==1].sum()
            
            # For v with label 1:
            #   Current: adj[v, labels==0].sum()
            #   New (flip to 0): adj[v, labels==1].sum()
            #   Gain = New - Current = adj[v, labels==1].sum() - adj[v, labels==0].sum()
            
            # We can compute "score" for being 0 and being 1 for each node.
            # score_0[v] = sum of edges from v to nodes labeled 1
            # score_1[v] = sum of edges from v to nodes labeled 0
            
            # Let M0 be mask of labels==0, M1 be mask of labels==1
            m0 = labels == 0
            m1 = labels == 1
            
            # Sum of adj to 1s for all nodes:
            sum_to_1 = adj @ m1.astype(np.float64)
            # Sum of adj to 0s for all nodes:
            sum_to_0 = adj @ m0.astype(np.float64)
            
            # If v is 0, gain to flip to 1 is sum_to_0[v] - sum_to_1[v]
            # If v is 1, gain to flip to 0 is sum_to_1[v] - sum_to_0[v]
            
            # Let's define gain_flip[v]
            # gain_flip[v] = (sum_to_0[v] - sum_to_1[v]) if labels[v]==0 else (sum_to_1[v] - sum_to_0[v])
            
            gain_flip = np.where(m0, sum_to_0 - sum_to_1, sum_to_1 - sum_to_0)
            
            # Flip if gain > 0
            to_flip = gain_flip > 0
            
            if np.any(to_flip):
                labels[to_flip] = 1 - labels[to_flip]
                changed = True
                
            # Check time
            if time.time() > deadline:
                break
                
        return labels

    # 1. Spectral Initialization
    # Compute Laplacian L = D - A
    # D = diag(sum(A, axis=1))
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj
    
    # Find eigenvector corresponding to smallest non-zero eigenvalue (Fiedler vector)
    # For n=100, full eigendecomposition is fast (<1ms)
    try:
        vals, vecs = np.linalg.eigh(L)
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # The smallest eigenvalue should be 0 (for connected graph).
        # The second smallest is the Fiedler vector.
        # If graph is disconnected, there might be multiple zeros.
        # We want the first non-zero eigenvalue's vector.
        fiedler_idx = 1
        # Check if vals[0] is close to 0. If vals[1] is also close to 0, graph might be disconnected.
        # Standard practice: take the eigenvector of the smallest positive eigenvalue.
        # For robustness, just take index 1 unless vals[1] is very small (close to vals[0]).
        # Actually, for Max-Cut, the Fiedler vector (2nd smallest) is the standard spectral partition.
        
        fiedler = vecs[:, 1]
        
        # Partition based on sign
        # Threshold at 0
        labels = (fiedler > 0).astype(int)
        
    except Exception:
        # Fallback to random
        labels = np.random.randint(0, 2, n)

    # 2. Local Search to improve spectral seed
    best_labels = local_search(labels.copy())
    best_score = cut_size(best_labels)
    
    # 3. Multi-restart with random perturbations
    # Use remaining time for random restarts
    # Each restart: take best_labels, flip a random subset, run local search
    
    # Estimate time per local search
    # For n=100, one LS pass is very fast. Let's allow many restarts.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0:  # Keep 5s safety margin
        # Generate new initial guess
        # Strategy: Perturb the best solution found so far
        # Flip 10-30% of vertices randomly
        flip_prob = 0.2
        mask = rng.random(n) < flip_prob
        new_labels = best_labels.copy()
        new_labels[mask] = 1 - new_labels[mask]
        
        # Run local search
        candidate_labels = local_search(new_labels)
        candidate_score = cut_size(candidate_labels)
        
        if candidate_score > best_score:
            best_score = candidate_score
            best_labels = candidate_labels
        
        # Occasionally try a completely random start to escape deep local optima
        if rng.random() < 0.1:
            random_labels = rng.integers(0, 2, n)
            candidate_labels = local_search(random_labels)
            candidate_score = cut_size(candidate_labels)
            if candidate_score > best_score:
                best_score = candidate_score
                best_labels = candidate_labels

    return best_labels
# EVOLVE-BLOCK-END
