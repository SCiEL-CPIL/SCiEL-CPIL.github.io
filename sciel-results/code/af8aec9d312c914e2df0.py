"""Maximum Cut via fast vectorized local search with 1-flip and 2-flip moves, SA escape, and deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.sparse.csgraph import laplacian as sp_laplacian
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses fast vectorized greedy local search, 2-flip moves to escape local optima,
    simulated annealing, multiple restarts, and an explicit deadline guard.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Deadline: return best result before 140s (10s safety margin from 150s kill)
    deadline = time.time() + 140.0

    # Precompute degree for gain calculation
    degree = adj.sum(axis=1)
    total_edges = adj.sum() // 2

    def compute_cut(labels):
        """Number of edges crossing the partition. Fast vectorized version."""
        diff = (labels[None, :] != labels[:, None])
        cut2 = (adj * diff).sum()
        return cut2 // 2

    def fast_local_search(labels, max_rounds=200):
        """
        Fast greedy 1-flip local search.
        Key insight: gain[v] = 2 * (num neighbors with same label as v) - degree[v]
        We maintain same_count incrementally for speed.
        """
        current_cut = compute_cut(labels)
        
        # Precompute same_count for each vertex
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        gains = 2 * same_count - degree
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            best_v = np.argmax(gains)
            best_gain = gains[best_v]
            
            if best_gain <= 0:
                break
            
            # Flip vertex best_v
            old_label = labels[best_v]
            labels[best_v] = 1 - old_label
            current_cut += best_gain
            
            # Update gains for neighbors of best_v
            new_label = 1 - old_label
            neighbors = np.where(adj[best_v] == 1)[0]
            if len(neighbors) > 0:
                sign = 2 * new_label - 1
                delta_same = (2 * labels[neighbors] - 1) * sign
                same_count[neighbors] += delta_same
                gains[neighbors] += 2 * delta_same
            
            # Also update gain for best_v itself
            same_count[best_v] = degree[best_v] - same_count[best_v]
            gains[best_v] = 2 * same_count[best_v] - degree[best_v]
        
        return labels, current_cut

    def two_flip_local_search(labels, max_rounds=50):
        """
        Try 2-flip moves to escape local optima.
        Vectorized computation of 2-flip gains for all pairs.
        """
        current_cut = compute_cut(labels)
        
        # Precompute same_count
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Compute gain for flipping each single vertex
            gains = 2 * same_count - degree
            
            improved = False
            # Try a larger batch of random pairs
            n_pairs_to_try = 500
            for _ in range(n_pairs_to_try):
                if time.time() > deadline:
                    break
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                
                old_li = labels[i]
                old_lj = labels[j]
                
                labels[i] = 1 - old_li
                labels[j] = 1 - old_lj
                
                new_cut = compute_cut(labels)
                gain_2flip = new_cut - current_cut
                
                if gain_2flip > 0:
                    current_cut = new_cut
                    improved = True
                    # Update same_count
                    same_mask = (labels[None, :] == labels[:, None])
                    same_count = (adj * same_mask).sum(axis=1)
                    break
                else:
                    labels[i] = old_li
                    labels[j] = old_lj
            
            if not improved:
                break
        
        return labels, current_cut

    def simulated_annealing(labels, initial_temp=15.0, cooling=0.9995, max_iters=20000):
        """
        Simulated Annealing with 1-flip moves.
        Uses incremental gain computation for speed.
        """
        current_cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # Precompute same_count
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        
        temp = initial_temp
        no_improve_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random vertex
            v = np.random.randint(n)
            
            # Compute gain of flipping v
            same_count_v = same_count[v]
            gain = 2 * same_count_v - degree[v]
            
            # Accept if gain > 0 or with probability exp(gain / temp) for gain < 0
            accepted = False
            if gain >= 0:
                accepted = True
            else:
                # probability = exp(gain / temp), gain is negative
                if np.random.rand() < np.exp(gain / temp):
                    accepted = True
            
            if accepted:
                old_label = labels[v]
                labels[v] = 1 - old_label
                current_cut += gain
                
                # Update same_count for neighbors
                new_label = 1 - old_label
                neighbors = np.where(adj[v] == 1)[0]
                if len(neighbors) > 0:
                    sign = 2 * new_label - 1
                    delta_same = (2 * labels[neighbors] - 1) * sign
                    same_count[neighbors] += delta_same
                
                # Update same_count for v
                same_count[v] = degree[v] - same_count[v]
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            temp *= cooling
            
            # Adaptive cooling: cool faster if stuck
            if no_improve_count > 500:
                temp *= 0.9
                no_improve_count = 0
        
        return best_labels, best_cut

    def greedy_init(adj, n, rng):
        """Greedy initialization: add vertices one by one to the side that maximizes cut."""
        labels = np.zeros(n, dtype=int)
        order = np.arange(n)
        rng.shuffle(order)
        
        for v in order:
            left_count = (adj[v] * (labels == 0)).sum()
            right_count = (adj[v] * (labels == 1)).sum()
            if right_count > left_count:
                labels[v] = 0
            else:
                labels[v] = 1
        return labels

    # Multiple restarts with different initializations
    best_labels_overall = None
    best_cut_overall = -1
    rng = np.random.RandomState(42)
    
    def try_restart(labels_init, tag=""):
        nonlocal best_labels_overall, best_cut_overall
        if time.time() > deadline:
            return
        
        labels = labels_init.copy()
        # Fast local search
        labels, cut = fast_local_search(labels, max_rounds=300)
        
        # Try 2-flip escape
        if time.time() < deadline - 5:
            labels2, cut2 = two_flip_local_search(labels, max_rounds=20)
            if cut2 > cut:
                labels, cut = labels2, cut2
                # Polish after 2-flip
                labels, cut = fast_local_search(labels, max_rounds=100)
        
        if cut > best_cut_overall:
            best_cut_overall = cut
            best_labels_overall = labels.copy()
    
    # Spectral initialization: Fiedler vector of graph Laplacian
    spectral_labels = None
    try:
        # Build sparse adjacency for Laplacian
        adj_sparse = csr_matrix(adj)
        L = sp_laplacian(adj_sparse, normed=False)
        
        # Get Fiedler vector (second-smallest eigenvalue)
        # For connected graphs, use k=2; for disconnected, get smallest non-trivial
        try:
            eigvals, eigvecs = eigsh(L, k=2, which='SM')
            # Sort by eigenvalue
            idx = np.argsort(eigvals)
            fiedler_vec = eigvecs[:, idx[1]]
        except Exception:
            # Fallback: use power iteration or just random
            fiedler_vec = rng.randn(n)
        
        # Convert Fiedler vector to 0/1 labels: threshold at median
        median_val = np.median(fiedler_vec)
        spectral_labels = (fiedler_vec > median_val).astype(np.int8)
    except Exception:
        spectral_labels = None
    
    # Restart 1: spectral (highest priority - uses global structure)
    if spectral_labels is not None:
        try_restart(spectral_labels, "spectral")
    
    # Restart 2: alternating
    try_restart(np.arange(n) % 2, "alt")
    
    # Restart 3: greedy
    try_restart(greedy_init(adj, n, rng), "greedy")
    
    # Restart 4: random
    try_restart(rng.randint(0, 2, size=n), "rand")
    
    # Restart 5: random with bias
    try_restart(rng.randint(0, 2, size=n), "rand2")
    
    # Restart 6: all zeros then flip random
    labels_init = np.zeros(n, dtype=np.int8)
    flip_mask = rng.rand(n) < 0.5
    labels_init[flip_mask] = 1
    try_restart(labels_init, "half")
    
    # Restart 7-10: more random starts
    for _ in range(4):
        if time.time() > deadline:
            break
        try_restart(rng.randint(0, 2, size=n), "rand_extra")
    
    # If we have time, do SA from best solution
    if best_labels_overall is not None and time.time() < deadline - 10:
        labels_sa = best_labels_overall.copy()
        labels_sa, cut_sa = simulated_annealing(labels_sa, initial_temp=20.0, cooling=0.999, max_iters=30000)
        
        # Polish
        labels_sa, cut_sa = fast_local_search(labels_sa, max_rounds=200)
        
        if cut_sa > best_cut_overall:
            best_cut_overall = cut_sa
            best_labels_overall = labels_sa.copy()
    
    # Final polish
    if best_labels_overall is not None and time.time() < deadline - 2:
        labels_final = best_labels_overall.copy()
        labels_final, cut_final = fast_local_search(labels_final, max_rounds=300)
        if cut_final > best_cut_overall:
            best_cut_overall = cut_final
            best_labels_overall = labels_final.copy()
    
    return best_labels_overall
# EVOLVE-BLOCK-END
