import numpy as np
import time
import random

def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # Ensure symmetry and no self-loops
    adj = (adj + adj.T) // 2
    np.fill_diagonal(adj, 0)

    start_time = time.time()
    DEADLINE = start_time + 140.0

    def time_left():
        return DEADLINE - time.time()

    # Precompute
    degrees = adj.sum(axis=1).astype(np.int32)
    E = int(degrees.sum()) // 2
    adj_float = adj.astype(np.float64)
    degrees_float = degrees.astype(np.float64)

    # Neighbor lists for fast iteration
    neighbors = [np.nonzero(adj[i])[0] for i in range(n)]

    def cut_size(labels):
        """Compute cut size efficiently."""
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def compute_gains(labels):
        """Vectorized computation of gain for flipping each vertex.
        gain[v] = 2 * (sum of adj[v,u] where labels[u] == labels[v]) - degrees[v]
        If gain > 0, flipping v increases the cut by gain/2."""
        l0_mask = (labels == 0).astype(np.float64)
        l1_mask = (labels == 1).astype(np.float64)
        
        adj_l0 = adj_float @ l0_mask
        adj_l1 = adj_float @ l1_mask
        
        same_count = np.where(labels == 0, adj_l0, adj_l1)
        gains = 2.0 * same_count - degrees_float
        return gains

    def local_search(labels, max_rounds=200):
        """Hill-climbing local search with pairwise flip moves. Returns best labels found."""
        labels = labels.copy()
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        
        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            
            # Compute gains for all vertices
            gains = compute_gains(labels)
            
            # Try single flips
            improved = False
            # Sort vertices by gain descending for better progress
            order = np.argsort(-gains)
            for v in order:
                if time_left() < 1.0:
                    break
                if gains[v] > 1e-9:
                    # Flip vertex v
                    labels[v] = 1 - labels[v]
                    improved = True
                    # Recompute gains after flip (expensive but correct)
                    gains = compute_gains(labels)
            
            # If no single flip improved, try pairwise flips
            if not improved:
                # Try a few random pairwise flips
                for _ in range(10):
                    if time_left() < 1.0:
                        break
                    v1 = np.random.randint(0, n)
                    v2 = np.random.randint(0, n)
                    if v1 == v2:
                        continue
                    
                    old_cut = cut_size(labels)
                    labels[v1] = 1 - labels[v1]
                    labels[v2] = 1 - labels[v2]
                    new_cut = cut_size(labels)
                    
                    if new_cut > old_cut:
                        improved = True
                        best_cut = new_cut
                        best_labels = labels.copy()
                    else:
                        labels[v1] = 1 - labels[v1]
                        labels[v2] = 1 - labels[v2]
                
                if not improved:
                    break
            
            # Update best
            current_cut = cut_size(labels)
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        return best_labels, best_cut

    def quantum_annealing(labels, max_iters=30000, T_start=5.0, T_end=0.01, alpha=0.99995):
        """Quantum-inspired annealing using transverse field scheduling.
        Uses spin representation and accepts moves based on quantum tunneling probability."""
        labels = labels.copy()
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        current_cut = best_cut
        T = T_start
        
        for it in range(max_iters):
            if time_left() < 3.0:
                break
            
            # Pick random vertex
            v = np.random.randint(0, n)
            
            # Compute gain from flipping v
            gains = compute_gains(labels)
            delta = gains[v] / 2.0  # Change in cut size
            
            # Quantum tunneling: accept with probability based on delta and T
            if delta > 0:
                accept = True
            else:
                # Use quantum-inspired acceptance: exp(delta/T) but with additional
                # transverse field term that allows tunneling through barriers
                # The transverse field increases the probability of accepting
                # slightly worse moves early in the schedule
                transverse_field = 1.0 / (1.0 + np.exp(-T))
                effective_T = T * (1.0 + transverse_field)
                accept = np.random.random() < np.exp(delta / effective_T)
            
            if accept:
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            # Cool down
            T = max(T * alpha, T_end)
        
        return best_labels, best_cut

    def spectral_init():
        """Initialize using spectral method (Laplacian eigenvector)."""
        # Compute Laplacian
        L = np.diag(degrees_float) - adj_float
        # Find eigenvector corresponding to smallest non-zero eigenvalue
        # Use scipy if available, otherwise use numpy
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            L_sparse = csr_matrix(L)
            # Find smallest non-zero eigenvalue
            vals, vecs = eigsh(L_sparse, k=2, which='SM')
            # Use the eigenvector corresponding to the second smallest eigenvalue
            idx = np.argsort(vals)
            vec = vecs[:, idx[1]]
            # Threshold to get 0/1 labels
            labels = (vec > 0).astype(np.int8)
        except Exception:
            # Fallback: use alternating pattern
            labels = np.arange(n) % 2
        return labels

    # Strategy:
    # 1. Spectral initialization + local search
    # 2. Alternating start + local search
    # 3. Multiple random restarts with local search
    # 4. Quantum annealing from best solution
    # 5. Final polish with local search including pairwise moves

    best_labels = None
    best_cut = -1

    # 1. Spectral initialization
    labels_spec = spectral_init()
    l, c = local_search(labels_spec, max_rounds=100)
    if c > best_cut:
        best_cut = c
        best_labels = l

    # 2. Alternating start
    labels_alt = np.arange(n) % 2
    l, c = local_search(labels_alt, max_rounds=100)
    if c > best_cut:
        best_cut = c
        best_labels = l

    # 3. Random restarts with local search
    num_restarts = 50
    for _ in range(num_restarts):
        if time_left() < 5.0:
            break
        labels_rand = np.random.randint(0, 2, size=n)
        l, c = local_search(labels_rand, max_rounds=50)
        if c > best_cut:
            best_cut = c
            best_labels = l

    # 4. Quantum annealing from best solution
    if time_left() > 20.0:
        l, c = quantum_annealing(best_labels, max_iters=20000, T_start=5.0, T_end=0.01)
        if c > best_cut:
            best_cut = c
            best_labels = l

    # 5. More quantum annealing restarts
    if time_left() > 15.0:
        for _ in range(10):
            if time_left() < 5.0:
                break
            labels_rand = np.random.randint(0, 2, size=n)
            l, c = quantum_annealing(labels_rand, max_iters=10000, T_start=3.0, T_end=0.01)
            if c > best_cut:
                best_cut = c
                best_labels = l

    # 6. Final polish with local search including pairwise moves
    if time_left() > 5.0:
        for _ in range(5):
            if time_left() < 2.0:
                break
            l, c = local_search(best_labels, max_rounds=100)
            if c > best_cut:
                best_cut = c
                best_labels = l
            else:
                break

    return best_labels.astype(int)
