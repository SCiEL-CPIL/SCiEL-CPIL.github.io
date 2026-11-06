"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline: 120 seconds total, leave 10s safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        s = 2 * labels - 1  # ±1 vector
        val = s @ (adj @ s)
        total_edges = adj.sum() // 2
        return total_edges - val // 2
    
    def spectral_init(adj):
        """Generate initial labeling using Fiedler vector of graph Laplacian."""
        # Build Laplacian L = D - A
        degree = adj.sum(axis=1)
        D = sp.diags(degree)
        A_sparse = sp.csr_matrix(adj)
        L = D - A_sparse
        
        # Compute smallest eigenvalues of L
        # We want the second smallest (Fiedler vector)
        try:
            # Use eigsh to find smallest eigenvalues
            # k=2 to get the smallest and second smallest
            eigenvalues, eigenvectors = spla.eigsh(L, k=2, which='SM')
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, idx[1]]
            
            # Convert to 0/1 labels based on sign
            labels = (fiedler > 0).astype(int)
            
            # If all labels are the same, try random
            if np.all(labels == labels[0]):
                labels = np.random.randint(0, 2, size=n)
            return labels
        except Exception:
            # Fallback to random if eigsh fails
            return np.random.randint(0, 2, size=n)
    
    def flip_gain(v, labels, adj):
        """Gain in cut size if we flip vertex v."""
        same = adj[v, labels == labels[v]].sum()
        deg = adj[v].sum()
        return 2 * same - deg
    
    def local_search_pq(labels, adj, max_iters=10000, time_limit=None):
        """Best-first local search using a priority queue of gains."""
        labels = labels.copy()
        # Compute initial gains for all vertices
        gains = np.array([flip_gain(v, labels, adj) for v in range(n)])
        # Max-heap via negation
        pq = [(-g, v) for v, g in enumerate(gains) if g > 0]
        heapq.heapify(pq)
        
        iters = 0
        while pq and iters < max_iters:
            if time_limit is not None and time.time() > time_limit:
                break
            iters += 1
            neg_gain, v = heapq.heappop(pq)
            gain = -neg_gain
            if gain <= 0:
                continue
            # Verify gain is still valid (labels may have changed)
            actual_gain = flip_gain(v, labels, adj)
            if actual_gain <= 0:
                continue
            # Flip vertex v
            labels[v] = 1 - labels[v]
            # Update gains for v and its neighbors
            gains[v] = -actual_gain  # After flip, gain becomes negative
            for u in np.nonzero(adj[v])[0]:
                gains[u] = flip_gain(u, labels, adj)
                if gains[u] > 0:
                    heapq.heappush(pq, (-gains[u], u))
        return labels
    
    def simulated_annealing(labels, adj, iters=50000, t_start=1.0, t_end=0.01, time_limit=None):
        """Simulated annealing for max cut."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / iters)
        
        for it in range(iters):
            if time_limit is not None and time.time() > time_limit:
                break
            t *= cooling
            # Pick random vertex
            v = random.randint(0, n - 1)
            gain = flip_gain(v, labels, adj)
            if gain > 0 or (t > 0.001 and random.random() < np.exp(gain / t)):
                labels[v] = 1 - labels[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut
    
    # Multi-start with different initializations
    best_labels = None
    best_cut = -1
    
    # Start 1: spectral initialization
    labels = spectral_init(adj)
    labels = local_search_pq(labels, adj, max_iters=10000, time_limit=deadline)
    c = cut_size(labels)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()
    
    # Start 2: alternating
    labels = np.arange(n) % 2
    labels = local_search_pq(labels, adj, max_iters=10000, time_limit=deadline)
    c = cut_size(labels)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()
    
    # Start 3: random
    for start_iter in range(5):
        if time.time() > deadline:
            break
        labels = np.random.randint(0, 2, size=n)
        labels = local_search_pq(labels, adj, max_iters=10000, time_limit=deadline)
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    # Simulated annealing from best solution found so far
    if time.time() < deadline - 5:
        sa_labels, sa_cut = simulated_annealing(
            best_labels, adj, iters=100000, t_start=1.0, t_end=0.01,
            time_limit=deadline - 5
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Final high-precision polish pass with more iterations
    if time.time() < deadline:
        best_labels = local_search_pq(best_labels, adj, max_iters=50000, time_limit=deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
