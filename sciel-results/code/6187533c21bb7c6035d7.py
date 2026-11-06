"""Rewritten Maximum Cut program with vectorized local search and more aggressive multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline: 150s wall-clock, leave 20s safety margin
    start_time = time.time()
    deadline = start_time + 125.0
    
    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        s = 2 * labels - 1
        val = s @ (adj @ s)
        total_edges = adj.sum() // 2
        return total_edges - val // 2
    
    def spectral_init(adj):
        """Generate initial labeling using Fiedler vector of graph Laplacian."""
        degree = adj.sum(axis=1)
        D = sp.diags(degree)
        A_sparse = sp.csr_matrix(adj)
        L = D - A_sparse
        
        try:
            eigenvalues, eigenvectors = spla.eigsh(L, k=2, which='SM')
            idx = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, idx[1]]
            labels = (fiedler > 0).astype(int)
            if np.all(labels == labels[0]):
                labels = np.random.randint(0, 2, size=n)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n)
    
    def vectorized_gain(labels, adj):
        """
        Compute gain for each vertex if flipped.
        gain[v] = 2 * (number of same-label neighbors) - degree(v)
        """
        # For each vertex v, count neighbors with same label
        # same_count[v] = sum_u adj[v,u] * (labels[v] == labels[u])
        # We can compute this as:
        #   same_count = (adj @ (labels == labels[None, :]).astype(int)).sum(axis=0)?
        # Actually, let's think differently.
        # Let s = 2*labels - 1 (±1 vector)
        # The number of edges from v to vertices with same label:
        #   same[v] = sum_u adj[v,u] * (s[v] * s[u] > 0)
        # This is hard to vectorize directly. Let's use a different approach.
        #
        # Alternative: 
        #   cut_contribution[v] = number of neighbors of v with different label
        #   same_contribution[v] = degree[v] - cut_contribution[v]
        #   gain[v] = 2 * same_contribution[v] - degree[v] = 2*(degree[v] - cut_contribution[v]) - degree[v]
        #            = degree[v] - 2*cut_contribution[v]
        #
        # cut_contribution[v] = sum_u adj[v,u] * (labels[v] != labels[u])
        # This can be computed as:
        #   Let's create a matrix where M[v,u] = adj[v,u] * (labels[v] != labels[u])
        #   Then cut_contribution[v] = M.sum(axis=1)[v]
        #
        # But this is O(n^2) per iteration which is slow.
        #
        # Better approach:
        #   Let s = 2*labels - 1
        #   adj @ s gives for each vertex v: sum_u adj[v,u] * s[u]
        #   This equals: (number of neighbors with s=1) - (number of neighbors with s=-1)
        #   Let d1[v] = number of neighbors of v with s=1, d_1[v] = number of neighbors of v with s=-1
        #   Then adj_s[v] = d1[v] - d_1[v]
        #   And degree[v] = d1[v] + d_1[v]
        #   So d1[v] = (degree[v] + adj_s[v]) / 2, d_1[v] = (degree[v] - adj_s[v]) / 2
        #
        # If s[v] = 1 (label 1):
        #   same neighbors = d_1[v] (neighbors with s=-1)... wait no.
        #   same neighbors = neighbors with same s = d1[v] if s[v]=1, d_1[v] if s[v]=-1
        #   Actually: same neighbors of v = number of neighbors u with s[u] == s[v]
        #   If s[v] = 1: same = d1[v]
        #   If s[v] = -1: same = d_1[v]
        #
        # cut_contribution[v] = number of neighbors with different s
        #   If s[v] = 1: cut = d_1[v]
        #   If s[v] = -1: cut = d1[v]
        #
        # gain[v] = degree[v] - 2 * cut_contribution[v]
        #   If s[v] = 1: gain = degree[v] - 2*d_1[v] = degree[v] - 2*(degree[v] - adj_s[v])/2 = adj_s[v]
        #   If s[v] = -1: gain = degree[v] - 2*d1[v] = degree[v] - 2*(degree[v] + adj_s[v])/2 = -adj_s[v]
        #
        # So gain[v] = s[v] * (adj @ s)[v]
        # Wait, let me re-verify:
        #   adj_s = adj @ s
        #   If s[v]=1: gain = adj_s[v]
        #   If s[v]=-1: gain = -adj_s[v]
        #   So gain[v] = s[v] * adj_s[v]
        
        s = 2 * labels - 1
        adj_s = adj @ s
        gains = s * adj_s
        return gains
    
    def local_search(labels, adj, max_iters=20000, time_limit=None):
        """Local search: flip vertices with positive gain until no improvement."""
        labels = labels.copy()
        
        iters = 0
        while iters < max_iters:
            if time_limit is not None and time.time() > time_limit:
                break
            iters += 1
            
            gains = vectorized_gain(labels, adj)
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain <= 0:
                break
            
            v = max_gain_idx
            labels[v] = 1 - labels[v]
        
        return labels
    
    def simulated_annealing(labels, adj, iters=50000, t_start=1.0, t_end=0.001, time_limit=None):
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
            
            v = random.randint(0, n - 1)
            s = 2 * labels - 1
            adj_s = adj @ s
            gain = s[v] * adj_s[v]
            
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
    labels = local_search(labels, adj, max_iters=20000, time_limit=deadline)
    c = cut_size(labels)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()
    
    # Start 2: alternating
    labels = np.arange(n) % 2
    labels = local_search(labels, adj, max_iters=20000, time_limit=deadline)
    c = cut_size(labels)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()
    
    # Start 3-8: random starts
    for start_iter in range(6):
        if time.time() > deadline - 30:
            break
        labels = np.random.randint(0, 2, size=n)
        labels = local_search(labels, adj, max_iters=20000, time_limit=deadline)
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    # Simulated annealing from best solution found so far
    if time.time() < deadline - 15:
        sa_labels, sa_cut = simulated_annealing(
            best_labels, adj, iters=200000, t_start=1.0, t_end=0.001,
            time_limit=deadline - 15
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Final polish pass
    if time.time() < deadline - 5:
        best_labels = local_search(best_labels, adj, max_iters=50000, time_limit=deadline - 5)
        c = cut_size(best_labels)
        if c > best_cut:
            best_cut = c
    
    return best_labels
# EVOLVE-BLOCK-END
