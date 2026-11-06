"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    # Convert to numpy array for fast operations
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Deadline guard: kill after 145s to be safe before 150s limit
    start_time = time.time()
    deadline = start_time + 145.0
    
    def cut_size(labels):
        # Count edges crossing the partition
        # An edge (i,j) crosses if labels[i] != labels[j]
        # For undirected graph, each edge counted once
        # labels is 0/1 array
        # Edge crosses if adj[i,j]=1 and labels[i]!=labels[j]
        # Sum over all i<j where adj[i,j]=1 and labels differ
        # Efficient: for each vertex i, count neighbors j with different label
        # But each edge counted twice (once from each end), so divide by 2
        # Actually for max cut, we want the number of edges with endpoints in different sets
        # cut = 0.5 * sum_i sum_j adj[i,j] * (labels[i] != labels[j])
        # Since adj is symmetric, sum_i sum_j adj[i,j]*(labels[i]!=labels[j]) counts each crossing edge twice
        diff = (labels[:, None] != labels[None, :]).astype(np.int8)
        return int(np.sum(adj * diff) // 2)
    
    def local_search(labels, max_rounds=50):
        # Greedy flip-based local search
        # For each vertex, compute gain from flipping
        # If gain > 0, flip
        # Repeat until no improvement or max_rounds
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            improved = False
            # Compute for each vertex the number of neighbors in same set vs different set
            # If same > different, flipping improves the cut
            for v in range(n):
                if time.time() > deadline - 5.0:
                    return best_labels, best_cut
                same = adj[v][best_labels == best_labels[v]].sum()
                other = adj[v][best_labels != best_labels[v]].sum()
                if same > other:
                    best_labels[v] = 1 - best_labels[v]
                    best_cut += (other - same)  # Gain from flip
                    improved = True
            if not improved:
                break
        return best_labels, best_cut
    
    def spectral_init():
        # Use Fiedler vector (second smallest eigenvector of Laplacian) for initialization
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Fiedler vector is eigenvector of second smallest eigenvalue
            # Sort by eigenvalue
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            # Threshold at median
            median_val = np.median(fiedler)
            labels = (fiedler > median_val).astype(np.int8)
            return labels
        except:
            # Fallback to alternating
            return np.arange(n) % 2
    
    def random_init():
        return np.random.randint(0, 2, size=n)
    
    def goemans_williamson_init():
        # SDP-based initialization using random hyperplane rounding
        # Build the matrix for SDP relaxation
        # For max cut, the SDP relaxation involves:
        # maximize sum_{(i,j) in E} (1 - Y[i,j]) / 2
        # subject to Y[i,i] = 1, Y is PSD, Y[i,j] in [-1,1]
        # A common approach: use the top eigenvectors of the adjacency matrix
        # or use the Laplacian-based approach
        
        # Simpler approach: use multiple eigenvectors for random hyperplane rounding
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj.astype(np.float64)
        
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            # Use first few eigenvectors (excluding trivial constant vector)
            # The constant vector corresponds to smallest eigenvalue (0 for connected graph)
            # Use eigenvectors 1 through k for the random hyperplane
            k = min(10, n - 1)
            # Exclude the first eigenvector (constant)
            V = eigvecs[:, idx[1:1+k]]  # shape (n, k)
            
            # Random hyperplane: pick random unit vector in R^k
            best_labels = None
            best_cut = -1
            for trial in range(50):
                if time.time() > deadline - 10.0:
                    break
                random_vec = np.random.randn(k)
                random_vec /= np.linalg.norm(random_vec)
                # Project vertices onto random direction
                projections = V @ random_vec
                # Threshold at 0
                labels = (projections > 0).astype(np.int8)
                c = cut_size(labels)
                if c > best_cut:
                    best_cut = c
                    best_labels = labels.copy()
            
            if best_labels is not None:
                return best_labels
        except:
            pass
        
        # Fallback
        return spectral_init()
    
    # Strategy: Try multiple initializations and local search from each
    # Keep the best result
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral initialization
    labels = spectral_init()
    labels, c = local_search(labels, max_rounds=30)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()
    
    # 2. Goemans-Williamson style initialization
    if time.time() < deadline - 15.0:
        labels = goemans_williamson_init()
        labels, c = local_search(labels, max_rounds=30)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    # 3. Random restarts with local search
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        labels = random_init()
        labels, c = local_search(labels, max_rounds=20)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    # 4. More intensive local search from the best solution found
    if time.time() < deadline - 5.0 and best_labels is not None:
        labels, c = local_search(best_labels, max_rounds=100)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    # 5. Final polish: try all single flips and small perturbations
    if time.time() < deadline - 3.0 and best_labels is not None:
        # Try flipping pairs of vertices
        for _ in range(5):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for v in range(n):
                if time.time() > deadline - 3.0:
                    break
                for w in range(v+1, n):
                    if time.time() > deadline - 3.0:
                        break
                    # Try flipping both v and w
                    old_v = best_labels[v]
                    old_w = best_labels[w]
                    best_labels[v] = 1 - old_v
                    best_labels[w] = 1 - old_w
                    c = cut_size(best_labels)
                    if c > best_cut:
                        best_cut = c
                        improved = True
                    else:
                        best_labels[v] = old_v
                        best_labels[w] = old_w
            if not improved:
                break
    
    # Ensure we return a valid 0/1 array
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
