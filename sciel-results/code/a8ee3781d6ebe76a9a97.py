"""Combined Maximum Cut: spectral init + vectorized local search + SA escape."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Combines:
      1) Vectorized greedy local search (fast 1-flip passes)
      2) Simulated Annealing to escape local optima
      3) Multiple restarts with different initializations
      4) Time guard to stay within deadline
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    # Precompute degrees for efficient gain calculation
    degrees = adj.sum(axis=1)
    
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    
    # Spectral initialization using Fiedler vector
    def spectral_init():
        """Use Fiedler vector of graph Laplacian for initial partition."""
        try:
            # Build sparse Laplacian
            L = laplacian(adj.astype(np.float64))
            # Get second smallest eigenvalue/eigenvector (Fiedler vector)
            # k=2 because we want the 2nd smallest (index 1), smallest is 0
            if n > 10:
                vals, vecs = eigsh(L, k=2, which='SM')
                # Fiedler vector is the eigenvector corresponding to 2nd smallest eigenvalue
                # Sort by eigenvalue
                idx = np.argsort(vals)
                fiedler = vecs[:, idx[1]]
            else:
                # For small graphs, use dense eigendecomposition
                L_dense = adj.astype(np.float64).copy()
                np.fill_diagonal(L_dense, -1)
                L_dense += np.diag(degrees.astype(np.float64))
                vals, vecs = np.linalg.eigh(L_dense)
                fiedler = vecs[:, 1]
            
            # Partition based on median of Fiedler vector
            median_val = np.median(fiedler)
            labels = (fiedler > median_val).astype(np.int8)
            return labels
        except Exception:
            # Fallback to random if spectral fails
            return np.random.randint(0, 2, size=n).astype(np.int8)
    
    best_labels = None
    best_cut = -1
    
    def compute_cut(labels):
        """Number of edges crossing the partition."""
        mask0 = labels == 0
        mask1 = labels == 1
        cut = adj[mask0][:, mask1].sum()
        return int(cut)
    
    def greedy_local_search(labels, max_passes=20):
        """
        Fast greedy local search: flip vertex if it increases cut size.
        Uses efficient gain computation: gain = degree[v] - 2*other
        """
        labels = labels.copy()
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            for v in range(n):
                if time.time() > deadline - 2.0:
                    return labels
                
                # Count neighbors in opposite group
                if labels[v] == 0:
                    other = adj[v, 1].sum()
                else:
                    other = adj[v, 0].sum()
                
                # Gain from flipping: degree[v] - 2*other
                gain = degrees[v] - 2 * other
                
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        
        return labels
    
    def simulated_annealing(labels, initial_temp=1.0, cooling_rate=0.995, max_iters=5000):
        """
        Simulated Annealing to escape local optima.
        Accepts worse moves with probability exp(-delta/T).
        """
        labels = labels.copy()
        temp = initial_temp
        current_cut = compute_cut(labels)
        best_local_cut = current_cut
        best_local_labels = labels.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            # Pick a random vertex to flip
            v = np.random.randint(n)
            
            # Compute gain from flipping v
            if labels[v] == 0:
                other = adj[v, 1].sum()
            else:
                other = adj[v, 0].sum()
            gain = degrees[v] - 2 * other
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if gain >= 0 or np.random.random() < np.exp(-gain / temp):
                labels[v] = 1 - labels[v]
                current_cut += gain
                
                if current_cut > best_local_cut:
                    best_local_cut = current_cut
                    best_local_labels = labels.copy()
            
            # Cool down
            temp *= cooling_rate
        
        return best_local_labels, best_local_cut
    
    rng = np.random.default_rng(seed=42)
    
    # Strategy 1: Spectral initialization (most promising)
    labels_spec = spectral_init()
    labels_improved = greedy_local_search(labels_spec)
    cut = compute_cut(labels_improved)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels_improved.copy()
    
    # Strategy 2: Alternating start
    labels_alt = np.arange(n) % 2
    labels_improved = greedy_local_search(labels_alt)
    cut = compute_cut(labels_improved)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels_improved.copy()
    
    # Strategy 3: Random restarts with greedy local search
    for restart in range(80):
        if time.time() > deadline - 15.0:
            break
        
        labels_rand = rng.integers(0, 2, size=n).astype(np.int8)
        labels_improved = greedy_local_search(labels_rand)
        cut = compute_cut(labels_improved)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels_improved.copy()
    
    # Strategy 4: Simulated Annealing from best solution found so far
    if time.time() < deadline - 20.0 and best_labels is not None:
        sa_labels, sa_cut = simulated_annealing(
            best_labels,
            initial_temp=1.0,
            cooling_rate=0.995,
            max_iters=10000
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Strategy 5: More random restarts with SA if time permits
    for restart in range(40):
        if time.time() > deadline - 5.0:
            break
        
        labels_rand = rng.integers(0, 2, size=n).astype(np.int8)
        # First do greedy to get to a local optimum
        labels_greedy = greedy_local_search(labels_rand)
        # Then apply SA to escape
        sa_labels, sa_cut = simulated_annealing(
            labels_greedy,
            initial_temp=0.5,
            cooling_rate=0.995,
            max_iters=2000
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
