"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.sparse.linalg as spla

def partition(adjacency):
    """
    Maximum Cut via Spectral Initialization and Greedy Local Search.
    This approach uses the Fiedler vector and other spectral properties
    to generate high-quality initial partitions, then refines them
    using a greedy hill-climbing algorithm.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Precompute degree
    degrees = np.sum(adj, axis=1)
    
    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        # Map 0/1 to -1/1
        s = 2 * labels - 1
        # Cut size = 0.5 * sum_{i,j} adj[i,j] * (1 - s_i * s_j)
        # Efficiently: 0.5 * (sum(adj) - sum(adj * outer(s,s)))
        # sum(adj) is 2 * E (total edges)
        # sum(adj * outer(s,s)) is sum_{i,j} adj[i,j] s_i s_j
        return 0.5 * (np.sum(adj) - np.sum(adj * np.outer(s, s)))

    def greedy_local_search(labels, max_passes=50):
        """
        Greedy hill climbing: flip a vertex if it increases the cut.
        """
        labels = labels.copy()
        # Precompute current state
        # For speed, we can maintain the "score" of each vertex
        # score[v] = sum_{u in N(v)} (1 if labels[v] != labels[u] else 0) - sum_{u in N(v)} (1 if labels[v] == labels[u] else 0)
        # Actually, let's just compute the delta on the fly or maintain a potential.
        
        # Let's use a simple iterative improvement.
        # For each vertex v, let same[v] be count of neighbors with same label.
        # Let diff[v] be count of neighbors with different label.
        # degree[v] = same[v] + diff[v].
        # If we flip v, new_diff = same[v], new_same = diff[v].
        # Gain = new_diff - diff[v] = same[v] - diff[v].
        # We flip if same[v] > diff[v].
        
        current_cut = cut_size(labels)
        
        for _ in range(max_passes):
            improved = False
            # Randomize order to avoid bias? Or fixed?
            # Fixed is faster and deterministic.
            for v in range(n):
                # Count neighbors with same label
                # labels == labels[v] is a boolean vector
                # adj[v] * (labels == labels[v]) gives 1 for neighbors with same label
                same_count = np.sum(adj[v] * (labels == labels[v]))
                # diff_count = degrees[v] - same_count
                diff_count = degrees[v] - same_count
                
                if same_count > diff_count:
                    labels[v] = 1 - labels[v]
                    improved = True
            
            if not improved:
                break
        
        return labels, cut_size(labels)

    # 1. Spectral Initialization
    # Compute Laplacian L = D - A
    D = np.diag(degrees)
    L = D - adj
    
    # Compute eigenvalues and eigenvectors of L
    # Since n=100, dense eigh is fine and fast
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        # Fallback to random if linear algebra fails
        eigvals = None
        eigvecs = None
    
    candidates = []
    
    if eigvals is not None:
        # Sort by eigenvalue (eigh returns sorted ascending)
        # Fiedler vector is index 1 (second smallest)
        # We can use the first few eigenvectors to create random combinations
        
        # Candidate 1: Fiedler vector
        fiedler = eigvecs[:, 1]
        # Split by sign
        init1 = (fiedler > 0).astype(int)
        candidates.append(init1)
        
        # Candidate 2: Fiedler vector with random sign flip (equivalent)
        # candidates.append(1 - init1)
        
        # Candidates 3-6: Random linear combinations of top 4 eigenvectors
        # This explores the spectral subspace
        num_spectral_restarts = 5
        for i in range(num_spectral_restarts):
            # Random weights for the first few eigenvectors
            # Using top 3-5 eigenvectors often helps
            k = min(5, n)
            weights = np.random.randn(k)
            vec = np.dot(eigvecs[:, :k], weights)
            init = (vec > 0).astype(int)
            candidates.append(init)

    # Add some random initializations to ensure diversity
    num_random_restarts = 5
    for i in range(num_random_restarts):
        init = np.random.randint(0, 2, size=n)
        candidates.append(init)

    best_labels = None
    best_cut = -1
    
    # Refine each candidate with Greedy Local Search
    for init_labels in candidates:
        refined_labels, cut_val = greedy_local_search(init_labels)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels

    return best_labels
# EVOLVE-BLOCK-END
