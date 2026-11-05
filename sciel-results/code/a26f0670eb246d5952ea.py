"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Solve Max-Cut using Spectral Initialization followed by Iterated Local Search (ILS).
    
    Approach:
    1. Compute the Fiedler vector (eigenvector of the graph Laplacian) for a high-quality initial partition.
    2. Refine the partition using a greedy local search (single-flip moves).
    3. Apply Iterated Local Search: perturb the current best solution and re-optimize.
    4. This approach leverages spectral structure for initialization and local optimality for refinement.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    degree = adj.sum(axis=1)
    
    # 1. Spectral Initialization
    # Construct Laplacian L = D - A
    D = np.diag(degree)
    L = D - adj
    
    # Compute eigenvalues and eigenvectors
    # For n=100, this is very fast
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # eigh returns eigenvalues in ascending order
        fiedler_vec = eigenvectors[:, 1]
        
        # Partition based on the sign of the Fiedler vector
        # If a component is close to 0, assign randomly or based on majority
        labels = (fiedler_vec >= 0).astype(np.int8)
        
        # Ensure both partitions are non-empty if possible, though not strictly required for cut size
        # If all are same, flip one arbitrarily
        if np.all(labels == labels[0]):
            labels[0] = 1 - labels[0]
            
    except Exception:
        # Fallback to random initialization if spectral fails
        labels = np.random.randint(0, 2, n).astype(np.int8)

    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        # Efficient calculation: sum of adj[i,j] where labels[i] != labels[j]
        # Using matrix operations
        # cut = sum(adj * (labels[:, None] != labels[None, :])) / 2 
        # But we can use the delta method for updates.
        # For initial calculation:
        return np.sum(adj[labels == 0][:, labels == 1])

    def local_search(labels):
        """
        Greedy Local Search: repeatedly flip the vertex with the maximum positive gain.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        # Precompute gains for each vertex
        # Gain of flipping v = 2 * (sum of adj[v, u] where label[u] == label[v]) - degree[v]
        # Let's maintain a 'gain' array.
        
        def compute_gains(labels):
            gains = np.zeros(n)
            for v in range(n):
                # Number of neighbors in the same partition
                same_count = np.sum(adj[v] * (labels == labels[v]))
                gains[v] = 2.0 * same_count - degree[v]
            return gains

        gains = compute_gains(labels)
        
        improved = True
        while improved:
            improved = False
            max_gain = np.max(gains)
            if max_gain > 1e-9:
                # Find the vertex with max gain
                v = np.argmax(gains)
                
                # Flip vertex v
                labels[v] = 1 - labels[v]
                current_cut += max_gain
                
                # Update gains for v and its neighbors
                # When v flips:
                # 1. Gain for v changes sign? No, gain is calculated based on current state.
                #    After flip, v is in new partition.
                #    New gain for v = 2 * (neighbors in NEW same partition) - degree[v]
                #    Neighbors in NEW same partition = degree[v] - (old neighbors in old same partition)
                #    Old neighbors in old same partition = (old_gain[v] + degree[v]) / 2
                #    So new_same_count = degree[v] - (old_gain[v] + degree[v]) / 2 = (degree[v] - old_gain[v]) / 2
                #    New gain for v = 2 * ((degree[v] - old_gain[v]) / 2) - degree[v] = degree[v] - old_gain[v] - degree[v] = -old_gain[v]
                #    So gain[v] = -gain[v]
                
                old_gain_v = gains[v]
                gains[v] = -old_gain_v
                
                # 2. Update gains for neighbors of v
                #    For a neighbor u of v:
                #    If u was in the same partition as v (before flip), after flip u is in DIFFERENT partition.
                #    The term "neighbors in same partition" for u decreases by 1.
                #    So gain[u] decreases by 2.
                #    If u was in DIFFERENT partition as v (before flip), after flip u is in SAME partition.
                #    The term "neighbors in same partition" for u increases by 1.
                #    So gain[u] increases by 2.
                
                neighbors = np.nonzero(adj[v])[0]
                for u in neighbors:
                    if labels[u] == labels[v]:
                        # u is now in SAME partition as v (after v flipped)
                        # Before flip, u was in DIFFERENT partition from v.
                        # So u gained a same-partition neighbor.
                        gains[u] += 2.0
                    else:
                        # u is now in DIFFERENT partition from v (after v flipped)
                        # Before flip, u was in SAME partition as v.
                        # So u lost a same-partition neighbor.
                        gains[u] -= 2.0
                        
                improved = True
                
        return labels, current_cut

    # Initial Local Search
    best_labels, best_cut = local_search(labels)
    
    # Iterated Local Search
    # Time limit: 110 seconds (leaving 10s margin)
    start_time = time.time()
    max_time = 110.0
    
    # Perturbation strength
    # Flip a small percentage of vertices randomly
    perturbation_ratio = 0.1
    
    while (time.time() - start_time) < max_time:
        # Perturb the best solution
        current_labels = best_labels.copy()
        num_perturbations = max(1, int(n * perturbation_ratio))
        
        # Randomly select vertices to flip
        indices = np.random.choice(n, num_perturbations, replace=False)
        current_labels[indices] = 1 - current_labels[indices]
        
        # Local Search on perturbed solution
        new_labels, new_cut = local_search(current_labels)
        
        # Update best
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = new_labels
            
        # Optional: Simulated Annealing style acceptance for diversity
        # But ILS usually just keeps the best.
        # To escape local optima more effectively, we can sometimes accept worse solutions.
        # However, for Max-Cut, keeping the best is standard.
        # We can increase perturbation strength if stuck.
        
    return best_labels
# EVOLVE-BLOCK-END
