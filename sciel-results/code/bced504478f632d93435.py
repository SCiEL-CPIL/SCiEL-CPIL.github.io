import numpy as np
import time
import random

def partition(adjacency):
    """
    Partition the graph into two sets to maximize the number of crossing edges.
    
    Args:
        adjacency: A boolean matrix (or array convertible to bool) representing the graph.
        
    Returns:
        A numpy array of 0/1 labels for each vertex.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to float for efficient matrix operations
    A = np.array(adjacency, dtype=np.float64)
    
    # Deadline: Use 100 seconds to be safe against the 120s limit
    deadline = time.time() + 100.0
    
    # 1. Spectral Initialization
    # Compute Laplacian L = D - A
    # For undirected graphs, D is the degree matrix
    degrees = A.sum(axis=1)
    D = np.diag(degrees)
    L = D - A
    
    # Find the eigenvector corresponding to the smallest non-zero eigenvalue
    # Since L is symmetric, we can use eigh.
    # For n=100, this is very fast.
    try:
        # We want the eigenvector for the second smallest eigenvalue (Fiedler vector)
        # If the graph is disconnected, the smallest eigenvalue is 0 with multiplicity > 1.
        # np.linalg.eigh returns eigenvalues in ascending order.
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Find the first eigenvalue that is significantly non-zero
        # Usually index 1, but if the graph is disconnected, index 0 might also be ~0.
        # We look for the first eigenvalue > 1e-10
        idx = 1
        for i in range(len(eigvals)):
            if eigvals[i] > 1e-10:
                idx = i
                break
        if idx >= len(eigvals):
            idx = 1 # Fallback
            
        # The eigenvector at idx gives the initial labels
        # Values > 0 -> 1, else -> 0
        labels = (eigvecs[:, idx] > 0).astype(np.int8)
        
    except Exception:
        # Fallback to random or alternating if spectral fails
        labels = np.random.randint(0, 2, n).astype(np.int8)

    # Helper function to calculate cut size
    def get_cut(labels):
        # Cut size is number of edges (u,v) where labels[u] != labels[v]
        # Efficient calculation:
        # Sum of A[i,j] where i in 0, j in 1 + i in 1, j in 0
        # This is equivalent to 0.5 * sum_{i,j} A[i,j] * (1 - (2*labels[i]-1)*(2*labels[j]-1))
        # Or simpler: 
        # Let S0 be set of nodes with label 0, S1 be set of nodes with label 1.
        # Cut = sum(A[S0, S1])
        # Using matrix multiplication is fast for n=100.
        diff = (labels[:, None] != labels[None, :])
        return int((A * diff).sum() // 2)

    current_cut = get_cut(labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # 2. Greedy Local Search
    # Iterate until no improvement is found or time runs out
    improved = True
    while improved and time.time() < deadline - 5.0: # Reserve 5s for SA
        improved = False
        # Randomize order to avoid bias, or just iterate 0..n-1
        # For n=100, iterating 0..n-1 is fine.
        for v in range(n):
            # Calculate gain of flipping v
            # Current contribution of v to cut: edges from v to opposite set
            # Let's compute the number of neighbors in the opposite set (crossing)
            # and neighbors in the same set (non-crossing).
            
            # Neighbors in opposite set: A[v, labels != labels[v]]
            # Neighbors in same set: A[v, labels == labels[v]]
            
            same_mask = labels == labels[v]
            opp_mask = labels != labels[v]
            
            same_count = A[v, same_mask].sum()
            opp_count = A[v, opp_mask].sum()
            
            # If we flip v, 'same' becomes 'opp' and vice versa.
            # New cut contribution from v's edges = same_count
            # Old cut contribution from v's edges = opp_count
            # Gain = same_count - opp_count
            
            if same_count > opp_count:
                labels[v] = 1 - labels[v]
                improved = True
                # Update cut size efficiently? 
                # For n=100, recalculating full cut is O(n^2) = 10k ops.
                # Doing it every flip is okay, but we can track it.
                # Gain was same_count - opp_count.
                # So current_cut += (same_count - opp_count)
                # However, same_count and opp_count are based on the state BEFORE flip.
                # Let's just recalculate to be safe and simple, it's fast enough.
                new_cut = get_cut(labels)
                if new_cut > best_cut:
                    best_cut = new_cut
                    best_labels = labels.copy()
                current_cut = new_cut
            else:
                # No flip
                pass
        
        # If no improvement in a full pass, we are at a local optimum for greedy.

    # 3. Simulated Annealing
    # Use remaining time to explore
    if time.time() < deadline - 2.0:
        # Start temperature
        T = 10.0
        # Cooling rate
        alpha = 0.999
        
        # We will perform moves until time runs out
        # A move is flipping a random vertex
        
        current_labels = best_labels.copy()
        current_cut = best_cut
        
        while time.time() < deadline - 2.0:
            # Pick a random vertex
            v = random.randint(0, n - 1)
            
            # Calculate delta for flipping v
            same_mask = current_labels == current_labels[v]
            opp_mask = current_labels != current_labels[v]
            
            same_count = A[v, same_mask].sum()
            opp_count = A[v, opp_mask].sum()
            
            delta = same_count - opp_count
            
            # Accept if delta > 0 or with probability exp(delta / T)
            if delta > 0:
                accept = True
            else:
                # delta is negative
                # Probability = exp(delta / T)
                # If T is small, this is very small
                prob = np.exp(delta / T)
                accept = random.random() < prob
            
            if accept:
                current_labels[v] = 1 - current_labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = current_labels.copy()
            
            # Cool down
            T *= alpha
            if T < 1e-6:
                # If temperature gets too low, we are essentially doing greedy again
                # We can reset temperature occasionally or just let it run
                # Resetting helps escape deep local minima
                T = 1.0
        
    return best_labels
