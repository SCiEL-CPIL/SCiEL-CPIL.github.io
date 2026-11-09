"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # cut = sum_{v} adj[v] @ (1 - labels)
    # Since adj is symmetric, each crossing edge is counted twice (once from each endpoint)
    # So we divide by 2
    return int(np.sum(adj @ (1 - labels)) // 2)


def _spectral_init(adj, rng):
    """Generate initial partition using spectral embedding (Fiedler vector)."""
    n = len(adj)
    # Compute the Laplacian L = D - A
    degrees = adj.sum(axis=1)
    laplacian = np.diag(degrees) - adj
    
    # Get eigenvalues and eigenvectors
    # We want the second smallest eigenvalue's eigenvector (Fiedler vector)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = eigenvectors[:, 1]
        # Use the sign of the Fiedler vector to create initial partition
        labels = (fiedler > 0).astype(int)
    except Exception:
        # Fallback to random initialization
        labels = rng.integers(0, 2, size=n).astype(int)
    
    return labels


def _greedy_flip_improve(adj, labels, max_passes=50):
    """Improve partition via greedy single-vertex flips."""
    n = len(adj)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    
    for _ in range(max_passes):
        improved = False
        for v in range(n):
            # Count neighbors in same and opposite partition
            same_count = adj[v][labels == labels[v]].sum()
            other_count = adj[v][labels != labels[v]].sum()
            
            # If flipping improves the cut
            if other_count > same_count:
                labels[v] = 1 - labels[v]
                # Update cut size: flipping v changes cut by (other_count - same_count)
                current_cut += (other_count - same_count)
                improved = True
        
        if not improved:
            break
    
    return labels, current_cut


def _two_opt_swap(adj, labels, max_passes=20):
    """Improve partition via 2-opt swaps (swap two vertices)."""
    n = len(adj)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                
                # Try swapping labels[i] and labels[j]
                # Compute gain from swapping
                # For vertex i: neighbors in same group (label 1-labels[i]) vs opposite
                # After swap, i gets label 1-labels[i], j gets label 1-labels[j]
                
                # Gain for i: (neighbors of i with label != 1-labels[i]) - (neighbors of i with label == 1-labels[i])
                # = (neighbors of i with label == labels[i]) - (neighbors of i with label != labels[i])
                gain_i = adj[i][labels == labels[i]].sum() - adj[i][labels != labels[i]].sum()
                
                # Gain for j
                gain_j = adj[j][labels == labels[j]].sum() - adj[j][labels != labels[j]].sum()
                
                # Total gain from swap
                total_gain = gain_i + gain_j
                
                if total_gain > 0:
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    current_cut += total_gain
                    improved = True
        
        if not improved:
            break
    
    return labels, current_cut


def partition(adjacency):
    """
    Partition vertices into two groups to maximize the cut size.
    
    Uses spectral initialization followed by greedy improvement and 2-opt swaps.
    """
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.int32)
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Deadline guard: 145 seconds (5 second safety margin)
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    num_restarts = 10
    
    for restart in range(num_restarts):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Generate initial partition
        if restart == 0:
            # First restart: spectral initialization
            labels = _spectral_init(adj, rng)
        else:
            # Subsequent restarts: random initialization
            labels = rng.integers(0, 2, size=n).astype(int)
        
        # Greedy improvement
        labels, cut = _greedy_flip_improve(adj, labels, max_passes=30)
        
        # Check deadline
        if time.time() > deadline:
            if cut > best_cut:
                best_labels = labels
                best_cut = cut
            break
        
        # 2-opt swaps
        labels, cut = _two_opt_swap(adj, labels, max_passes=10)
        
        # Check deadline
        if time.time() > deadline:
            if cut > best_cut:
                best_labels = labels
                best_cut = cut
            break
        
        # Track best
        if cut > best_cut:
            best_labels = labels
            best_cut = cut
    
    # If no valid result, return alternating pattern
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
