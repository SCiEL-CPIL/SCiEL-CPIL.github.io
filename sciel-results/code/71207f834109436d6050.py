"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Deadline guard: 115s safety margin for 120s limit
    start_time = time.time()
    deadline = start_time + 115.0
    
    def cut_size(labels):
        # Efficient cut size calculation
        # Cross edges = sum(A) / 2 - same_side_edges
        # Or directly: 0.5 * sum(A[i,j] * (1 - (labels[i] == labels[j])))
        # Vectorized:
        diff = labels[:, None] != labels[None, :]
        return int((adj * diff).sum() // 2)
    
    def local_search(labels):
        """
        Perform local search on a given label assignment.
        Uses batch flipping where possible, then sequential for fine-tuning.
        Returns improved labels and cut size.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        # Phase 1: Batch flips
        for _ in range(50):
            if time.time() > deadline - 5.0:  # Leave 5s buffer
                break
            
            # Calculate gain for each vertex
            # gain[v] = 2 * other_count[v] - degree[v]
            # other_count[v] = sum(adj[v, u] where labels[u] != labels[v])
            # same_count[v] = sum(adj[v, u] where labels[u] == labels[v])
            # degree[v] = other_count[v] + same_count[v]
            # gain = other - same = degree - 2 * same_count
            
            same_mask = labels[:, None] == labels[None, :]
            same_count = (adj * same_mask).sum(axis=1)
            degree = adj.sum(axis=1)
            gain = degree - 2.0 * same_count
            
            to_flip = gain > 0
            if not to_flip.any():
                break
                
            labels[to_flip] = 1 - labels[to_flip]
            new_cut = cut_size(labels)
            if new_cut > current_cut:
                current_cut = new_cut
            else:
                # If batch flip didn't improve (shouldn't happen if gain > 0 for all, but safe)
                labels[to_flip] = 1 - labels[to_flip]
        
        # Phase 2: Sequential fine-tuning (single vertex flips)
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for v in range(n):
                same = adj[v, labels == labels[v]].sum()
                other = adj[v, labels != labels[v]].sum()
                if other > same:
                    labels[v] = 1 - labels[v]
                    improved = True
            if not improved:
                break
            current_cut = cut_size(labels)
            
        return labels, current_cut

    # --- Multi-Start Strategy ---
    
    # 1. Spectral Initialization (Fiedler vector)
    best_labels = None
    best_cut = -1
    
    # Try spectral first as it's often a strong single start
    try:
        degree = adj.sum(axis=1)
        L = np.diag(degree) - adj
        eigvals, eigvecs = np.linalg.eigh(L)
        # Use the second smallest eigenvector (index 1)
        # If graph is disconnected, there might be multiple zero eigenvalues.
        # We want the first non-zero eigenvalue's eigenvector.
        # For connected graphs, eigvals[0] is ~0, eigvals[1] is the Fiedler value.
        fiedler = eigvecs[:, 1]
        init_labels = (fiedler >= 0).astype(np.int8)
        if init_labels.sum() == 0 or init_labels.sum() == n:
            init_labels = np.arange(n) % 2
        
        refined_labels, refined_cut = local_search(init_labels)
        if refined_cut > best_cut:
            best_cut = refined_cut
            best_labels = refined_labels
    except Exception:
        pass
    
    # 2. Random Restarts with Local Search
    # Use remaining time for random restarts
    # Each restart: generate random labels, run local search
    # We can also try "random greedy" construction, but random + LS is simpler and effective.
    
    # Estimate cost of one LS run.
    # Batch LS is O(n^2) per iteration, ~50 iterations.
    # Sequential LS is O(n^2) per pass, ~20 passes.
    # Total ~ 70 * n^2 operations. For n=100, 700,000 ops. Very fast.
    # So we can do hundreds of restarts.
    
    # Let's run as many restarts as possible.
    # To add diversity, we can also bias the random initialization.
    
    while time.time() < deadline - 2.0:  # Leave 2s for final return
        # Random initialization
        # 50% chance pure random, 50% chance random with some structure?
        # Let's stick to pure random for now, it explores well.
        rand_labels = np.random.randint(0, 2, size=n).astype(np.int8)
        
        refined_labels, refined_cut = local_search(rand_labels)
        
        if refined_cut > best_cut:
            best_cut = refined_cut
            best_labels = refined_labels
            
        # Optional: If we are stuck for a long time, try a different init strategy?
        # For now, just keep restarting.
        
    # Fallback if no best found (should not happen)
    if best_labels is None:
        best_labels = np.arange(n) % 2
        
    return best_labels
# EVOLVE-BLOCK-END
