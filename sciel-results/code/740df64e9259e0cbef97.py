"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    n = len(adjacency)
    # Convert to numpy int array for fast operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: keep a safety margin before the 150s limit
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute degrees for potential use
    degrees = adj.sum(axis=1)
    
    def cut_size(labels):
        # Count edges crossing the partition
        # For each edge (i,j) where i<j, count if labels[i] != labels[j]
        # Vectorized: sum of adj[i,j] * (labels[i] != labels[j])
        # Efficient: (labels[:, None] != labels[None, :]) * adj, sum / 2
        # But even faster: for each vertex, count neighbors with different label
        # cut = sum_v sum_u adj[v,u] * (labels[v] != labels[u]) / 2
        # = 0.5 * sum_v (sum_u adj[v,u] * (labels[v] != labels[u]))
        # For vertex v: if label=0, count neighbors with label 1; if label=1, count neighbors with label 0
        diff_mask = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff_mask) // 2)
    
    def local_search(labels, max_passes=10):
        """Single-flip local search until no improvement."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        for _ in range(max_passes):
            improved = False
            for v in range(n):
                # Count same-side and other-side neighbors
                same_side = adj[v, labels == labels[v]].sum()
                other_side = adj[v, labels != labels[v]].sum()
                if same_side > other_side:
                    labels[v] = 1 - labels[v]
                    current_cut += (other_side - same_side)
                    improved = True
            if not improved:
                break
        return labels, current_cut
    
    def simulated_annealing(labels, t_start=1.0, t_end=0.01, n_iterations=None):
        """Simulated annealing with single-flip moves."""
        if n_iterations is None:
            n_iterations = 20000
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / n_iterations)
        
        for it in range(n_iterations):
            if time.time() > deadline:
                break
            # Pick random vertex to flip
            v = random.randint(0, n - 1)
            same_side = adj[v, labels == labels[v]].sum()
            other_side = adj[v, labels != labels[v]].sum()
            delta = other_side - same_side  # change in cut size if we flip
            
            if delta > 0 or (delta < 0 and random.random() < np.exp(delta / t)):
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            t *= cooling
        
        return best_labels, best_cut
    
    # Multi-start strategy
    best_labels = None
    best_cut = -1
    
    # Start 1: Alternating initialization
    labels1 = np.arange(n) % 2
    labels1, cut1 = local_search(labels1)
    if cut1 > best_cut:
        best_cut = cut1
        best_labels = labels1.copy()
    
    # Start 2: Random initialization
    labels2 = np.random.randint(0, 2, n)
    labels2, cut2 = local_search(labels2)
    if cut2 > best_cut:
        best_cut = cut2
        best_labels = labels2.copy()
    
    # Start 3: Degree-based initialization (high degree nodes start in group 0)
    labels3 = np.zeros(n, dtype=np.int8)
    # Sort by degree descending, assign first half to 0, second half to 1
    order = np.argsort(-degrees)
    labels3[order[:n//2]] = 0
    labels3[order[n//2:]] = 1
    labels3, cut3 = local_search(labels3)
    if cut3 > best_cut:
        best_cut = cut3
        best_labels = labels3.copy()
    
    # Run simulated annealing from the best solution
    if best_labels is not None and time.time() < deadline:
        sa_labels, sa_cut = simulated_annealing(best_labels, t_start=1.0, t_end=0.01, n_iterations=30000)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Run additional local search passes from SA result
    if best_labels is not None and time.time() < deadline:
        final_labels, final_cut = local_search(best_labels, max_passes=20)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()
    
    # If time remains, try more random restarts with SA
    if best_labels is not None and time.time() < deadline:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            rand_labels = np.random.randint(0, 2, n)
            rand_labels, rand_cut = local_search(rand_labels, max_passes=5)
            sa_labels, sa_cut = simulated_annealing(rand_labels, t_start=0.5, t_end=0.01, n_iterations=15000)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
