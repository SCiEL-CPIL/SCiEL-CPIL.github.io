"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Simulated Annealing with greedy local improvement for Max-Cut
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 150s wall-clock limit, return best with safety margin
    deadline = time.time() + 140.0
    
    def cut_size(labels):
        # Efficient cut size calculation
        # An edge (i,j) is in the cut if labels[i] != labels[j]
        # cut_size = sum over all edges where labels differ
        # = 0.5 * sum over all i,j of adj[i,j] * (1 - labels[i]*labels[j] + labels[i] + labels[j]) / 2... 
        # Actually simpler: for each vertex v, count neighbors with different label
        # But that double counts. Let's use:
        # cut = sum_{i<j} adj[i,j] * (labels[i] != labels[j])
        # Efficient: sum over all v of (sum of adj[v, labels != labels[v]]) / 2
        # Since each cut edge is counted twice (once from each endpoint)
        diff = 1 - (labels[None, :] == labels[:, None]).astype(np.int8)
        return np.sum(adj * diff) // 2
    
    def local_improve(labels):
        # Greedy local improvement: flip vertex if it improves cut
        improved = True
        while improved:
            improved = False
            for v in range(n):
                # Current contribution of v to cut: neighbors with different label
                cur_cross = adj[v][labels != labels[v]].sum()
                # If we flip v, new cross count = neighbors with same label (old)
                new_cross = adj[v][labels == labels[v]].sum()
                if new_cross > cur_cross:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    # Multiple restarts with simulated annealing
    best_labels = None
    best_cut = -1
    
    # Initial solutions: alternating, random, and spectral-inspired
    initial_solutions = []
    
    # Alternating
    initial_solutions.append(np.arange(n) % 2)
    
    # Random
    rng = np.random.default_rng(42)
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    
    # Another random
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    
    # Spectral-like: use degree-based initialization
    degrees = adj.sum(axis=1)
    initial_solutions.append((degrees > degrees.mean()).astype(np.int8))
    
    for init_labels in initial_solutions:
        if time.time() > deadline:
            break
        
        labels = init_labels.copy().astype(np.int8)
        
        # Local improvement first
        labels = local_improve(labels)
        cur_cut = cut_size(labels)
        
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()
        
        # Simulated Annealing
        # Temperature schedule: geometric cooling
        T_start = 10.0
        T_end = 0.1
        n_steps = 5000
        cooling_rate = (T_end / T_start) ** (1.0 / n_steps)
        
        T = T_start
        sa_labels = labels.copy()
        sa_cut = cur_cut
        
        for step in range(n_steps):
            if time.time() > deadline:
                break
            
            # Pick a random vertex to flip
            v = rng.integers(0, n)
            
            # Compute delta in cut size
            cur_cross_v = adj[v][sa_labels != sa_labels[v]].sum()
            new_cross_v = adj[v][sa_labels == sa_labels[v]].sum()
            delta = new_cross_v - cur_cross_v
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                sa_labels[v] = 1 - sa_labels[v]
                sa_cut += delta
            
            # Update best
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
            
            T *= cooling_rate
        
        # Final local improvement on SA result
        if time.time() < deadline:
            polished = local_improve(sa_labels)
            pol_cut = cut_size(polished)
            if pol_cut > best_cut:
                best_cut = pol_cut
                best_labels = polished
    
    # Final check
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
