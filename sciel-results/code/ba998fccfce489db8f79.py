"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int((adjacency & diff).sum() // 2)


def _compute_flip_gain(adj, labels, v):
    """Compute the change in cut size if we flip vertex v."""
    lv = labels[v]
    # Number of neighbors with same label as v
    same_count = int((adj[v] & (labels == lv)).sum())
    # Number of neighbors with opposite label to v
    other_count = int((adj[v] & (labels == 1 - lv)).sum())
    # Flipping v: edges to same partition become crossing, edges to opposite become non-crossing
    return same_count - other_count


def _simulated_annealing(adj, labels, initial_temp, final_temp, steps_per_restart, deadline):
    """Run simulated annealing on the given labels."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Precompute gains for all vertices
    gains = np.array([_compute_flip_gain(adj, labels, v) for v in range(n)], dtype=np.int32)
    
    # Number of steps to run
    num_steps = steps_per_restart
    
    for step in range(num_steps):
        if time.time() > deadline:
            break
        
        # Cooling schedule: exponential decay
        temp = initial_temp * (final_temp / initial_temp) ** (step / max(num_steps - 1, 1))
        
        # Pick a random vertex to flip
        v = np.random.randint(0, n)
        
        # Compute the gain of flipping v
        delta = _compute_flip_gain(adj, labels, v)
        
        # Accept if improvement, or with probability exp(-delta/temp) if worsening
        if delta > 0:
            # Improvement: always accept
            labels[v] = 1 - labels[v]
            current_cut += delta
            
            # Update gains for v and its neighbors
            for u in np.where(adj[v] == 1)[0]:
                if u != v:
                    gains[u] = _compute_flip_gain(adj, labels, u)
            gains[v] = _compute_flip_gain(adj, labels, v)
        else:
            # Worsening: accept with probability exp(-delta/temp)
            if temp > 0:
                prob = np.exp(-delta / temp)
                if np.random.random() < prob:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    
                    # Update gains for v and its neighbors
                    for u in np.where(adj[v] == 1)[0]:
                        if u != v:
                            gains[u] = _compute_flip_gain(adj, labels, u)
                    gains[v] = _compute_flip_gain(adj, labels, v)
        
        # Track best solution found
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def _local_search(adj, labels, max_time=None):
    """Final greedy polish: strict hill-climbing to local optimum."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    
    start_time = time.time()
    deadline = start_time + (max_time if max_time else 100)
    
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() > deadline:
                return labels, current_cut
            
            delta = _compute_flip_gain(adj, labels, v)
            if delta > 0:
                labels[v] = 1 - labels[v]
                current_cut += delta
                improved = True
    
    return labels, current_cut


def partition(adjacency):
    """Find a good partition using simulated annealing with multiple restarts."""
    n = len(adjacency)
    start_time = time.time()
    total_deadline = start_time + 118  # Safety margin before 120s limit
    
    best_labels = None
    best_cut = -1
    
    adj = adjacency.astype(np.int8)
    
    # Parameters for simulated annealing
    initial_temp = 50.0
    final_temp = 0.1
    steps_per_sa = 5000
    
    # Number of restarts
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > total_deadline - 10:
            break
        
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Run simulated annealing
        sa_deadline = total_deadline - time.time() - 5
        labels, cut = _simulated_annealing(adj, labels, initial_temp, final_temp, steps_per_sa, sa_deadline)
        
        # Final greedy polish
        polish_deadline = total_deadline - time.time() - 2
        if time.time() < polish_deadline:
            labels, cut = _local_search(adj, labels, max_time=polish_deadline - time.time())
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we've used up most of our time, stop
        if time.time() > total_deadline - 15:
            break
    
    # If no best found, use alternating
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
