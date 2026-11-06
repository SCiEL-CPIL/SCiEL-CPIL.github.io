"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using multi-start local search with simulated annealing.
    
    Strategy:
    1. Convert adjacency to a numpy boolean matrix for fast operations.
    2. Use multiple random restarts.
    3. For each restart, run a combination of:
       a. Greedy local search (flip vertices that improve cut)
       b. Simulated annealing to escape local optima
    4. Keep the best solution found within the time budget.
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, leave 5s safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy bool matrix for efficient operations
    adj = np.array(adjacency, dtype=bool)
    
    # Precompute degrees for efficiency
    degrees = adj.sum(axis=1)
    total_degree = degrees.sum()
    
    def compute_cut(labels):
        """Compute the number of edges crossing the partition."""
        # For each vertex, count neighbors with same label
        same_counts = np.array([adj[v][labels == labels[v]].sum() for v in range(n)])
        return (total_degree - same_counts.sum()) / 2.0
    
    def greedy_improve(labels, max_iters=50):
        """
        Greedy local search: repeatedly flip vertices that improve cut size.
        Returns improved labels and cut size.
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            improved = False
            # Try flipping each vertex
            for v in range(n):
                if time.time() > DEADLINE:
                    break
                same_side = (labels == labels[v])
                other_side = (labels != labels[v])
                gain = np.sum(adj[v, same_side])
                loss = np.sum(adj[v, other_side])
                delta = gain - loss
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True
            if not improved:
                break
        
        return labels, current_cut
    
    def simulated_annealing(labels, start_temp=1.0, end_temp=0.01, iterations=200):
        """
        Simulated annealing to escape local optima.
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        for i in range(iterations):
            if time.time() > DEADLINE:
                break
            temp = start_temp * (end_temp / start_temp) ** (i / iterations)
            
            # Pick a random vertex to flip
            v = np.random.randint(n)
            old_label = labels[v]
            labels[v] = 1 - old_label
            new_cut = compute_cut(labels)
            delta = new_cut - current_cut
            
            if delta > 0 or np.random.random() < np.exp(delta / max(temp, 1e-10)):
                current_cut = new_cut
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                labels[v] = old_label  # Revert
        
        return best_labels, best_cut
    
    # Multi-start approach
    best_labels = None
    best_cut = -1
    
    # First try the alternating start
    labels = np.arange(n) % 2
    labels, current_cut = greedy_improve(labels)
    if current_cut > best_cut:
        best_cut = current_cut
        best_labels = labels.copy()
    
    # Then try random restarts
    rng = np.random.default_rng(42)
    restart_count = 0
    
    while time.time() < DEADLINE:
        # Random initial labeling
        labels = rng.integers(0, 2, size=n)
        
        # Greedy improvement
        labels, current_cut = greedy_improve(labels)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
        
        # If we still have time, try simulated annealing from this solution
        if time.time() < DEADLINE - 5.0:  # Leave some buffer
            sa_labels, sa_cut = simulated_annealing(labels, iterations=150)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
        
        restart_count += 1
        # Limit restarts to avoid infinite loop if time check is too lenient
        if restart_count > 1000:
            break
    
    if best_labels is None:
        # Fallback: return alternating pattern
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
