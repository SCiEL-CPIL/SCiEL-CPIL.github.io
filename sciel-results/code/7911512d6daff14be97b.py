"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Greedy local improvement from an alternating start.
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Use vectorized operations for speed
    adj = np.array(adjacency, dtype=np.int8)
    
    # Try multiple random restarts with greedy improvement
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random initialization
        if restart == 0:
            labels = np.arange(n) % 2
        else:
            labels = np.random.randint(0, 2, size=n)
        
        # Greedy improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(n):
                if time.time() > deadline:
                    break
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    improved = True
        
        # Calculate cut size
        cut_size = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] and labels[i] != labels[j]:
                    cut_size += 1
        
        if cut_size > best_cut:
            best_cut = cut_size
            best_labels = labels.copy()
    
    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
