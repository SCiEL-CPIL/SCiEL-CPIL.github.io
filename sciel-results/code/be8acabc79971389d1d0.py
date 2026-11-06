import numpy as np
import time

def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int32)
    n = len(adj)
    deadline = time.time() + 110.0  # Safety margin
    
    # Helper to compute cut size
    def _cut_size(labels):
        diff = (labels[:, None] != labels[None, :]).astype(np.int32)
        return int(np.sum(adj * diff) // 2)
    
    # Spectral initialization
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        labels = (fiedler > 0).astype(np.int32)
    except Exception:
        labels = np.random.randint(0, 2, n).astype(np.int32)
    
    best_labels = labels.copy()
    best_cut = _cut_size(labels)
    
    # Local Search: 1-flip hill climbing
    def _local_search(labels):
        labels = labels.copy()
        cut = _cut_size(labels)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Compute gain for flipping each vertex
            # Vectorized computation of S
            eq = (labels[:, None] == labels[None, :]).astype(np.int32)
            S = adj @ eq
            gains = deg - 2 * S
            
            # Find best gain
            max_gain = np.max(gains)
            if max_gain > 0:
                u = np.argmax(gains)
                labels[u] = 1 - labels[u]
                cut += max_gain
                improved = True
                # Update best
                nonlocal best_cut, best_labels
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels.copy()
            else:
                break
        return cut, labels

    # Run local search from spectral init
    _local_search(labels)
    
    # Run random restarts if time permits
    while time.time() < deadline:
        random_labels = np.random.randint(0, 2, n).astype(np.int32)
        _local_search(random_labels)
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    return best_labels
