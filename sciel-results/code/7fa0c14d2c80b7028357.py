import numpy as np
import time

def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    deg = adj.sum(axis=1)
    
    best_labels = None
    best_cut = -1
    
    # Pre-compute spectral init
    try:
        L = np.diag(deg) - adj
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        spectral_labels = (fiedler > 0).astype(np.int8)
    except Exception:
        spectral_labels = np.arange(n) % 2

    def get_cut(labels):
        diff = (labels[:, None] != labels[None, :]).astype(np.int8)
        return int(np.sum(adj * diff) // 2)

    def local_search(labels):
        labels = labels.copy()
        
        # 1-flip local search (parallel updates)
        for _ in range(100):
            same = (labels[:, None] == labels[None, :]).astype(np.int8)
            same_counts = (adj * same).sum(axis=1)
            gains = 2 * same_counts - deg
            to_flip = np.where(gains > 0)[0]
            if len(to_flip) == 0:
                break
            labels[to_flip] = 1 - labels[to_flip]
            
        # 2-flip local search
        for _ in range(100):
            same = (labels[:, None] == labels[None, :]).astype(np.int8)
            same_counts = (adj * same).sum(axis=1)
            gains_1 = 2 * same_counts - deg
            
            # M[u, v] = gains_1[u] + gains_1[v] + 2 * adj[u, v]
            M = gains_1[:, None] + gains_1[None, :] + 2 * adj
            
            # Mask out pairs with same label and diagonal
            mask = (labels[:, None] != labels[None, :])
            np.fill_diagonal(mask, False)
            
            M_masked = np.where(mask, M, -np.inf)
            
            max_val = np.max(M_masked)
            if max_val <= 0:
                break
                
            max_idx = np.unravel_index(np.argmax(M_masked), M_masked.shape)
            u, v = max_idx
            labels[u] = 1 - labels[u]
            labels[v] = 1 - labels[v]
            
        return labels

    # Restart 1: Spectral
    labels = local_search(spectral_labels)
    cut = get_cut(labels)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
        
    # Restarts: Random and Noisy Spectral
    while time.time() < deadline:
        r = np.random.rand()
        if r < 0.5:
            labels = np.random.randint(0, 2, n).astype(np.int8)
        else:
            labels = spectral_labels.copy()
            noise_mask = np.random.rand(n) < 0.1
            labels[noise_mask] = 1 - labels[noise_mask]
            
        labels = local_search(labels)
        cut = get_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    return best_labels
