import numpy as np
import time

def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    start_time = time.time()
    DEADLINE = start_time + 140.0

    def time_left():
        return DEADLINE - time.time()

    E = int(np.sum(adj)) // 2
    degrees = adj.sum(axis=1)

    def cut_size(x):
        # x is +/- 1
        return E - 0.5 * (x @ (adj @ x))

    def spectral_init():
        # Laplacian
        L = np.diag(degrees) - adj
        # Symmetric eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(L)
        # Second smallest eigenvalue
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        v = eigvecs[:, idx[1]]
        # Threshold at 0
        x = np.sign(v)
        # Handle zeros (rare)
        x[x == 0] = 1
        return x

    def local_search(x, max_iters=10000):
        # x is +/- 1
        Ax = adj @ x
        gains = 2 * x * Ax
        current_cut = cut_size(x)
        
        for _ in range(max_iters):
            if time_left() < 1.0:
                break
            
            # Find vertex with max gain
            v = np.argmax(gains)
            max_gain = gains[v]
            
            if max_gain <= 0:
                break
            
            # Flip v
            x_old_v = x[v]
            
            # Update Ax
            # Ax = Ax - 2 * x_old_v * adj[:, v]
            Ax -= 2 * x_old_v * adj[:, v]
            
            # Flip x
            x[v] = -x_old_v
            
            # Update gains
            # For neighbors u of v:
            # gains[u] -= 4 * x[u] * x_old_v * adj[u, v]
            # Note: x[u] is current x[u]
            # adj[u, v] is 1 if connected
            neighbors = np.where(adj[v] > 0)[0]
            if len(neighbors) > 0:
                gains[neighbors] -= 4 * x[neighbors] * x_old_v * adj[neighbors, v]
            
            # For v itself:
            # gains[v] = -gains[v]
            gains[v] = -gains[v]
            
            # Update current cut
            current_cut += max_gain
            
        return x, current_cut

    def ils(x, num_iters=1000):
        best_x = x.copy()
        best_cut = cut_size(x)
        
        for i in range(num_iters):
            if time_left() < 5.0:
                break
            
            # Perturb
            x_pert = x.copy()
            # Flip 5 random vertices
            k = 5
            if n > k:
                flip_idx = np.random.choice(n, k, replace=False)
                x_pert[flip_idx] = -x_pert[flip_idx]
            
            # Local search
            x_ls, cut_ls = local_search(x_pert)
            
            if cut_ls > best_cut:
                best_cut = cut_ls
                best_x = x_ls
                x = x_ls
            else:
                # Revert to previous best (standard ILS)
                # Or accept with probability? Let's stick to standard first.
                pass
                
        return best_x, best_cut

    # 1. Spectral Init
    x = spectral_init()
    
    # 2. Initial Local Search
    x, cut = local_search(x)
    best_x = x
    best_cut = cut
    
    # 3. ILS
    # Estimate time per ILS iteration
    # Local search is fast.
    x, cut = ils(x, num_iters=5000)
    if cut > best_cut:
        best_cut = cut
        best_x = x

    # 4. Final Polish
    if time_left() > 1.0:
        x, cut = local_search(best_x)
        if cut > best_cut:
            best_cut = cut
            best_x = x

    # Convert to 0/1 labels
    # x is +/- 1. Map 1 -> 1, -1 -> 0
    labels = (best_x + 1) // 2
    return labels.astype(int)
