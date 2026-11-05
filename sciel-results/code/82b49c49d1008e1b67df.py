import numpy as np
import time

def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    start_time = time.time()
    DEADLINE = start_time + 135.0

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
        idx = np.argsort(eigvals)
        v = eigvecs[:, idx[1]]
        # Threshold at 0
        x = np.sign(v)
        # Handle zeros (rare)
        x[x == 0] = 1
        return x

    def local_search(x, max_iters=100000):
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

    def two_flip_local_search(x, max_iters=50000):
        # x is +/- 1
        # Gain for flipping pair (i, j): g[i] + g[j] - 2*x[i]*x[j]*adj[i,j]
        # where g[k] = x[k] * (Ax)[k]
        
        Ax = adj @ x
        g = x * Ax
        
        for _ in range(max_iters):
            if time_left() < 1.0:
                break
            
            # Vectorized: G[i,j] = g[i] + g[j] - 2*x[i]*x[j]*adj[i,j] for i<j
            G = g[:, None] + g[None, :] - 2.0 * x[:, None] * x[None, :] * adj
            # Zero out diagonal and lower triangle
            np.fill_diagonal(G, -np.inf)
            G = np.triu(G, k=1)
            
            idx = np.unravel_index(np.argmax(G), G.shape)
            best_gain = G[idx]
            
            if best_gain <= 0:
                break
            
            i, j = idx
            x_old_i = x[i]
            x_old_j = x[j]
            
            # Update Ax: Ax[k] -= 2*x_old_i*adj[k,i] + 2*x_old_j*adj[k,j]
            Ax -= 2.0 * x_old_i * adj[:, i]
            Ax -= 2.0 * x_old_j * adj[:, j]
            
            # Flip x
            x[i] = -x_old_i
            x[j] = -x_old_j
            
            # Recompute g (cheap for n=100)
            g = x * Ax
            
        return x

    def hybrid_local_search(x, max_iters=50000):
        # Alternate between single flip and double flip
        # Or just do single flip until stuck, then double flip until stuck.
        
        x, _ = local_search(x, max_iters=max_iters//2)
        x = two_flip_local_search(x, max_iters=max_iters//2)
        x, _ = local_search(x, max_iters=max_iters//2)
        
        return x

    def ils(x, num_iters=5000):
        best_x = x.copy()
        best_cut = cut_size(x)
        no_improve = 0
        max_no_improve = 100
        
        for i in range(num_iters):
            if time_left() < 5.0:
                break
            
            # Adaptive perturbation: start with 3, increase when stuck
            base_k = 3 + no_improve // 30
            k = min(base_k, max(1, n // 4))
            if n > k:
                flip_idx = np.random.choice(n, k, replace=False)
                x_pert = x.copy()
                x_pert[flip_idx] = -x_pert[flip_idx]
            else:
                x_pert = x.copy()
            
            # Local search
            x_ls = hybrid_local_search(x_pert)
            cut_ls = cut_size(x_ls)
            
            if cut_ls > best_cut:
                best_cut = cut_ls
                best_x = x_ls
                x = x_ls
                no_improve = 0
            else:
                no_improve += 1
                # If stuck for too long, try a bigger jump
                if no_improve >= max_no_improve:
                    k2 = min(10, n // 2)
                    if n > k2:
                        flip_idx2 = np.random.choice(n, k2, replace=False)
                        x_pert2 = x.copy()
                        x_pert2[flip_idx2] = -x_pert2[flip_idx2]
                        x_ls2 = hybrid_local_search(x_pert2)
                        cut_ls2 = cut_size(x_ls2)
                        if cut_ls2 > best_cut:
                            best_cut = cut_ls2
                            best_x = x_ls2
                            x = x_ls2
                            no_improve = 0
                        else:
                            no_improve = 0  # reset to avoid infinite big jumps
                
        return best_x, best_cut

    # 1. Spectral Init
    x = spectral_init()
    
    # 2. Initial Local Search
    x = hybrid_local_search(x)
    best_x = x
    best_cut = cut_size(x)
    
    # 3. ILS
    x, cut = ils(x, num_iters=5000)
    if cut > best_cut:
        best_cut = cut
        best_x = x

    # 4. Final Polish - run multiple rounds of local search
    if time_left() > 2.0:
        for _ in range(3):
            if time_left() < 2.0:
                break
            x = hybrid_local_search(best_x)
            cut = cut_size(x)
            if cut > best_cut:
                best_cut = cut
                best_x = x
            else:
                break

    # Convert to 0/1 labels
    # x is +/- 1. Map 1 -> 1, -1 -> 0
    labels = (best_x + 1) // 2
    return labels.astype(int)
