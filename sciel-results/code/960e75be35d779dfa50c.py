import numpy as np
import time
import random

def partition(adjacency):
    adj_np = np.asarray(adjacency, dtype=np.int8)
    n = adj_np.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    
    # Build neighbor list for fast iteration
    neighbors = [[] for _ in range(n)]
    degrees = np.zeros(n, dtype=int)
    for i in range(n):
        idxs = np.nonzero(adj_np[i])[0]
        neighbors[i] = idxs.tolist()
        degrees[i] = len(idxs)
    
    E = int(np.sum(adj_np)) // 2
    
    start_time = time.time()
    DEADLINE = start_time + 110.0
    
    def time_left():
        return DEADLINE - time.time()
        
    def init_cross(labels):
        # Vectorized initialization of cross array
        # cross[v] = number of neighbors of v with label != labels[v]
        # diff_matrix[i, j] is 1 if labels[i] != labels[j]
        diff_matrix = (labels[None, :] != labels[:, None])
        # adj_np is 0/1, diff_matrix is bool (0/1)
        # sum over neighbors for each v
        return np.sum(adj_np * diff_matrix, axis=1).astype(int)

    def run_local_search(labels, max_rounds=1000):
        labels = labels.copy()
        cross = init_cross(labels)
        
        best_cut = 0.5 * np.sum(cross)
        best_labels = labels.copy()
        
        for rnd in range(max_rounds):
            if time_left() < 1.0:
                break
            
            # gains = degrees - 2 * cross
            gains = degrees - 2 * cross
            
            # Find vertex with max gain
            best_v = np.argmax(gains)
            max_gain = gains[best_v]
            
            if max_gain <= 0:
                break
                
            # Flip best_v
            v = best_v
            labels[v] = 1 - labels[v]
            
            # Update cross for v and neighbors
            old_cross_v = cross[v]
            cross[v] = degrees[v] - old_cross_v
            
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    # Now same, were different -> cross[u] decreases
                    cross[u] -= 1
                else:
                    # Now different, were same -> cross[u] increases
                    cross[u] += 1
            
            current_cut = 0.5 * np.sum(cross)
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
        return best_labels, best_cut

    def run_simulated_annealing(labels, max_iters=50000, T_start=10.0, T_end=0.1, alpha=0.99995):
        labels = labels.copy()
        cross = init_cross(labels)
            
        current_cut = 0.5 * np.sum(cross)
        best_cut = current_cut
        best_labels = labels.copy()
        
        T = T_start
        for it in range(max_iters):
            if time_left() < 2.0:
                break
                
            v = random.randint(0, n-1)
            
            # Gain = degrees[v] - 2 * cross[v]
            gain = degrees[v] - 2 * cross[v]
            
            # delta cut = gain / 2
            delta = gain / 2.0
            
            if delta > 0 or (T > 0 and random.random() < np.exp(delta / T)):
                # Flip v
                labels[v] = 1 - labels[v]
                
                # Update cross
                old_cross_v = cross[v]
                cross[v] = degrees[v] - old_cross_v
                
                for u in neighbors[v]:
                    if labels[u] == labels[v]:
                        cross[u] -= 1
                    else:
                        cross[u] += 1
                        
                current_cut += delta
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                    
            T = max(T * alpha, T_end)
            
        return best_labels, best_cut

    # Strategy
    best_labels = None
    best_cut = -1
    
    # 1. Spectral Initialization
    # Compute eigenvector corresponding to largest eigenvalue
    # This provides a strong initial partition
    try:
        vals, vecs = np.linalg.eigh(adj_np)
        eigvec = vecs[:, -1]
        labels_spec = (eigvec > 0).astype(int)
    except:
        labels_spec = np.arange(n) % 2

    l, c = run_local_search(labels_spec)
    if c > best_cut:
        best_cut = c
        best_labels = l
        
    # 2. Alternating start (backup/diversity)
    labels_alt = np.arange(n) % 2
    l, c = run_local_search(labels_alt)
    if c > best_cut:
        best_cut = c
        best_labels = l

    # 3. Random starts with Local Search
    num_restarts = 200 # Increased due to faster init
    for _ in range(num_restarts):
        if time_left() < 2.0:
            break
        labels_rand = np.random.randint(0, 2, size=n)
        l, c = run_local_search(labels_rand, max_rounds=100)
        if c > best_cut:
            best_cut = c
            best_labels = l
            
    # 4. Simulated Annealing from best
    if time_left() > 10.0:
        l, c = run_simulated_annealing(best_labels)
        if c > best_cut:
            best_cut = c
            best_labels = l
            
    # 5. More SA restarts
    if time_left() > 5.0:
        for _ in range(20):
            if time_left() < 2.0:
                break
            labels_rand = np.random.randint(0, 2, size=n)
            l, c = run_simulated_annealing(labels_rand, max_iters=10000, T_start=5.0, T_end=0.1)
            if c > best_cut:
                best_cut = c
                best_labels = l
                
    # 6. Final polish
    if time_left() > 2.0:
        l, c = run_local_search(best_labels, max_rounds=500)
        if c > best_cut:
            best_cut = c
            best_labels = l
            
    return best_labels.astype(int)
