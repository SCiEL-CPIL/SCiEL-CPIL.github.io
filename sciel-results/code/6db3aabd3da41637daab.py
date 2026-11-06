"""Maximum Cut via Vectorized 2-Flip Neighborhood, Spectral Init, and Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def partition(adjacency):
    """
    Computes a maximum cut partition for the given adjacency matrix.
    Uses spectral initialization, vectorized 2-flip neighborhood search,
    and simulated annealing for exploration.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n == 0:
        return np.array([], dtype=int)
    
    # Precompute neighbor lists
    neighbors = [list(np.nonzero(adj[i])[0]) for i in range(n)]
    degrees = np.array([len(neighbors[i]) for i in range(n)])
    
    # Helper to calculate cut size
    def cut_size(labels):
        labels = np.asarray(labels)
        total = 0
        for i in range(n):
            for j in neighbors[i]:
                if i < j and labels[i] != labels[j]:
                    total += 1
        return total
    
    # Compute 1-flip gain for vertex v given current labels
    def flip1_gain(v, labels):
        lv = labels[v]
        gain = 0
        for u in neighbors[v]:
            if labels[u] == lv:
                gain += 1
            else:
                gain -= 1
        return gain
    
    # Vectorized 2-flip gain computation for all pairs
    # Returns a matrix of 2-flip gains for all pairs (i, j)
    def compute_all_2flip_gains(labels, g1):
        """
        For each pair (i, j), the 2-flip gain is:
        g2[i,j] = g1[i] + g1[j] - 2 * adj[i,j] * sign(i,j)
        where sign = +1 if labels[i] == labels[j], -1 otherwise
        """
        # sign matrix: +1 if same label, -1 if different
        labels_col = labels[:, np.newaxis]
        labels_row = labels[np.newaxis, :]
        same = (labels_col == labels_row).astype(np.int8)
        sign = 2 * same - 1  # +1 if same, -1 if different
        
        # g1 outer sum: g1[i] + g1[j]
        g1_sum = g1[:, np.newaxis] + g1[np.newaxis, :]
        
        # 2 * adj * sign
        adj_float = adj.astype(np.float64)
        correction = 2.0 * adj_float * sign
        
        g2 = g1_sum - correction
        
        # Zero out diagonal
        np.fill_diagonal(g2, 0)
        
        return g2
    
    # Fast local search using vectorized 2-flip neighborhood
    def local_search(labels, time_budget=10.0):
        labels = labels.copy()
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        
        # Precompute 1-flip gains
        g1 = np.array([flip1_gain(v, labels) for v in range(n)], dtype=np.float64)
        
        # Compute all 2-flip gains
        g2 = compute_all_2flip_gains(labels, g1)
        
        local_start = time.time()
        max_iter = 50000
        iter_count = 0
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() - local_start < time_budget and iter_count < max_iter:
            iter_count += 1
            
            # Find best 1-flip
            best_1flip_v = -1
            best_1flip_gain = 0
            for v in range(n):
                if g1[v] > best_1flip_gain:
                    best_1flip_gain = g1[v]
                    best_1flip_v = v
            
            # Find best 2-flip
            best_2flip_u = -1
            best_2flip_v = -1
            best_2flip_gain = 0
            # Use vectorized search: find max in upper triangle
            np.fill_diagonal(g2, -np.inf)
            idx = np.unravel_index(np.argmax(g2), g2.shape)
            if g2[idx] > best_2flip_gain:
                best_2flip_gain = g2[idx]
                best_2flip_u = idx[0]
                best_2flip_v = idx[1]
                if best_2flip_u > best_2flip_v:
                    best_2flip_u, best_2flip_v = best_2flip_v, best_2flip_u
                np.fill_diagonal(g2, 0)
            
            # Choose the best move
            if best_1flip_gain >= best_2flip_gain and best_1flip_v >= 0:
                # Apply 1-flip
                v = best_1flip_v
                labels[v] = 1 - labels[v]
                current_gain = best_1flip_gain
                labels[v] = labels[v]  # already flipped
                
                # Update g1
                old_g1_v = g1[v]
                g1[v] = -old_g1_v
                for u in neighbors[v]:
                    if labels[u] != labels[v]:
                        g1[u] += 2
                    else:
                        g1[u] -= 2
                
                # Recompute g2
                g2 = compute_all_2flip_gains(labels, g1)
                
            elif best_2flip_gain > 0 and best_2flip_u >= 0:
                # Apply 2-flip
                u, v = best_2flip_u, best_2flip_v
                labels[u] = 1 - labels[u]
                labels[v] = 1 - labels[v]
                current_gain = best_2flip_gain
                
                # Recompute g1 for affected vertices
                affected = set([u, v])
                for x in [u, v]:
                    for y in neighbors[x]:
                        affected.add(y)
                for x in affected:
                    g1[x] = flip1_gain(x, labels)
                
                # Recompute g2
                g2 = compute_all_2flip_gains(labels, g1)
                
            else:
                # No improving move found
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                continue
            
            no_improve_count = 0
            best_cut += current_gain
            
            if best_cut > cut_size(best_labels):
                best_labels = labels.copy()
        
        # Final polish
        final_cut = cut_size(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        
        return best_labels, best_cut
    
    # Spectral initialization using Fiedler vector
    def spectral_init():
        deg = np.sum(adj, axis=1).astype(float)
        D = np.diag(deg)
        L = D - adj.astype(float)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler = eigenvectors[:, 1]
            labels = (fiedler >= 0).astype(int)
            if np.all(labels == labels[0]):
                labels = np.random.randint(0, 2, n)
            return labels
        except Exception:
            return np.random.randint(0, 2, n)
    
    # GW-style rounding with random hyperplane in eigenspace
    def gw_init():
        deg = np.sum(adj, axis=1).astype(float)
        D = np.diag(deg)
        L = D - adj.astype(float)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            k = min(5, n - 1)
            if k < 1:
                return np.random.randint(0, 2, n)
            
            weights = np.random.randn(k)
            weights /= np.linalg.norm(weights)
            
            v = np.zeros(n)
            for i in range(k):
                v += weights[i] * eigenvectors[:, 1 + i]
            
            labels = (v >= 0).astype(int)
            if np.all(labels == labels[0]):
                labels = np.random.randint(0, 2, n)
            return labels
        except Exception:
            return np.random.randint(0, 2, n)
    
    # 1. Multiple initialization strategies
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral (Fiedler)
    labels_spectral = spectral_init()
    ls_labels, ls_cut = local_search(labels_spectral, time_budget=10.0)
    if ls_cut > best_cut:
        best_cut = ls_cut
        best_labels = ls_labels.copy()
    
    # Strategy 2: GW rounding
    if time.time() < deadline - 30.0:
        labels_gw = gw_init()
        ls_labels, ls_cut = local_search(labels_gw, time_budget=10.0)
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels.copy()
    
    # Strategy 3: Random restarts
    if time.time() < deadline - 25.0:
        for _ in range(6):
            if time.time() >= deadline - 20.0:
                break
            labels_rand = np.random.randint(0, 2, n)
            ls_labels, ls_cut = local_search(labels_rand, time_budget=8.0)
            if ls_cut > best_cut:
                best_cut = ls_cut
                best_labels = ls_labels.copy()
    
    # 2. Simulated Annealing with 1-flip and 2-flip moves
    if time.time() < deadline - 15.0:
        current_labels = best_labels.copy()
        current_cut = best_cut
        g1 = np.array([flip1_gain(v, current_labels) for v in range(n)], dtype=np.float64)
        
        T_start = 15.0
        T_end = 0.1
        cooling_rate = 0.985
        T = T_start
        sa_deadline = deadline - 8.0
        
        while T > T_end and time.time() < sa_deadline:
            # Randomly choose between 1-flip and 2-flip
            if np.random.rand() < 0.6:
                # 1-flip
                v = np.random.randint(0, n)
                gain = g1[v]
                
                if gain > 0 or (T > 0 and np.random.rand() < math.exp(-gain / T)):
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gain
                    g1[v] = -gain
                    for u in neighbors[v]:
                        if current_labels[u] != current_labels[v]:
                            g1[u] += 2
                        else:
                            g1[u] -= 2
                    
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = current_labels.copy()
            else:
                # 2-flip
                u = np.random.randint(0, n)
                v = np.random.randint(0, n)
                if u == v:
                    continue
                lu = current_labels[u]
                lv = current_labels[v]
                auv = adj[u, v]
                if lu == lv:
                    sign = 1
                else:
                    sign = -1
                gain = g1[u] + g1[v] - 2 * auv * sign
                
                if gain > 0 or (T > 0 and np.random.rand() < math.exp(-gain / T)):
                    current_labels[u] = 1 - current_labels[u]
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gain
                    
                    # Recompute g1 for affected vertices
                    affected = set([u, v])
                    for x in [u, v]:
                        for y in neighbors[x]:
                            affected.add(y)
                    for x in affected:
                        g1[x] = flip1_gain(x, current_labels)
                    
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = current_labels.copy()
            
            T *= cooling_rate
        
        # Final local search polish
        if time.time() < deadline - 5.0:
            ls_labels, ls_cut = local_search(best_labels, time_budget=5.0)
            if ls_cut > best_cut:
                best_cut = ls_cut
                best_labels = ls_labels.copy()
    
    # Final deadline check
    if best_labels is None:
        best_labels = np.random.randint(0, 2, n)
    
    return best_labels

# EVOLVE-BLOCK-END
