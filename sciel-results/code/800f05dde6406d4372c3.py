"""Maximum Cut via Fiedler vector bisection + aggressive 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    if n == 0:
        return np.array([], dtype=np.int8)

    adj = np.array(adjacency, dtype=np.int32)
    total_edges = int(np.sum(adj)) // 2

    # Precompute neighbor lists and degrees
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def flip_greedy(labels, max_passes=50):
        """Fast 1-flip greedy using incremental cut tracking."""
        labels = labels.copy()
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                deg = degrees[v]
                gain = 2 * same_count[v] - deg
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    changed = True
                    new_same_v = deg - same_count[v]
                    same_count[v] = new_same_v
                    for u in neighbors[v]:
                        if labels[u] == labels[v]:
                            same_count[u] += 1
                        else:
                            same_count[u] -= 1
            if not changed:
                break
        return labels

    def two_swap_search(labels, max_passes=30):
        """2-swap local search with efficient gain computation."""
        labels = labels.copy()
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(n))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    if labels[i] == labels[j]:
                        continue
                    
                    li = labels[i]
                    lj = labels[j]
                    
                    n_ii = 0
                    n_ij = 0
                    for u in neighbors[i]:
                        if labels[u] == li:
                            n_ii += 1
                        else:
                            n_ij += 1
                    
                    n_ji = 0
                    n_jj = 0
                    for u in neighbors[j]:
                        if labels[u] == li:
                            n_ji += 1
                        else:
                            n_jj += 1
                    
                    e_ij = adj[i, j]
                    gain = n_ii + n_jj - n_ij - n_ji + 2 * e_ij
                    
                    if gain > 0:
                        labels[i] = lj
                        labels[j] = li
                        improved = True
            
            if not improved:
                break
        
        return labels

    def tabu_search(labels, tabu_len=20, max_iter=2000):
        """Cerny-style tabu search to escape local optima."""
        labels = labels.copy()
        best_labels = labels.copy()
        best_cut = compute_cut(labels)
        cur_cut = best_cut
        
        # Precompute same_count for incremental updates
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1
        
        tabu_list = [0] * n  # iteration when vertex became tabu
        current_iter = 0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            current_iter = it
            
            best_gain = -10**9
            best_v = -1
            best_u = -1
            best_is_tabu = False
            
            # Find best move (1-flip or 2-swap)
            # Try 1-flips first
            for v in range(n):
                if time.time() > deadline:
                    break
                deg = degrees[v]
                gain = 2 * same_count[v] - deg
                is_tabu = (current_iter - tabu_list[v]) < tabu_len
                
                if not is_tabu or gain > best_gain:
                    if gain > best_gain or (gain == best_gain and not is_tabu and best_is_tabu):
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                            best_u = -1
                            best_is_tabu = is_tabu
            
            # If no good 1-flip, try 2-swaps
            if best_gain <= 0:
                for i in range(n):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n):
                        if time.time() > deadline:
                            break
                        if labels[i] == labels[j]:
                            continue
                        
                        li = labels[i]
                        lj = labels[j]
                        
                        n_ii = 0
                        n_ij = 0
                        for u in neighbors[i]:
                            if labels[u] == li:
                                n_ii += 1
                            else:
                                n_ij += 1
                        
                        n_ji = 0
                        n_jj = 0
                        for u in neighbors[j]:
                            if labels[u] == li:
                                n_ji += 1
                            else:
                                n_jj += 1
                        
                        e_ij = adj[i, j]
                        gain = n_ii + n_jj - n_ij - n_ji + 2 * e_ij
                        
                        i_tabu = (current_iter - tabu_list[i]) < tabu_len
                        j_tabu = (current_iter - tabu_list[j]) < tabu_len
                        is_tabu = i_tabu or j_tab
                        
                        if gain > best_gain or (gain == best_gain and not is_tabu and best_is_tabu):
                            if gain > best_gain:
                                best_gain = gain
                                best_v = i
                                best_u = j
                                best_is_tabu = is_tabu
            
            if best_v < 0:
                break
            
            # Apply best move
            if best_u == -1:
                # 1-flip
                v = best_v
                old_label = labels[v]
                labels[v] = 1 - labels[v]
                tabu_list[v] = current_iter
                
                # Update same_count
                new_same_v = degrees[v] - same_count[v]
                for u in neighbors[v]:
                    if labels[u] == labels[v]:
                        same_count[u] += 1
                    else:
                        same_count[u] -= 1
                same_count[v] = new_same_v
            else:
                # 2-swap
                i, j = best_v, best_u
                li = labels[i]
                lj = labels[j]
                labels[i] = lj
                labels[j] = li
                tabu_list[i] = current_iter
                tabu_list[j] = current_iter
                
                # Update same_count for i and j and their neighbors
                # Recompute for simplicity (n is small)
                for u in neighbors[i]:
                    if labels[u] == labels[i]:
                        same_count[u] += 1
                    else:
                        same_count[u] -= 1
                for u in neighbors[j]:
                    if labels[u] == labels[j]:
                        same_count[u] += 1
                    else:
                        same_count[u] -= 1
                # Fix double-counting for i and j themselves
                same_count[i] = 0
                for u in neighbors[i]:
                    if labels[u] == labels[i]:
                        same_count[i] += 1
                same_count[j] = 0
                for u in neighbors[j]:
                    if labels[u] == labels[j]:
                        same_count[j] += 1
            
            # Update current cut
            cur_cut += best_gain
            if cur_cut > best_cut:
                best_cut = cur_cut
                best_labels = labels.copy()
        
        return best_labels

    def fiedler_vector_partition():
        """Create initial partition using Fiedler vector of graph Laplacian."""
        D = np.diag(degrees)
        L = D - adj
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler_idx = 1
            fiedler = eigenvectors[:, fiedler_idx]
            labels = (fiedler > 0).astype(np.int8)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n, dtype=np.int8)

    def spectral_embedding_partition():
        """Create initial partition using spectral embedding with multiple eigenvectors."""
        try:
            D = np.diag(degrees)
            L = D - adj
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            
            k = min(8, n - 1)
            embedding = eigenvectors[:, 1:k+1]
            
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms
            
            random_vec = np.random.randn(k)
            random_vec /= np.linalg.norm(random_vec)
            projection = embedding @ random_vec
            labels = (projection > 0).astype(np.int8)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n, dtype=np.int8)

    best_cut = -1
    best_labels = None

    # Generate multiple initial partitions
    init_labels_list = []
    
    # 1. Fiedler vector partition (primary)
    init_labels_list.append(fiedler_vector_partition())
    
    # 2. Spectral embedding with random hyperplane
    init_labels_list.append(spectral_embedding_partition())
    
    # 3. Additional spectral embedding variations
    for _ in range(5):
        if time.time() > deadline:
            break
        init_labels_list.append(spectral_embedding_partition())
    
    # 4. Alternating pattern
    init_labels_list.append(np.arange(n) % 2)
    
    # 5. Random
    init_labels_list.append(np.random.randint(0, 2, size=n, dtype=np.int8))

    # Process each initial label assignment
    for init_labels in init_labels_list:
        if time.time() > deadline:
            break
        
        # 1-flip greedy
        labels1 = flip_greedy(init_labels, max_passes=40)
        cut1 = compute_cut(labels1)
        if cut1 > best_cut:
            best_cut = cut1
            best_labels = labels1.copy()
        
        # Tabu search to escape local optima
        if time.time() < deadline:
            labels_tabu = tabu_search(labels1, tabu_len=15, max_iter=1000)
            cut_tabu = compute_cut(labels_tabu)
            if cut_tabu > best_cut:
                best_cut = cut_tabu
                best_labels = labels_tabu.copy()
        
        # 2-swap after tabu
        if time.time() < deadline:
            labels2 = two_swap_search(labels_tabu, max_passes=10)
            cut2 = compute_cut(labels2)
            if cut2 > best_cut:
                best_cut = cut2
                best_labels = labels2.copy()

    # Additional random restarts with tabu
    num_random = 20
    for r in range(num_random):
        if time.time() > deadline:
            break
        
        labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        
        # 1-flip greedy
        labels = flip_greedy(labels, max_passes=20)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Tabu search
        if time.time() < deadline:
            labels_tabu = tabu_search(labels, tabu_len=15, max_iter=500)
            cut_tabu = compute_cut(labels_tabu)
            if cut_tabu > best_cut:
                best_cut = cut_tabu
                best_labels = labels_tabu.copy()
        
        # 2-swap
        if time.time() < deadline:
            labels_sw = two_swap_search(labels_tabu, max_passes=5)
            cut_sw = compute_cut(labels_sw)
            if cut_sw > best_cut:
                best_cut = cut_sw
                best_labels = labels_sw.copy()

    # Final polish: more aggressive 2-swap
    if best_labels is not None and time.time() < deadline:
        final_labels = two_swap_search(best_labels, max_passes=20)
        final_cut = compute_cut(final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
