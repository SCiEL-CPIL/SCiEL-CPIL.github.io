"""Maximum Cut via GW-style spectral embedding + fast 1-flip + optimized 2-swap local search."""
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

    def tabu_search(labels, max_iter=5000):
        """Cerny tabu search: accepts worsening moves with tabu list to escape local optima."""
        labels = labels.copy()
        # Precompute same_count
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1
        
        # Tabu list: array for speed
        tabu_until = np.zeros(n, dtype=np.int32)
        tabu_len = 20
        current_cut = compute_cut(labels)
        best_cut_ts = current_cut
        best_labels_ts = labels.copy()
        iteration = 0
        
        # Precompute degree array for speed
        deg_arr = degrees
        
        while iteration < max_iter:
            if time.time() > deadline:
                break
            
            # Find best move (including worsening, respecting tabu)
            best_gain = -999999
            best_v = -1
            
            # Vectorized gain computation
            gains = 2 * same_count - deg_arr
            # Mask tabu vertices
            tabu_mask = tabu_until > iteration
            gains[tabu_mask] = -999999
            
            # Aspiration: if a tabu move gives best overall, allow it
            # Find best non-tabu gain
            non_tabu_gains = gains.copy()
            non_tabu_gains[tabu_mask] = -999999
            best_non_tabu_gain = np.max(non_tabu_gains)
            best_overall_gain = np.max(gains)
            
            if best_overall_gain > best_non_tabu_gain and best_overall_gain > -999999:
                # Aspiration: use the tabu move that gives best overall
                best_v = int(np.argmax(gains))
                best_gain = best_overall_gain
            else:
                best_v = int(np.argmax(non_tabu_gains))
                best_gain = best_non_tabu_gain
            
            if best_gain <= -999998:
                break
            
            # Perform the flip
            v = best_v
            old_label = labels[v]
            labels[v] = 1 - old_label
            tabu_until[v] = iteration + tabu_len
            
            # Update same_count
            old_sc_v = same_count[v]
            same_count[v] = deg_arr[v] - old_sc_v
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[u] += 1
                else:
                    same_count[u] -= 1
            
            # Update cut
            current_cut += best_gain
            iteration += 1
            
            if current_cut > best_cut_ts:
                best_cut_ts = current_cut
                best_labels_ts = labels.copy()
        
        return best_labels_ts

    def two_swap_search(labels, max_passes=20):
        """Optimized 2-swap local search with vectorized gain computation."""
        labels = labels.copy()
        
        # Precompute: for each vertex v, count neighbors in same partition
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(n))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    break
                li = labels[i]
                
                # Vectorized: find all j with different label
                diff_mask = labels != li
                diff_mask[i] = False
                candidates = np.where(diff_mask)[0]
                
                if len(candidates) == 0:
                    continue
                
                # Compute delta_same for all candidates at once
                # delta_same = degrees[i] + degrees[j] - 2*same_count[i] - 2*same_count[j] - 2*adj[i,j]
                e_ij = adj[i, candidates]
                delta_same = (degrees[i] + degrees[candidates] - 2*same_count[i] - 2*same_count[candidates] - 2*e_ij)
                
                # Find best candidate
                best_idx = int(np.argmax(delta_same))
                if delta_same[best_idx] <= 0:
                    continue
                
                j = candidates[best_idx]
                lj = labels[j]
                e_ij_val = adj[i, j]
                
                # Perform swap
                labels[i] = lj
                labels[j] = li
                improved = True
                
                # Update same_count for i and j and their neighbors
                old_same_i = same_count[i]
                old_same_j = same_count[j]
                same_count[i] = degrees[i] - old_same_i - e_ij_val
                same_count[j] = degrees[j] - old_same_j - e_ij_val
                
                # Update neighbors of i
                for u in neighbors[i]:
                    if u == j:
                        continue
                    if labels[u] == labels[i]:  # labels[i] is now lj
                        same_count[u] += 1
                    else:
                        same_count[u] -= 1
                
                # Update neighbors of j
                for u in neighbors[j]:
                    if u == i:
                        continue
                    if labels[u] == labels[j]:  # labels[j] is now li
                        same_count[u] += 1
                    else:
                        same_count[u] -= 1
            
            if not improved:
                break
        
        return labels

    def gw_spectral_partition():
        """Goemans-Williamson style: full spectral embedding with random hyperplane."""
        try:
            D = np.diag(degrees)
            L = D - adj
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            
            # Use all eigenvectors except the first (zero eigenvalue)
            # For GW, we want a rank-(n-1) embedding
            k = n - 1
            embedding = eigenvectors[:, 1:k+1]
            
            # Normalize rows to unit vectors (GW constraint)
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms
            
            # Random hyperplane
            random_vec = np.random.randn(k)
            random_vec /= np.linalg.norm(random_vec)
            projection = embedding @ random_vec
            labels = (projection > 0).astype(np.int8)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n, dtype=np.int8)

    def gw_spectral_partition_low_rank(k=16):
        """GW-style with reduced rank for speed, multiple random hyperplanes."""
        try:
            D = np.diag(degrees)
            L = D - adj
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            
            k = min(k, n - 1)
            embedding = eigenvectors[:, 1:k+1]
            
            # Normalize rows to unit vectors
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms
            
            # Random hyperplane
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
    
    # 1. Full GW spectral (rank n-1)
    if time.time() < deadline:
        init_labels_list.append(gw_spectral_partition())
    
    # 2. GW spectral with reduced rank, multiple random hyperplanes
    for _ in range(8):
        if time.time() > deadline:
            break
        init_labels_list.append(gw_spectral_partition_low_rank(k=16))
    
    # 3. Fiedler vector (simplest spectral)
    try:
        D = np.diag(degrees)
        L = D - adj
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        init_labels_list.append((fiedler > 0).astype(np.int8))
    except Exception:
        pass
    
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
        
        # 2-swap after 1-flip
        if time.time() < deadline:
            labels2 = two_swap_search(labels1, max_passes=10)
            cut2 = compute_cut(labels2)
            if cut2 > best_cut:
                best_cut = cut2
                best_labels = labels2.copy()
        
        # Tabu search after 2-swap to escape local optima
        if time.time() < deadline:
            labels3 = tabu_search(labels2, max_iter=3000)
            cut3 = compute_cut(labels3)
            if cut3 > best_cut:
                best_cut = cut3
                best_labels = labels3.copy()

    # Additional random restarts with spectral bias
    num_random = 20
    for r in range(num_random):
        if time.time() > deadline:
            break
        
        # Mix: 50% pure random, 50% spectral-based
        if r % 2 == 0:
            labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        else:
            labels = gw_spectral_partition_low_rank(k=8)
        
        # 1-flip greedy
        labels = flip_greedy(labels, max_passes=30)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # 2-swap
        if time.time() < deadline:
            labels_sw = two_swap_search(labels, max_passes=8)
            cut_sw = compute_cut(labels_sw)
            if cut_sw > best_cut:
                best_cut = cut_sw
                best_labels = labels_sw.copy()
        
        # Tabu search on best from this restart
        if time.time() < deadline:
            start_tabu = labels_sw if cut_sw > cut else labels
            labels_tab = tabu_search(start_tabu, max_iter=2000)
            cut_tab = compute_cut(labels_tab)
            if cut_tab > best_cut:
                best_cut = cut_tab
                best_labels = labels_tab.copy()

    # Final high-precision polish pass on the best solution
    if best_labels is not None and time.time() < deadline - 5.0:
        # More aggressive 2-swap with more passes
        polished = two_swap_search(best_labels, max_passes=15)
        cut_polished = compute_cut(polished)
        if cut_polished > best_cut:
            best_cut = cut_polished
            best_labels = polished.copy()
        
        # Final tabu polish
        if time.time() < deadline - 2.0:
            final_tabu = tabu_search(best_labels, max_iter=5000)
            cut_final = compute_cut(final_tabu)
            if cut_final > best_cut:
                best_cut = cut_final
                best_labels = final_tabu.copy()
        
        # One more 2-swap after final tabu
        if time.time() < deadline - 1.0:
            final_swap = two_swap_search(best_labels, max_passes=10)
            cut_final2 = compute_cut(final_swap)
            if cut_final2 > best_cut:
                best_cut = cut_final2
                best_labels = final_swap.copy()

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
