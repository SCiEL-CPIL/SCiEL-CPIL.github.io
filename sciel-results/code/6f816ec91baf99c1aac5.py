"""Maximum Cut via fast vectorized local search with batched flips, targeted 2-flip moves, SA escape, and deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.sparse.csgraph import laplacian as sp_laplacian
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses batched vectorized local search, targeted 2-flip moves to escape local optima,
    simulated annealing, multiple restarts with spectral k-means, and an explicit deadline guard.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Deadline: return best result before 140s (10s safety margin from 150s kill)
    deadline = time.time() + 140.0

    # Precompute degree for gain calculation
    degree = adj.sum(axis=1)
    total_edges = adj.sum() // 2
    # Precompute neighbor lists for O(d) updates
    neighbors_list = [np.where(adj[v] == 1)[0] for v in range(n)]

    def compute_cut(labels):
        """Number of edges crossing the partition. Fast vectorized version."""
        diff = (labels[None, :] != labels[:, None])
        cut2 = (adj * diff).sum()
        return cut2 // 2

    def batched_local_search(labels, max_rounds=200):
        """
        Batched greedy 1-flip local search: apply all non-conflicting positive-gain flips per round.
        Key insight: gain[v] = 2 * (num neighbors with same label as v) - degree[v]
        """
        current_cut = compute_cut(labels)
        
        # Precompute same_count for each vertex
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        gains = 2 * same_count - degree
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Find all vertices with positive gain
            positive = gains > 0
            if not np.any(positive):
                break
            
            # Select non-conflicting flips: no two selected vertices are adjacent
            pos_indices = np.where(positive)[0]
            selected = []
            blocked = set()
            # Sort by gain descending to pick best first
            order = pos_indices[np.argsort(-gains[pos_indices])]
            for v in order:
                if v in blocked:
                    continue
                selected.append(v)
                # Block all neighbors of v
                for nb in neighbors_list[v]:
                    blocked.add(nb)
            
            if not selected:
                break
            
            # Apply all selected flips
            for v in selected:
                old_label = labels[v]
                labels[v] = 1 - old_label
                current_cut += gains[v]
            
            # Update same_count and gains for all affected vertices
            affected = set(selected)
            for v in selected:
                for nb in neighbors_list[v]:
                    affected.add(nb)
            
            for v in affected:
                # Recompute same_count for v
                same_count[v] = (adj[v] * (labels == labels[v])).sum()
                gains[v] = 2 * same_count[v] - degree[v]
        
        return labels, current_cut

    def targeted_two_flip(labels, max_rounds=30):
        """
        Targeted 2-flip moves: for each vertex v with negative gain, try flipping v
        along with its best neighbor (most same-label neighbors).
        O(n * d) per round instead of O(n^2).
        """
        current_cut = compute_cut(labels)
        
        # Precompute same_count
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        gains = 2 * same_count - degree
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            
            # For vertices with negative gain, try 2-flips with neighbors
            neg_indices = np.where(gains < 0)[0]
            if len(neg_indices) == 0:
                break
            
            # Try up to 200 random negative-gain vertices
            try_count = min(len(neg_indices), 200)
            sampled = neg_indices[np.random.choice(len(neg_indices), try_count, replace=False)]
            
            for v in sampled:
                if time.time() > deadline:
                    break
                
                # Get neighbors sorted by how many same-label neighbors they have
                nbs = neighbors_list[v]
                if len(nbs) == 0:
                    continue
                
                # For each neighbor u of v, compute 2-flip gain:
                # Flipping v: gain_v = 2*same_count[v] - degree[v] (negative)
                # Flipping u: gain_u = 2*same_count[u] - degree[u]
                # Interaction: if v and u are adjacent, flipping both changes their
                # same/diff status twice, so interaction term = 2*adj[v,u]
                # Total 2-flip gain = gain_v + gain_u + 2*adj[v,u]
                # But we need to be careful: after flipping v, u's gain changes.
                # More precisely:
                # gain_2flip(v,u) = gain(v) + gain(u) + 2*adj[v,u]
                # where gain(v) is computed BEFORE any flips
                
                gain_v = gains[v]
                # For each neighbor u, the 2-flip gain is:
                # gain_v + (2*same_count[u] - degree[u]) + 2*adj[v,u]
                # adj[v,u] = 1 for neighbors
                # So: gain_v + 2*same_count[u] - degree[u] + 2
                
                cand_gains = gain_v + 2 * same_count[nbs] - degree[nbs] + 2
                best_idx = np.argmax(cand_gains)
                
                if cand_gains[best_idx] <= 0:
                    continue
                
                u = nbs[best_idx]
                old_lv = labels[v]
                old_lu = labels[u]
                
                labels[v] = 1 - old_lv
                labels[u] = 1 - old_lu
                
                new_cut = compute_cut(labels)
                actual_gain = new_cut - current_cut
                
                if actual_gain > 0:
                    current_cut = new_cut
                    improved = True
                    # Recompute same_count for affected vertices
                    affected = set([v, u])
                    for nb in neighbors_list[v]:
                        affected.add(nb)
                    for nb in neighbors_list[u]:
                        affected.add(nb)
                    for w in affected:
                        same_count[w] = (adj[w] * (labels == labels[w])).sum()
                        gains[w] = 2 * same_count[w] - degree[w]
                    break
                else:
                    labels[v] = old_lv
                    labels[u] = old_lu
            
            if not improved:
                break
        
        return labels, current_cut

    def simulated_annealing(labels, initial_temp=15.0, cooling=0.9995, max_iters=20000):
        """
        Simulated Annealing with 1-flip moves.
        Uses incremental gain computation for speed.
        """
        current_cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # Precompute same_count
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        
        temp = initial_temp
        no_improve_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random vertex
            v = np.random.randint(n)
            
            # Compute gain of flipping v
            same_count_v = same_count[v]
            gain = 2 * same_count_v - degree[v]
            
            # Accept if gain > 0 or with probability exp(gain / temp) for gain < 0
            accepted = False
            if gain >= 0:
                accepted = True
            else:
                # probability = exp(gain / temp), gain is negative
                if np.random.rand() < np.exp(gain / temp):
                    accepted = True
            
            if accepted:
                old_label = labels[v]
                labels[v] = 1 - old_label
                current_cut += gain
                
                # Update same_count for neighbors
                new_label = 1 - old_label
                nbs = neighbors_list[v]
                if len(nbs) > 0:
                    sign = 2 * new_label - 1
                    delta_same = (2 * labels[nbs] - 1) * sign
                    same_count[nbs] += delta_same
                
                # Update same_count for v
                same_count[v] = degree[v] - same_count[v]
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            temp *= cooling
            
            # Adaptive cooling: cool faster if stuck
            if no_improve_count > 500:
                temp *= 0.9
                no_improve_count = 0
        
        return best_labels, best_cut

    def greedy_init(adj, n, rng):
        """Greedy initialization: add vertices one by one to the side that maximizes cut."""
        labels = np.zeros(n, dtype=int)
        order = np.arange(n)
        rng.shuffle(order)
        
        for v in order:
            left_count = (adj[v] * (labels == 0)).sum()
            right_count = (adj[v] * (labels == 1)).sum()
            if right_count > left_count:
                labels[v] = 0
            else:
                labels[v] = 1
        return labels

    def spectral_kmeans_init(adj, n, rng, k=4):
        """
        Spectral initialization using k-means on multiple Laplacian eigenvectors.
        Produces structurally distinct partitions that seed different basins.
        """
        try:
            adj_sparse = csr_matrix(adj)
            L = sp_laplacian(adj_sparse, normed=False)
            
            # Get k+1 smallest eigenvectors (skip trivial constant)
            eig_k = min(k + 1, n - 1)
            eigvals, eigvecs = eigsh(L, k=eig_k, which='SM')
            idx = np.argsort(eigvals)
            # Use eigenvectors 1..k (skip index 0 which is trivial)
            X = eigvecs[:, idx[1:k+1]]
            
            # K-means on eigenvector coordinates
            centers = X[np.random.choice(n, k, replace=False), :]
            labels_km = np.zeros(n, dtype=int)
            
            for _ in range(10):  # 10 k-means iterations
                # Assign to nearest center
                dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                labels_km = np.argmin(dists, axis=1)
                
                # Update centers
                new_centers = np.zeros_like(centers)
                for c in range(k):
                    mask = labels_km == c
                    if mask.sum() > 0:
                        new_centers[c] = X[mask].mean(axis=0)
                    else:
                        new_centers[c] = centers[c]
                centers = new_centers
            
            # Convert k-means labels to 0/1: split into two groups
            # Use the two largest clusters as the two sides
            cluster_sizes = np.bincount(labels_km)
            top2 = np.argsort(-cluster_sizes)[:2]
            spectral = np.zeros(n, dtype=np.int8)
            spectral[labels_km == top2[1]] = 1
            return spectral
        except Exception:
            return None

    # Multiple restarts with different initializations
    best_labels_overall = None
    best_cut_overall = -1
    rng = np.random.RandomState(42)
    
    def try_restart(labels_init, tag=""):
        nonlocal best_labels_overall, best_cut_overall
        if time.time() > deadline:
            return
        
        labels = labels_init.copy()
        # Batched local search (faster convergence)
        labels, cut = batched_local_search(labels, max_rounds=300)
        
        # Try targeted 2-flip escape
        if time.time() < deadline - 5:
            labels2, cut2 = targeted_two_flip(labels, max_rounds=30)
            if cut2 > cut:
                labels, cut = labels2, cut2
                # Polish after 2-flip
                labels, cut = batched_local_search(labels, max_rounds=100)
        
        if cut > best_cut_overall:
            best_cut_overall = cut
            best_labels_overall = labels.copy()
    
    # Spectral k-means initialization (highest priority - uses global structure)
    spectral_labels = spectral_kmeans_init(adj, n, rng, k=4)
    
    # Restart 1: spectral k-means
    if spectral_labels is not None:
        try_restart(spectral_labels, "spectral_km")
    
    # Restart 2: Fiedler vector (single eigenvector)
    try:
        adj_sparse = csr_matrix(adj)
        L = sp_laplacian(adj_sparse, normed=False)
        eigvals, eigvecs = eigsh(L, k=2, which='SM')
        idx = np.argsort(eigvals)
        fiedler_vec = eigvecs[:, idx[1]]
        median_val = np.median(fiedler_vec)
        fiedler_labels = (fiedler_vec > median_val).astype(np.int8)
        try_restart(fiedler_labels, "fiedler")
    except Exception:
        pass
    
    # Restart 3: alternating
    try_restart(np.arange(n) % 2, "alt")
    
    # Restart 4: greedy
    try_restart(greedy_init(adj, n, rng), "greedy")
    
    # Restart 5: random
    try_restart(rng.randint(0, 2, size=n), "rand")
    
    # Restart 6: random with bias
    try_restart(rng.randint(0, 2, size=n), "rand2")
    
    # Restart 7: all zeros then flip random
    labels_init = np.zeros(n, dtype=np.int8)
    flip_mask = rng.rand(n) < 0.5
    labels_init[flip_mask] = 1
    try_restart(labels_init, "half")
    
    # Restart 8-12: more random starts
    for _ in range(5):
        if time.time() > deadline:
            break
        try_restart(rng.randint(0, 2, size=n), "rand_extra")
    
    # If we have time, do SA from best solution
    if best_labels_overall is not None and time.time() < deadline - 10:
        labels_sa = best_labels_overall.copy()
        labels_sa, cut_sa = simulated_annealing(labels_sa, initial_temp=20.0, cooling=0.999, max_iters=30000)
        
        # Polish
        labels_sa, cut_sa = batched_local_search(labels_sa, max_rounds=200)
        
        if cut_sa > best_cut_overall:
            best_cut_overall = cut_sa
            best_labels_overall = labels_sa.copy()
    
    # Final polish
    if best_labels_overall is not None and time.time() < deadline - 2:
        labels_final = best_labels_overall.copy()
        labels_final, cut_final = batched_local_search(labels_final, max_rounds=300)
        if cut_final > best_cut_overall:
            best_cut_overall = cut_final
            best_labels_overall = labels_final.copy()
    
    return best_labels_overall
# EVOLVE-BLOCK-END
