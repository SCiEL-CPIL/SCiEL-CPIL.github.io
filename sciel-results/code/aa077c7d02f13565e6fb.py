"""Kissing number in dimension 11 - Combined design with Spectral Initialization."""
import numpy as np
import itertools
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill

    # 1. Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)

    # 2. Build Conflict Graph
    # Conflict if dot > 2
    # We can exploit sparsity: only check pairs with support intersection >= 3
    # But for simplicity and given N=5280, dense dot product is feasible?
    # 5280 * 5280 * 11 flops ~ 300M. NumPy is fast, this takes < 1s.
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to sparse for Laplacian
    # We need the adjacency matrix A
    A = sp.csr_matrix(conflicts)
    
    # Degrees
    degrees = np.asarray(A.sum(axis=1)).flatten().astype(np.int64)
    
    # Laplacian L = D - A
    L = sp.diags(degrees) - A
    
    # 3. Spectral Embedding
    # Find smallest eigenvalues of L
    # We want a good partition. Try K=10 eigenvectors.
    try:
        # eigsh finds eigenvalues near sigma. By default, smallest magnitude.
        # For Laplacian, smallest are 0, then positive.
        # We want the first few non-zero ones.
        # Use k=10, sigma=0 to find smallest.
        # Note: L is symmetric.
        vals, vecs = spla.eigsh(L, k=10, sigma=0, which='LM') 
        # 'LM' with sigma=0 finds eigenvalues closest to 0.
        # We want the smallest positive ones.
        # The first one is 0 (constant vector).
        # We take the next 9 or 10.
        # Let's take the last 10 computed, which are the largest among the smallest?
        # eigsh with sigma=0 returns eigenvalues sorted by distance to 0.
        # So the last ones in the list are the largest magnitude, i.e., furthest from 0.
        # We want the smallest non-zero eigenvalues.
        # So we should take the first few non-zero ones.
        # Let's just take the last 10 of the returned vectors, assuming the 0 eigenvalue is first.
        # Actually, it's safer to compute a few more and pick.
        
        # Let's try to get the Fiedler vector and others.
        # For k=10, we get 10 eigenvalues.
        # The 0 eigenvalue is usually present.
        # We will use the embedding from these vectors.
        
        # Normalize rows of embedding matrix
        embedding = vecs.T # Shape (N4, 10)
        
        # K-Means clustering
        # We want to partition into clusters that are independent sets.
        # The chromatic number is likely around 20-30?
        # Let's try k_clusters = 20, 30, 40.
        
        best_init_set = None
        best_init_size = 0
        
        for k_clusters in [10, 20, 30]:
            # Simple K-Means implementation using scipy
            # scipy.cluster.vq.kmeans2
            from scipy.cluster.vq import kmeans2
            
            # Normalize embedding rows to unit length for spherical k-means behavior
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding_norm = embedding / norms
            
            # Run kmeans
            # num_iter is important for convergence
            centroids, labels = kmeans2(embedding_norm, k_clusters, minit='++', num_iter=10)
            
            # For each cluster, extract independent set
            for c in range(k_clusters):
                mask = (labels == c)
                cluster_indices = np.where(mask)[0]
                if len(cluster_indices) == 0:
                    continue
                
                # Greedy independent set on this subgraph
                # Since the cluster is "dense" in conflict graph? No, we want it to be sparse.
                # If spectral embedding works, nodes in same cluster should have similar connections.
                # If the graph is well-clustered into independent sets, then within a cluster, there should be few conflicts.
                # We greedily remove conflicts.
                
                in_set = np.zeros(N4, dtype=bool)
                # Order by degree in the cluster? Or just arbitrary?
                # Let's pick nodes with fewest conflicts within the cluster first.
                sub_conflicts = conflicts[np.ix_(cluster_indices, cluster_indices)]
                sub_degrees = sub_conflicts.sum(axis=1)
                
                # Sort indices by sub_degree
                order = np.argsort(sub_degrees)
                
                for idx in order:
                    orig_idx = cluster_indices[idx]
                    # Check if it conflicts with any already added in this cluster
                    # We can maintain a list of added nodes
                    # For efficiency, we can just check against the current in_set restricted to cluster
                    # But N4 is small enough.
                    if not in_set[orig_idx]:
                        # Check conflicts with current in_set
                        # conflicts[orig_idx] & in_set
                        if not np.any(conflicts[orig_idx] & in_set):
                            in_set[orig_idx] = True
                
                size = int(in_set.sum())
                if size > best_init_size:
                    best_init_size = size
                    best_init_set = in_set.copy()
                    
    except Exception as e:
        print(f"Spectral init failed: {e}")
        best_init_set = None

    # 4. Local Search Setup
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    rng = np.random.default_rng(42)

    def polish(in_set):
        """Greedy add every zero-conflict vertex repeatedly until fixed point."""
        in_set = in_set.copy()
        cc = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1

        changed = True
        while changed:
            changed = False
            cand = np.where((cc == 0) & (~in_set))[0]
            for v in cand:
                if cc[v] == 0 and not in_set[v]:
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    changed = True
        return in_set, int(in_set.sum())

    def sa_local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        cc = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1

        T = 3.0
        cooling = 0.9995
        min_T = 0.05

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        stuck = 0

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0:
                break

            r = rng.random()

            if r < 0.55 and len(not_in_set) > 0:
                # ADD: pick vertex with fewest in-set conflicts
                counts = cc[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = int(min_count)
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
                else:
                    accept = rng.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            cc[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    cc[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck = 0
                    else:
                        stuck += 1
                else:
                    stuck += 1

            elif r < 0.75 and len(in_set_indices) > 0:
                # REMOVE a random in-set vertex
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    cc[neighbors[v]] -= 1
                    stuck += 1
                else:
                    stuck += 1

            else:
                # SWAP: remove one in-set, add one not-in-set
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    if len(other_conflicts) == 0:
                        # Clean swap
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        cc[neighbors[v_remove]] -= 1
                        cc[neighbors[v_add]] += 1
                        stuck += 1
                    else:
                        to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                        delta = 1 - len(to_remove)
                        accept = delta > 0 or (rng.random() < np.exp(delta / T))
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                cc[neighbors[u]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            cc[neighbors[v_add]] += 1
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                                stuck = 0
                            else:
                                stuck += 1
                        else:
                            stuck += 1

            if stuck > 2000:
                T = max(T, 2.0)
                stuck = 0

            T *= cooling
            if T < min_T:
                T = min_T

        # Final polish
        best_set, best_size = polish(local_best_in_set)
        return best_set, best_size

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    # Strategy:
    # 1. If spectral init is good, use it.
    # 2. Also run greedy min-degree.
    # 3. Run SA on both.

    # Greedy Min-Degree Init
    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            if len(cands) > 1:
                scores = np.array([np.sum(neighbor_counts[neighbors[c]]) for c in cands])
                chosen = cands[int(np.argmin(scores))]
            else:
                chosen = cands[0]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    # Run SA from Greedy
    in_set_greedy = greedy_min_degree()
    if time.time() < deadline - 15:
        sa_time = min((deadline - 15 - time.time()) * 0.5, 15.0)
        set_g, size_g = sa_local_search(in_set_greedy, max_time=sa_time)
        if size_g > best_size:
            best_size = size_g
            best_in_set = set_g.copy()

    # Run SA from Spectral Init
    if best_init_set is not None and best_init_size > 0:
        if time.time() < deadline - 15:
            sa_time = min((deadline - 15 - time.time()) * 0.5, 15.0)
            set_s, size_s = sa_local_search(best_init_set, max_time=sa_time)
            if size_s > best_size:
                best_size = size_s
                best_in_set = set_s.copy()
    
    # Additional restarts from best
    if time.time() < deadline - 10:
        perturbed = best_in_set.copy()
        indices = np.where(perturbed)[0]
        n_rem = max(2, int(len(indices) * 0.1))
        if len(indices) > n_rem:
            to_remove = rng.choice(indices, size=n_rem, replace=False)
            perturbed[to_remove] = False
        
        sa_time = min(deadline - 10 - time.time(), 10.0)
        set_p, size_p = sa_local_search(perturbed, max_time=sa_time)
        if size_p > best_size:
            best_size = size_p
            best_in_set = set_p.copy()

    # Final result: selected W4 vectors + all W1 (norm-4) vectors
    # We can add 11 W1 vectors (one per axis, e.g., 2*e_i)
    # They don't conflict with W4.
    # Do they conflict with each other?
    # 2*e_i and 2*e_j (i!=j) -> dot 0. No conflict.
    # So we can add all 11.
    
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]

    v1_candidates = []
    for pos in range(D):
        v = [0] * D
        v[pos] = 2
        v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
