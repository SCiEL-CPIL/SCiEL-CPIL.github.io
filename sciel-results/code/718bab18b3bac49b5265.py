"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)
    N1 = len(vectors_v1)

    if N4 == 0:
        return vectors_v1

    # Compute conflict matrix for W4: conflict if dot > 2 (distance^2 < 4)
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def fast_greedy_add(in_set_init, max_time=5.0):
        """Fast polish: repeatedly add all vertices with zero conflicts."""
        polish_set = in_set_init.copy()
        polish_deadline = time.time() + max_time

        while time.time() < polish_deadline:
            not_in_mask = ~polish_set
            if not np.any(not_in_mask):
                break
            not_in_indices = np.where(not_in_mask)[0]
            counts = conflicts[not_in_indices] @ polish_set
            addable_mask = counts == 0
            if not np.any(addable_mask):
                break
            add_indices = not_in_indices[addable_mask]
            polish_set[add_indices] = True

        return polish_set, int(polish_set.sum())

    def local_search(in_set_init, max_time=10.0):
        """Simulated annealing local search with add/swap/remove moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 3.0
        cooling = 0.9999
        min_T = 0.005

        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        ls_deadline = time.time() + max_time
        no_improve = 0

        while time.time() < ls_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0:
                break

            move_type = rng.random()

            if move_type < 0.40:
                # ADD move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = min_count
                delta = 1 - k

                accept = False
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            elif move_type < 0.80 and len(in_set_indices) > 0:
                # 1-for-1 SWAP
                out_counts = count_conflicts[not_in_set]
                min_out = np.min(out_counts)
                min_out_cands = not_in_set[out_counts == min_out]
                v_add = min_out_cands[rng.integers(len(min_out_cands))]
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]

                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]

                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = False
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        accept = rng.random() < 0.4
                    else:
                        accept = rng.random() < np.exp(delta / T)

                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1

            else:
                # REMOVE move (perturbation)
                if len(in_set_indices) > 0:
                    v = in_set_indices[rng.integers(len(in_set_indices))]
                    if rng.random() < np.exp(-1.0 / T):
                        in_set[v] = False
                        current_size -= 1
                        count_conflicts[neighbors[v]] -= 1
                        no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            if no_improve > 2000:
                if len(in_set_indices) > 5:
                    n_perturb = max(4, len(in_set_indices) // 8)
                    perturb_victims = in_set_indices[rng.choice(len(in_set_indices), size=n_perturb, replace=False)]
                    in_set[perturb_victims] = False
                    current_size -= len(perturb_victims)
                    for u in perturb_victims:
                        count_conflicts[neighbors[u]] -= 1
                    for v in perturb_victims:
                        if not np.any(conflicts[v] & in_set):
                            in_set[v] = True
                            current_size += 1
                            count_conflicts[neighbors[v]] += 1
                T = max(T, 2.5)
                no_improve = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]

            if len(cands) > 1:
                best_score = float('inf')
                best_cand = cands[0]
                for c in cands:
                    nbrs = neighbors[c]
                    alive_nbrs = nbrs[alive[nbrs]]
                    if len(alive_nbrs) > 0:
                        score = np.sum(degrees[alive_nbrs])
                    else:
                        score = 0
                    if score < best_score:
                        best_score = score
                        best_cand = c
                chosen = best_cand
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

    # Spectral clustering to identify communities
    def spectral_init(n_clusters=8):
        """Use spectral clustering on the conflict graph to identify communities."""
        try:
            # Build sparse adjacency
            adj = csr_matrix(conflicts)
            # Compute Laplacian
            L = sparse_laplacian(adj)
            # Get smallest eigenvectors (excluding trivial ones)
            k = min(n_clusters, N4 - 1)
            try:
                eigenvalues, eigenvectors = eigsh(L, k=k+2, sigma=0, which='LM')
            except:
                eigenvalues, eigenvectors = eigsh(L, k=k+2, which='SM')
            
            # Use the first k non-trivial eigenvectors
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            eigenvectors = eigenvectors[:, idx[1:k+1]]  # skip trivial
            
            # Normalize rows
            norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedded = eigenvectors / norms
            
            # Simple k-means clustering
            # Initialize centroids randomly
            perm = rng.permutation(N4)
            centroids = embedded[perm[:k]].copy()
            
            labels = np.zeros(N4, dtype=np.int32)
            for _ in range(20):
                dists = np.sum((embedded[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
                new_labels = np.argmin(dists, axis=1)
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels
                for c in range(k):
                    mask = labels == c
                    if np.any(mask):
                        centroids[c] = embedded[mask].mean(axis=0)
            
            # For each cluster, run greedy MIS
            in_set = np.zeros(N4, dtype=bool)
            for c in range(k):
                cluster_nodes = np.where(labels == c)[0]
                if len(cluster_nodes) == 0:
                    continue
                # Greedy within cluster
                alive = np.zeros(N4, dtype=bool)
                alive[cluster_nodes] = True
                while np.any(alive):
                    alive_indices = np.where(alive)[0]
                    if len(alive_indices) == 0:
                        break
                    # Pick node with fewest alive neighbors in cluster
                    best_node = alive_indices[0]
                    best_count = float('inf')
                    for node in alive_indices:
                        nbrs = neighbors[node]
                        alive_nbrs = nbrs[alive[nbrs]]
                        count = len(alive_nbrs)
                        if count < best_count:
                            best_count = count
                            best_node = node
                    in_set[best_node] = True
                    nbrs = neighbors[best_node]
                    to_remove = np.unique(np.concatenate(([best_node], nbrs)))
                    alive[to_remove] = False
            
            return in_set
        except:
            return greedy_min_degree()

    # Multiple restarts with diverse initialization
    n_restarts = 0
    max_restarts = 200

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        if n_restarts % 5 == 1:
            # Strategy 1: Spectral clustering init
            in_set = spectral_init(n_clusters=8)
        elif n_restarts % 5 == 2:
            # Strategy 2: Greedy min-degree
            in_set = greedy_min_degree()
        elif n_restarts % 5 == 3:
            # Strategy 3: Random + greedy polish
            in_set = np.zeros(N4, dtype=bool)
            alive = np.ones(N4, dtype=bool)
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                chosen = alive_indices[rng.integers(len(alive_indices))]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
        elif n_restarts % 5 == 4:
            # Strategy 4: Weighted random based on inverse degree
            in_set = np.zeros(N4, dtype=bool)
            alive = np.ones(N4, dtype=bool)
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                weights = 1.0 / (neighbor_counts[alive_indices] + 1)
                weights /= weights.sum()
                chosen = alive_indices[rng.choice(len(alive_indices), p=weights)]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
        else:
            # Strategy 5: Spectral with different cluster count
            in_set = spectral_init(n_clusters=12)

        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # Fast greedy add polish
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break

        polish_time = min(remaining_time * 0.10, 2.0)
        if polish_time > 0.3:
            polish_result, polish_size = fast_greedy_add(in_set, max_time=polish_time)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
                current_size = polish_size
                in_set = polish_result

        # Local search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break

        ls_time = min(remaining_time * 0.70, 10.0)
        if ls_time < 1.5:
            ls_time = 1.5

        ls_result, ls_size = local_search(in_set, max_time=ls_time)

        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

        # Final polish
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            final_polish, final_size = fast_greedy_add(ls_result, max_time=min(remaining_time * 0.4, 4.0))
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_polish.copy()

        # Extra SA on best if time permits
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 4 and n_restarts >= 5:
            extra_ls, extra_size = local_search(best_in_set, max_time=min(remaining_time * 0.35, 6.0))
            if extra_size > best_size:
                best_size = extra_size
                best_in_set = extra_ls.copy()

    # Final result: W4 independent set + all W1 vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
