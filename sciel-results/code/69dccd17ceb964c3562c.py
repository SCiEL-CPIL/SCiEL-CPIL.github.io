import numpy as np
import time
import random

def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # Ensure symmetry and no self-loops
    adj = (adj + adj.T) // 2
    np.fill_diagonal(adj, 0)

    start_time = time.time()
    DEADLINE = start_time + 145.0

    def time_left():
        return DEADLINE - time.time()

    degrees = adj.sum(axis=1).astype(np.int32)
    E = int(degrees.sum()) // 2
    adj_float = adj.astype(np.float64)
    degrees_float = degrees.astype(np.float64)

    def cut_size(labels):
        """Compute cut size: number of edges with endpoints in different sets."""
        # cut = sum over edges (u,v) where labels[u] != labels[v]
        # = 0.5 * sum_{u,v} adj[u,v] * (labels[u] != labels[v])
        diff = (labels[:, None] != labels[None, :]).astype(np.float64)
        return int(np.sum(adj_float * diff)) // 2

    def compute_gains_vectorized(labels):
        """Gain for flipping vertex v: 2*(same-side neighbors) - degree[v].
        If gain > 0, flipping v increases cut by gain/2."""
        l0 = (labels == 0).astype(np.float64)
        l1 = (labels == 1).astype(np.float64)
        adj_l0 = adj_float @ l0
        adj_l1 = adj_float @ l1
        same = np.where(labels == 0, adj_l0, adj_l1)
        return 2.0 * same - degrees_float

    def local_search(labels, max_rounds=200, use_pairwise=True):
        """Hill-climbing with single and pairwise flips."""
        labels = labels.copy().astype(np.int8)
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        current_cut = best_cut

        for rnd in range(max_rounds):
            if time_left() < 1.5:
                break

            improved = False

            # Single flips
            gains = compute_gains_vectorized(labels)
            order = np.argsort(-gains)
            for v in order:
                if time_left() < 1.0:
                    break
                if gains[v] > 1e-9:
                    labels[v] = 1 - labels[v]
                    current_cut += int(gains[v]) // 2
                    improved = True
                    gains = compute_gains_vectorized(labels)

            # Pairwise flips if no single improvement
            if not improved and use_pairwise:
                for _ in range(20):
                    if time_left() < 1.0:
                        break
                    v1 = np.random.randint(0, n)
                    v2 = np.random.randint(0, n)
                    if v1 == v2:
                        continue
                    labels[v1] = 1 - labels[v1]
                    labels[v2] = 1 - labels[v2]
                    new_cut = cut_size(labels)
                    if new_cut > current_cut:
                        current_cut = new_cut
                        improved = True
                    else:
                        labels[v1] = 1 - labels[v1]
                        labels[v2] = 1 - labels[v2]
                if not improved:
                    break

            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

        return best_labels, best_cut

    def simulated_annealing(labels, max_iters=50000, T_start=10.0, T_end=0.01, alpha=None):
        """Simulated annealing with single flips."""
        if alpha is None:
            alpha = (T_end / T_start) ** (1.0 / max(1, max_iters - 1))
        labels = labels.copy().astype(np.int8)
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        current_cut = best_cut
        T = T_start

        for it in range(max_iters):
            if time_left() < 2.0:
                break

            v = np.random.randint(0, n)
            gains = compute_gains_vectorized(labels)
            delta = gains[v] / 2.0

            if delta >= 0 or np.random.random() < np.exp(delta / max(T, 1e-10)):
                labels[v] = 1 - labels[v]
                current_cut += int(delta)
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()

            T = max(T * alpha, T_end)

        return best_labels, best_cut

    def goemans_williamson_rounding(vec, n_samples=100):
        """Randomized rounding using random hyperplanes through origin.
        For each sample, pick a random direction u, label[i] = 0 if vec[i].u >= 0 else 1."""
        vec = vec.astype(np.float64)
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-10:
            return np.random.randint(0, 2, size=n).astype(np.int8)
        vec = vec / vec_norm

        best_labels = None
        best_cut = -1
        for _ in range(n_samples):
            u = np.random.randn(n)
            u /= np.linalg.norm(u)
            proj = vec @ u
            labels = (proj >= 0).astype(np.int8)
            c = cut_size(labels)
            if c > best_cut:
                best_cut = c
                best_labels = labels
        return best_labels, best_cut

    # Get spectral vector
    def get_spectral_vector():
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            L = np.diag(degrees_float) - adj_float
            L_sparse = csr_matrix(L)
            vals, vecs = eigsh(L_sparse, k=3, which='SM')
            idx = np.argsort(vals)
            # Use second smallest (Fiedler vector)
            vec = vecs[:, idx[1]]
            return vec
        except Exception:
            return np.random.randn(n)

    best_labels = None
    best_cut = -1

    # Strategy 1: Spectral + Goemans-Williamson rounding
    vec = get_spectral_vector()
    gw_labels, gw_cut = goemans_williamson_rounding(vec, n_samples=50)
    l, c = local_search(gw_labels, max_rounds=80)
    if c > best_cut:
        best_cut = c
        best_labels = l

    # Strategy 2: Spectral threshold + local search
    vec_thresh = (vec > 0).astype(np.int8)
    l, c = local_search(vec_thresh, max_rounds=80)
    if c > best_cut:
        best_cut = c
        best_labels = l

    # Strategy 3: Alternating + local search
    labels_alt = np.arange(n) % 2
    l, c = local_search(labels_alt, max_rounds=60)
    if c > best_cut:
        best_cut = c
        best_labels = l

    # Strategy 4: Random restarts with local search
    num_restarts = 30
    for _ in range(num_restarts):
        if time_left() < 8.0:
            break
        labels_rand = np.random.randint(0, 2, size=n).astype(np.int8)
        l, c = local_search(labels_rand, max_rounds=40, use_pairwise=True)
        if c > best_cut:
            best_cut = c
            best_labels = l

    # Strategy 5: Simulated annealing from best
    if time_left() > 25.0:
        l, c = simulated_annealing(best_labels, max_iters=30000, T_start=8.0, T_end=0.05)
        if c > best_cut:
            best_cut = c
            best_labels = l

    # Strategy 6: More SA restarts
    if time_left() > 15.0:
        for _ in range(5):
            if time_left() < 5.0:
                break
            labels_rand = np.random.randint(0, 2, size=n).astype(np.int8)
            l, c = simulated_annealing(labels_rand, max_iters=15000, T_start=5.0, T_end=0.05)
            if c > best_cut:
                best_cut = c
                best_labels = l

    # Strategy 7: Final polish
    if time_left() > 5.0:
        for _ in range(10):
            if time_left() < 2.0:
                break
            l, c = local_search(best_labels, max_rounds=100, use_pairwise=True)
            if c > best_cut:
                best_cut = c
                best_labels = l
            else:
                break

    return best_labels.astype(int)
