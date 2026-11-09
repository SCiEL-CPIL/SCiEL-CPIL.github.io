"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def greedy_1flip(labels, max_passes=10):
        labels = labels.copy()
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                same = int(adj[v][labels == labels[v]].sum())
                other = int(adj[v][labels != labels[v]].sum())
                if same > other:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed:
                break
        return labels

    def fast_2swap(labels, max_passes=100):
        """2-swap local search with incremental O(n) updates per swap."""
        labels = labels.copy()
        sum_same = np.zeros(n, dtype=np.int32)
        sum_other = np.zeros(n, dtype=np.int32)
        for v in range(n):
            sum_same[v] = int(adj[v][labels == labels[v]].sum())
            sum_other[v] = int(adj[v][labels != labels[v]].sum())

        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            order = list(range(n))
            random.shuffle(order)
            for i_idx in range(n):
                if time.time() > deadline:
                    break
                i = order[i_idx]
                for j_idx in range(i_idx + 1, n):
                    if time.time() > deadline:
                        break
                    j = order[j_idx]
                    if labels[i] == labels[j]:
                        continue

                    a = int(adj[i, j])
                    si = int(sum_same[i])
                    oi = int(sum_other[i])
                    sj = int(sum_same[j])
                    oj = int(sum_other[j])

                    gain = si + sj - oi - oj + 2 * a

                    if gain > 0:
                        li = labels[i]
                        lj = labels[j]
                        labels[i], labels[j] = labels[j], labels[i]

                        # Update sum_same/sum_other for i and j
                        sum_same[i], sum_other[i] = oi, si
                        sum_same[j], sum_other[j] = oj, sj

                        # Update all other nodes k in O(n)
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            if labels[k] == li:
                                # k was same-side with i (now other), was other-side with j (now same)
                                delta = int(adj[k, j]) - int(adj[k, i])
                                sum_same[k] += delta
                                sum_other[k] -= delta
                            else:
                                # k was other-side with i (now same), was same-side with j (now other)
                                delta = int(adj[k, i]) - int(adj[k, j])
                                sum_same[k] += delta
                                sum_other[k] -= delta

                        improved = True

        return labels

    def try_labels(labels):
        nonlocal best_labels, best_cut
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    def goemans_williamson_init(num_hyperplanes=20):
        """Use GW SDP relaxation to generate initial partitions via hyperplane rounding."""
        # For max-cut, the SDP relaxation gives vectors y_i with y_i . y_j = 1 - 2*adj[i][j]
        # Approximate by spectral embedding: use the Laplacian's eigenvectors
        # L = D - A, where D is degree matrix
        degree = adj.sum(axis=1).astype(np.float64)
        D = np.diag(degree)
        L = D - adj.astype(np.float64)
        
        # Get the top eigenvectors of L (smallest eigenvalues give the embedding)
        # Actually for max-cut, we want the Fiedler vector and related
        # Use eigendecomposition of L
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
        except:
            return None
        
        # Use the first few non-trivial eigenvectors (skip the trivial zero eigenvalue)
        # For a connected graph, eigvals[0] ≈ 0, use eigvecs[:, 1:k]
        k = min(10, n - 1)
        if k < 2:
            return None
        
        # Embedding matrix: each row is a node's coordinates in R^k
        Y = eigvecs[:, 1:k+1]
        
        # Normalize rows to unit length
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        Y = Y / norms
        
        # Generate random hyperplanes and project
        inits = []
        for _ in range(num_hyperplanes):
            # Random hyperplane direction in R^k
            u = np.random.randn(k)
            u = u / np.linalg.norm(u)
            # Project: sign(Y @ u)
            proj = Y @ u
            labels = (proj > 0).astype(np.int8)
            inits.append(labels)
        
        return inits

    # Strategy 1: Alternating start
    if time.time() < deadline:
        labels = np.arange(n) % 2
        labels = greedy_1flip(labels, max_passes=10)
        labels = fast_2swap(labels, max_passes=80)
        try_labels(labels)

    # Strategy 2: Goemans-Williamson SDP-based initialization
    if time.time() < deadline:
        gw_inits = goemans_williamson_init(num_hyperplanes=20)
        if gw_inits is not None:
            for init_labels in gw_inits:
                if time.time() > deadline:
                    break
                labels = init_labels.copy()
                labels = greedy_1flip(labels, max_passes=5)
                labels = fast_2swap(labels, max_passes=40)
                try_labels(labels)

    # Strategy 3: Multiple random restarts with varied initialization
    for restart in range(30):
        if time.time() > deadline:
            break

        # Vary initialization strategy
        if restart % 3 == 0:
            labels = np.random.randint(0, 2, size=n)
        elif restart % 3 == 1:
            # Biased random: start with slight imbalance
            bias = 0.4 + 0.2 * random.random()
            labels = (np.random.random(n) < bias).astype(np.int8)
        else:
            # Perturbed alternating
            labels = np.arange(n) % 2
            flip_mask = np.random.random(n) < 0.3
            labels[flip_mask] = 1 - labels[flip_mask]

        labels = greedy_1flip(labels, max_passes=5)
        labels = fast_2swap(labels, max_passes=40)
        try_labels(labels)

    # Final polish on best found
    if best_labels is not None and time.time() < deadline:
        final_labels = best_labels.copy()
        final_labels = fast_2swap(final_labels, max_passes=100)
        try_labels(final_labels)

        # One more 1-flip polish
        if time.time() < deadline:
            final_labels = greedy_1flip(final_labels, max_passes=5)
            final_labels = fast_2swap(final_labels, max_passes=50)
            try_labels(final_labels)

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
