"""Maximum Cut via Spectral Initialization, Fast Local Search, and Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import heapq

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses spectral (Fiedler vector) initialization, fast local search,
    and simulated annealing refinement.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Precompute degrees and neighbor lists
    degrees = adj.sum(axis=1).astype(np.int32)
    neighbors = [np.where(adj[v] == 1)[0] for v in range(n)]
    total_edges = int(adj.sum()) // 2

    def compute_cut(labels):
        """Compute the total cut size."""
        in_group0 = (labels == 0).astype(np.int32)
        in_group1 = (labels == 1).astype(np.int32)
        return int(in_group0 @ adj @ in_group1)

    def compute_same(labels):
        """Compute same[v] = number of neighbors u where labels[u] == labels[v]."""
        same = np.zeros(n, dtype=np.int32)
        for v in range(n):
            if labels[v] == 0:
                same[v] = int(adj[v, labels == 0].sum())
            else:
                same[v] = int(adj[v, labels == 1].sum())
        return same

    def local_search(labels, time_limit):
        """
        Run local search until no improvement or time limit reached.
        Returns (labels, cut_size).
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        same = compute_same(labels)

        # Heap of (-gain, vertex)
        heap = []
        for v in range(n):
            gain = 2 * same[v] - degrees[v]
            if gain > 0:
                heapq.heappush(heap, (-gain, v))

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            while heap:
                neg_gain, v = heapq.heappop(heap)
                gain = -neg_gain

                # Verify gain is still valid
                current_gain = 2 * same[v] - degrees[v]
                if current_gain != gain or current_gain <= 0:
                    continue

                # Flip vertex v
                old_label = labels[v]
                labels[v] = 1 - old_label
                new_label = labels[v]

                # Update same[v]
                if new_label == 0:
                    same[v] = int(adj[v, labels == 0].sum())
                else:
                    same[v] = int(adj[v, labels == 1].sum())

                # Update neighbors of v
                for u in neighbors[v]:
                    if u == v:
                        continue
                    if labels[u] == old_label:
                        same[u] -= 1
                    elif labels[u] == new_label:
                        same[u] += 1

                    # Check if u now has positive gain
                    u_gain = 2 * same[u] - degrees[u]
                    if u_gain > 0:
                        heapq.heappush(heap, (-u_gain, u))

                cut += current_gain
                improved = True
                break

        return labels, cut

    def simulated_annealing(labels, time_limit):
        """
        Run simulated annealing to escape local minima.
        Returns (labels, cut_size).
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = cut

        T_start = 10.0
        T_end = 0.1
        cooling_rate = 0.95
        T = T_start

        same = compute_same(labels)

        while T > T_end and time.time() < time_limit:
            v = np.random.randint(0, n)

            gain = 2 * same[v] - degrees[v]

            # Accept if gain > 0 or with probability exp(-gain/T)
            if gain > 0 or (np.random.rand() < np.exp(-gain / T)):
                old_label = labels[v]
                labels[v] = 1 - old_label
                new_label = labels[v]

                # Update same[v]
                if new_label == 0:
                    same[v] = int(adj[v, labels == 0].sum())
                else:
                    same[v] = int(adj[v, labels == 1].sum())

                # Update neighbors
                for u in neighbors[v]:
                    if u == v:
                        continue
                    if labels[u] == old_label:
                        same[u] -= 1
                    elif labels[u] == new_label:
                        same[u] += 1

                cut += gain

                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels.copy()

            T *= cooling_rate

        return best_labels, best_cut

    def spectral_init():
        """
        Generate initial labeling using the Fiedler vector (second smallest eigenvector).
        """
        try:
            adj_sym = (adj + adj.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(adj_sym)

            # Use the Fiedler vector (index 1, second smallest eigenvalue)
            fiedler = eigvecs[:, 1]

            # Round based on sign
            labels = (fiedler > 0).astype(np.int8)

            # Ensure both classes are non-empty
            if np.all(labels == 0) or np.all(labels == 1):
                labels = (fiedler > np.median(fiedler)).astype(np.int8)

            return labels
        except Exception:
            return None

    rng = np.random.default_rng(42)

    best_labels = None
    best_cut = -1

    # Generate multiple candidate initializations
    candidates = []

    # 1. Spectral (Fiedler) initialization
    spectral = spectral_init()
    if spectral is not None:
        candidates.append(spectral)

    # 2. Random initialization
    random_labels = rng.integers(0, 2, size=n, dtype=np.int8)
    candidates.append(random_labels)

    # 3. Greedy initialization
    greedy_labels = np.zeros(n, dtype=np.int8)
    order = np.argsort(-degrees)
    for v in order:
        if len(neighbors[v]) == 0:
            continue
        count_1 = np.sum(greedy_labels[neighbors[v]] == 1)
        count_0 = np.sum(greedy_labels[neighbors[v]] == 0)
        if count_1 > count_0:
            greedy_labels[v] = 1
        else:
            greedy_labels[v] = 0
    candidates.append(greedy_labels)

    # 4. Additional spectral-based candidates using different eigenvectors
    try:
        adj_sym = (adj + adj.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(adj_sym)

        # Try a few different eigenvectors
        for idx in [0, 2, 3]:
            if idx < n:
                vec = eigvecs[:, idx]
                labels = (vec > np.median(vec)).astype(np.int8)
                if not (np.all(labels == 0) or np.all(labels == 1)):
                    candidates.append(labels)
    except Exception:
        pass

    # 5. Add a few more random candidates
    for _ in range(3):
        random_labels = rng.integers(0, 2, size=n, dtype=np.int8)
        candidates.append(random_labels)

    # Refine each candidate with local search
    for labels in candidates:
        if time.time() > deadline - 15:
            break

        ls_labels, ls_cut = local_search(labels, deadline - 15)
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels.copy()

    # Run simulated annealing on the best solution found
    if best_labels is not None and time.time() < deadline - 10:
        sa_labels, sa_cut = simulated_annealing(best_labels, deadline - 10)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()

    # Final local search polish
    if best_labels is not None:
        final_labels, final_cut = local_search(best_labels, deadline - 1)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)

    return best_labels.astype(np.int8)

# EVOLVE-BLOCK-END
