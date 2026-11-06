"""Maximum Cut via multi-restart local search with simulated annealing and Goemans-Williamson rounding."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses multi-restart local search with simulated annealing and GW rounding.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    # Precompute neighbor lists for fast access
    neighbors = [np.where(adj[v] == 1)[0] for v in range(n)]

    # Deadline: leave 15 seconds margin for safety
    start_time = time.time()
    deadline = start_time + 135.0

    def time_left():
        return deadline - time.time()

    def compute_cut(labels):
        """Compute the total cut size."""
        in_group0 = (labels == 0).astype(np.int32)
        in_group1 = (labels == 1).astype(np.int32)
        return int(in_group0 @ adj @ in_group1)

    def compute_same(labels):
        """Compute for each vertex the number of neighbors in the same group."""
        in_group0 = (labels == 0).astype(np.int32)
        in_group1 = (labels == 1).astype(np.int32)
        same = np.zeros(n, dtype=np.int32)
        mask0 = labels == 0
        mask1 = labels == 1
        if np.any(mask0):
            same[mask0] = (adj @ in_group0)[mask0]
        if np.any(mask1):
            same[mask1] = (adj @ in_group1)[mask1]
        return same

    def local_search(labels, deadline):
        """Run first-improvement local search until no improvement or deadline reached."""
        labels = labels.copy()
        same = compute_same(labels)
        cut = compute_cut(labels)
        improved = True
        while improved and time_left() > 0:
            improved = False
            for v in range(n):
                if time_left() <= 0:
                    break
                gain = 2 * same[v] - degrees[v]
                if gain > 0:
                    # Flip vertex v
                    old_group = labels[v]
                    new_group = 1 - old_group
                    labels[v] = new_group
                    # Update same[v]
                    if new_group == 0:
                        same[v] = int(adj[v, labels == 0].sum())
                    else:
                        same[v] = int(adj[v, labels == 1].sum())
                    # Update neighbors
                    for u in neighbors[v]:
                        if u == v:
                            continue
                        if labels[u] == old_group:
                            same[u] -= 1
                        elif labels[u] == new_group:
                            same[u] += 1
                    cut += gain
                    improved = True
        return labels, cut

    def simulated_annealing(labels, deadline, rng):
        """Run simulated annealing starting from given labels."""
        labels = labels.copy()
        same = compute_same(labels)
        cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = cut

        T = 50.0
        cooling = 0.95
        min_T = 0.1
        max_steps = 10000

        for step in range(max_steps):
            if time_left() <= 0:
                break
            if T < min_T:
                break

            # Pick a random vertex
            v = rng.integers(0, n)
            gain = 2 * same[v] - degrees[v]

            # Decide whether to accept
            if gain > 0 or (T > 0 and rng.random() < np.exp(gain / T)):
                # Flip vertex v
                old_group = labels[v]
                new_group = 1 - old_group
                labels[v] = new_group
                # Update same[v]
                if new_group == 0:
                    same[v] = int(adj[v, labels == 0].sum())
                else:
                    same[v] = int(adj[v, labels == 1].sum())
                # Update neighbors
                for u in neighbors[v]:
                    if u == v:
                        continue
                    if labels[u] == old_group:
                        same[u] -= 1
                    elif labels[u] == new_group:
                        same[u] += 1
                cut += gain
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels.copy()

            T *= cooling

        # Final local search polish
        best_labels, best_cut = local_search(best_labels, deadline)
        return best_labels, best_cut

    def goemans_williamson_init(rng):
        """Generate a random vector for each vertex and round to get a partition."""
        # Generate random unit vectors in R^n
        vecs = rng.standard_normal((n, n))
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs /= norms

        # For each edge (i,j), compute dot product
        # The probability that i and j are on opposite sides is (1 - dot(i,j))/pi
        # We can use a random hyperplane to round
        # Generate random hyperplane normal
        normal = rng.standard_normal(n)
        normal /= (np.linalg.norm(normal) + 1e-10)

        # Project each vertex onto the normal
        projections = vecs @ normal

        # Round: if projection > 0, label 1; else label 0
        labels = (projections > 0).astype(np.int8)
        return labels

    def random_init(rng):
        return rng.integers(0, 2, size=n, dtype=np.int8)

    def greedy_init():
        """Greedy sequential initialization."""
        labels = np.zeros(n, dtype=np.int8)
        order = np.argsort(-degrees)
        for v in order:
            nbrs = neighbors[v]
            if len(nbrs) == 0:
                continue
            count_0 = np.sum(labels[nbrs] == 0)
            count_1 = np.sum(labels[nbrs] == 1)
            if count_1 > count_0:
                labels[v] = 1
            else:
                labels[v] = 0
        return labels

    def alternating_init():
        return np.arange(n) % 2

    # Generate initializations
    rng = np.random.default_rng(42)

    initializations = []
    initializations.append(alternating_init())
    initializations.append(greedy_init())
    initializations.append(goemans_williamson_init(rng))
    initializations.append(goemans_williamson_init(rng))
    initializations.append(goemans_williamson_init(rng))

    for _ in range(30):
        initializations.append(random_init(rng))

    # All zeros with one flip
    labels_zero = np.zeros(n, dtype=np.int8)
    labels_zero[0] = 1
    initializations.append(labels_zero.copy())

    # All ones with one flip
    labels_one = np.ones(n, dtype=np.int8)
    labels_one[0] = 0
    initializations.append(labels_one.copy())

    best_labels = None
    best_cut = -1

    # Phase 1: Local search on all initializations
    for init_labels in initializations:
        if time_left() <= 10:
            break
        labels, cut = local_search(init_labels, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Phase 2: Simulated annealing from best and some random starts
    if best_labels is not None and time_left() > 20:
        sa_labels, sa_cut = simulated_annealing(best_labels, deadline, rng)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()

    # Phase 3: More SA restarts if time permits
    sa_count = 0
    while time_left() > 15 and sa_count < 10:
        init = random_init(rng)
        sa_labels, sa_cut = simulated_annealing(init, deadline, rng)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        sa_count += 1

    # Final polish
    if best_labels is not None and time_left() > 2:
        best_labels, best_cut = local_search(best_labels, deadline)

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)

    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
