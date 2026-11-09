"""
Maximum Cut solver combining spectral/SDP initialization with tabu search,
pairwise swaps, perturbation, and simulated annealing.
"""
import numpy as np
import time
import random


def partition(adjacency):
    """
    Find a maximum cut in an unweighted graph using:
    1. Multiple diverse initializations (spectral, SDP, Chebyshev, random)
    2. Greedy single-flip local search
    3. Tabu search with aspiration to escape local optima
    4. Pairwise swap local search
    5. Perturbation search
    6. Simulated annealing
    7. Explicit deadline guard
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    adj = np.asarray(adjacency, dtype=np.int32)
    n = len(adj)

    if n == 0:
        return np.array([], dtype=np.int8)
    if n == 1:
        return np.array([0], dtype=np.int8)

    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    total_edges = int(degrees.sum()) // 2

    def compute_cut(labels):
        """Compute cut size efficiently using vectorized operations."""
        diff = (labels[:, None] != labels[None, :]).astype(np.int32)
        return int(np.sum(adj * diff) // 2)

    def flip_greedy(labels, max_passes=50):
        """Single-flip greedy improvement. Fast O(n^2) per pass."""
        labels = labels.copy()
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                same_side = adj[v, labels == labels[v]].sum()
                other_side = adj[v, labels != labels[v]].sum()
                if same_side > other_side:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed:
                break
        return labels

    def pairwise_swap_search(labels, max_passes=20):
        """2-swap local search: swap labels of two vertices."""
        labels = labels.copy()
        current_cut = compute_cut(labels)

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = np.random.permutation(n)
            for idx_i in range(n):
                if time.time() > deadline:
                    break
                i = order[idx_i]
                for idx_j in range(idx_i + 1, n):
                    if time.time() > deadline:
                        break
                    j = order[idx_j]
                    if labels[i] == labels[j]:
                        continue

                    labels[i], labels[j] = labels[j], labels[i]
                    new_cut = compute_cut(labels)
                    if new_cut > current_cut:
                        current_cut = new_cut
                        improved = True
                    else:
                        labels[i], labels[j] = labels[j], labels[i]
            if not improved:
                break
        return labels, current_cut

    def tabu_search(labels, max_iter=5000, tabu_tenure=15):
        """
        Tabu search with aspiration criterion.
        Maintains a tabu list of recently flipped vertices.
        Allows worsening moves if they don't violate tabu or meet aspiration.
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = current_cut

        # Tabu list: stores iteration when vertex was last flipped
        tabu_until = np.zeros(n, dtype=np.int32)
        iteration = 0

        for iteration in range(max_iter):
            if time.time() > deadline:
                break

            # Find the best move (single flip) that either:
            # - Improves the cut (aspiration)
            # - Is not tabu
            best_move = -1
            best_delta = -np.inf

            # Sample a subset of vertices for efficiency
            sample_size = min(n, 30)
            candidates = np.random.choice(n, size=sample_size, replace=False)

            for v in candidates:
                if tabu_until[v] > iteration:
                    continue  # Tabu, skip unless aspiration

                # Compute delta for flipping vertex v
                same_side = adj[v, labels == labels[v]].sum()
                other_side = adj[v, labels != labels[v]].sum()
                delta = other_side - same_side

                if delta > best_delta:
                    best_delta = delta
                    best_move = v

            if best_move == -1:
                # All candidates are tabu, pick the best regardless
                for v in candidates:
                    same_side = adj[v, labels == labels[v]].sum()
                    other_side = adj[v, labels != labels[v]].sum()
                    delta = other_side - same_side
                    if delta > best_delta:
                        best_delta = delta
                        best_move = v

            if best_move == -1:
                break

            # Apply the move
            labels[best_move] = 1 - labels[best_move]
            tabu_until[best_move] = iteration + tabu_tenure
            current_cut += best_delta

            # Update best
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

            # Adaptive tabu tenure: increase if stuck, decrease if improving
            if current_cut >= best_cut:
                tabu_tenure = max(10, tabu_tenure - 1)
            else:
                tabu_tenure = min(30, tabu_tenure + 1)

        return best_labels, best_cut

    def spectral_init():
        """Spectral initialization using Fiedler vector."""
        L = np.diag(degrees.astype(np.float64)) - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            median_val = np.median(fiedler)
            labels = (fiedler > median_val).astype(np.int32)
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.random.randint(0, 2, size=n)
        except Exception:
            labels = np.random.randint(0, 2, size=n)
        return labels

    def random_init():
        return np.random.randint(0, 2, size=n)

    def sdp_hyperplane_init():
        """
        Approximate SDP hyperplane initialization.
        Uses top eigenvectors of the adjacency matrix to create a subspace,
        then projects a random vector onto this subspace to generate a partition.
        """
        try:
            k = min(5, n)
            eigvals, eigvecs = np.linalg.eigh(adj)
            idx = np.argsort(eigvals)[::-1]
            top_vecs = eigvecs[:, idx[:k]]

            random_coeff = np.random.randn(k)
            scores = top_vecs @ random_coeff

            median_val = np.median(scores)
            labels = (scores > median_val).astype(np.int32)

            if np.all(labels == 0) or np.all(labels == 1):
                labels = np.random.randint(0, 2, size=n)

        except Exception:
            labels = np.random.randint(0, 2, size=n)

        return labels

    def chebyshev_init():
        """Use top eigenvector of adjacency matrix."""
        try:
            eigvals, eigvecs = np.linalg.eigh(adj)
            idx = np.argmax(eigvals)
            vec = eigvecs[:, idx]
            median_val = np.median(vec)
            labels = (vec > median_val).astype(np.int32)
            if np.all(labels == 0) or np.all(labels == 1):
                labels = np.random.randint(0, 2, size=n)
        except Exception:
            labels = np.random.randint(0, 2, size=n)
        return labels

    def perturbation_search(labels, num_perturbations=50):
        """
        Apply random perturbations to escape local optima.
        Each perturbation flips a random subset of vertices, then
        runs local search to find a new local optimum.
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)

        for _ in range(num_perturbations):
            if time.time() > deadline:
                break

            # Create a perturbed version
            perturbed = labels.copy()
            flip_fraction = random.uniform(0.1, 0.3)
            num_flips = int(n * flip_fraction)
            flip_indices = random.sample(range(n), min(num_flips, n))
            for idx in flip_indices:
                perturbed[idx] = 1 - perturbed[idx]

            # Run local search on perturbed solution
            perturbed = flip_greedy(perturbed, max_passes=20)
            perturbed, cut = pairwise_swap_search(perturbed, max_passes=10)

            if cut > current_cut:
                current_cut = cut
                labels = perturbed.copy()

        return labels, current_cut

    def simulated_annealing(labels, max_iter=5000):
        """Simulated annealing to escape local optima."""
        current = labels.copy()
        current_cut = compute_cut(current)
        best = current.copy()
        best_cut = current_cut

        T = 2.0
        T_min = 0.05
        cooling = 0.9995

        for iteration in range(max_iter):
            if time.time() > deadline:
                break

            # Pick a random vertex to flip
            v = np.random.randint(n)
            new_labels = current.copy()
            new_labels[v] = 1 - new_labels[v]
            new_cut = compute_cut(new_labels)

            delta = new_cut - current_cut

            if delta >= 0:
                current = new_labels
                current_cut = new_cut
                if current_cut > best_cut:
                    best = current.copy()
                    best_cut = current_cut
            else:
                if np.random.random() < np.exp(delta / T):
                    current = new_labels
                    current_cut = new_cut

            T *= cooling
            if T < T_min:
                T = T_min

        return best, best_cut

    # Strategy: multiple diverse initializations + local search + tabu + perturbation + annealing
    best_labels = None
    best_cut = -1

    def try_candidate(labels):
        nonlocal best_labels, best_cut
        if time.time() > deadline:
            return
        labels = flip_greedy(labels)
        labels, c = pairwise_swap_search(labels, max_passes=15)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()

    # 1. SDP hyperplane init
    if time.time() < deadline:
        try_candidate(sdp_hyperplane_init())

    # 2. Spectral init
    if time.time() < deadline:
        try_candidate(spectral_init())

    # 3. Chebyshev init
    if time.time() < deadline:
        try_candidate(chebyshev_init())

    # 4. Alternating parity init
    if time.time() < deadline:
        try_candidate(np.arange(n) % 2)

    # 5. Multiple random restarts
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        try_candidate(random_init())

    # 6. Biased random restarts
    for restart in range(10):
        if time.time() > deadline - 30:
            break
        p = np.random.uniform(0.3, 0.7)
        labels = (np.random.random(n) < p).astype(np.int32)
        try_candidate(labels)

    # 7. Tabu search on best solution
    if best_labels is not None and time.time() < deadline - 50:
        tabu_labels, tabu_cut = tabu_search(best_labels, max_iter=3000, tabu_tenure=15)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()
        # Polish after tabu
        if time.time() < deadline - 45:
            polished = flip_greedy(best_labels)
            polished, pc = pairwise_swap_search(polished, max_passes=10)
            if pc > best_cut:
                best_cut = pc
                best_labels = polished.copy()

    # 8. Perturbation search on best solution
    if best_labels is not None and time.time() < deadline - 40:
        pert_labels, pert_cut = perturbation_search(best_labels, num_perturbations=50)
        if pert_cut > best_cut:
            best_cut = pert_cut
            best_labels = pert_labels.copy()

    # 9. Simulated annealing from best solution
    if best_labels is not None and time.time() < deadline - 20:
        sa_labels, sa_cut = simulated_annealing(best_labels, max_iter=5000)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        # Final greedy polish after SA
        if time.time() < deadline - 5:
            polished = flip_greedy(best_labels)
            polished, pc = pairwise_swap_search(polished, max_passes=10)
            if pc > best_cut:
                best_cut = pc
                best_labels = polished.copy()

    # 10. One more round of pairwise swaps if time permits
    if best_labels is not None and time.time() < deadline - 5:
        swapped, sc = pairwise_swap_search(best_labels, max_passes=30)
        if sc > best_cut:
            best_cut = sc
            best_labels = swapped.copy()

    # Final greedy polish
    if best_labels is not None and time.time() < deadline - 1:
        final_labels = flip_greedy(best_labels)
        final_labels, final_cut = pairwise_swap_search(final_labels, max_passes=10)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels

    if best_labels is None:
        # Fallback: alternating pattern
        best_labels = np.arange(n) % 2

    return best_labels.astype(np.int8)
