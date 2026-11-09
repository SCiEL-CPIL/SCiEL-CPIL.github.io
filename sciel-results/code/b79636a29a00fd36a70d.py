"""Maximum Cut via multi-start spectral + Cerny tabu + 2-swap + 4-swap local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit

    if n == 0:
        return np.array([], dtype=np.int8)

    adj = np.array(adjacency, dtype=np.int32)
    total_edges = int(np.sum(adj)) // 2

    # Precompute neighbor lists and degrees
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)
    max_degree = int(degrees.max()) if n > 0 else 0

    def compute_cut(labels):
        """Compute the cut size for a given labeling."""
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def compute_same_count(labels):
        """For each vertex, count neighbors on the same side."""
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1
        return same_count

    def flip_greedy(labels, max_passes=80):
        """Fast 1-flip greedy using incremental cut tracking with randomized order."""
        labels = labels.copy()
        same_count = compute_same_count(labels)

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            order = list(range(n))
            random.shuffle(order)
            for v in order:
                if time.time() > deadline:
                    break
                deg = degrees[v]
                gain = 2 * same_count[v] - deg
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    changed = True
                    old_same = same_count[v]
                    same_count[v] = deg - old_same
                    for u in neighbors[v]:
                        if labels[u] == labels[v]:
                            same_count[u] += 1
                        else:
                            same_count[u] -= 1
            if not changed:
                break
        return labels

    def cerny_tabu(labels, max_iters=6000, tabu_tenure=18):
        """Cerny tabu search: full-scan, accepts worsening moves, tabu with aspiration."""
        labels = labels.copy()
        same_count = compute_same_count(labels)
        current_cut = compute_cut(labels)
        best_cut = current_cut
        best_labels = labels.copy()

        # tabu_until[v] = iteration index at which v becomes non-tabu
        tabu_until = {}
        iteration = 0
        no_improve = 0
        stall_limit = max_iters // 4

        while iteration < max_iters:
            if time.time() > deadline:
                break

            # Find best move among all non-tabu vertices (aspiration: allow tabu if beats best)
            best_gain = -999999
            best_v = -1

            for v in range(n):
                gain = 2 * same_count[v] - degrees[v]
                is_tabu = (v in tabu_until) and (tabu_until[v] > iteration)
                if is_tabu:
                    # Aspiration: only allow if this move exceeds current best
                    if current_cut + gain > best_cut:
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                else:
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v

            if best_v < 0:
                break

            # Perform the flip
            v = best_v
            labels[v] = 1 - labels[v]
            tabu_until[v] = iteration + tabu_tenure

            old_same = same_count[v]
            same_count[v] = degrees[v] - old_same
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[u] += 1
                else:
                    same_count[u] -= 1

            current_cut += best_gain
            iteration += 1

            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > stall_limit:
                    break

        return best_labels

    def two_swap_search(labels, max_passes=12):
        """2-swap local search with incremental updates."""
        labels = labels.copy()
        same_count = compute_same_count(labels)

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

                    e_ij = adj[i, j]
                    # After swapping i and j:
                    new_same_i = degrees[i] - same_count[i] - e_ij
                    new_same_j = degrees[j] - same_count[j] - e_ij
                    delta_same = (new_same_i + new_same_j) - (same_count[i] + same_count[j])

                    if delta_same > 0:
                        labels[i], labels[j] = labels[j], labels[i]
                        improved = True

                        old_same_i = same_count[i]
                        old_same_j = same_count[j]
                        same_count[i] = new_same_i
                        same_count[j] = new_same_j

                        for u in neighbors[i]:
                            if u == j:
                                continue
                            if labels[u] == labels[i]:
                                same_count[u] += 1
                            else:
                                same_count[u] -= 1

                        for u in neighbors[j]:
                            if u == i:
                                continue
                            if labels[u] == labels[j]:
                                same_count[u] += 1
                            else:
                                same_count[u] -= 1

            if not improved:
                break

        return labels

    def four_swap_search(labels, max_passes=6):
        """4-swap local search: flip two vertices from each side."""
        labels = labels.copy()
        same_count = compute_same_count(labels)

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(n))
            random.shuffle(order)

            # Get vertices on each side
            side0 = [v for v in order if labels[v] == 0]
            side1 = [v for v in order if labels[v] == 1]

            for i in side0:
                if time.time() > deadline:
                    break
                for j in side0:
                    if time.time() > deadline:
                        break
                    if i >= j:
                        continue
                    for k in side1:
                        if time.time() > deadline:
                            break
                        for l in side1:
                            if time.time() > deadline:
                                break
                            if k >= l:
                                continue

                            # Compute gain of flipping i, j (from 0) and k, l (from 1)
                            # Gain = sum of gains for each individual flip, adjusted for interactions
                            gain_i = 2 * same_count[i] - degrees[i]
                            gain_j = 2 * same_count[j] - degrees[j]
                            gain_k = 2 * same_count[k] - degrees[k]
                            gain_l = 2 * same_count[l] - degrees[l]

                            # Adjust for edges between the four vertices
                            # If i-k edge exists and they're on different sides, flipping both removes it from cut
                            # We need to be more careful: compute net change in cut
                            # For simplicity, use the sum of individual gains as an approximation
                            # and verify by actually computing the cut change
                            total_gain = gain_i + gain_j + gain_k + gain_l

                            # Subtract interaction terms: for each pair among {i,j,k,l},
                            # if they are connected, the individual gains double-count the edge
                            # Edge (i,j): both flip from 0, so edge stays non-cut. No change.
                            # Edge (k,l): both flip from 1, so edge stays non-cut. No change.
                            # Edge (i,k): i flips 0->1, k flips 1->0. Before: cut (different). After: non-cut (same).
                            #   So this edge is removed from cut: gain decreases by 1 (we lose a cut edge)
                            #   But wait: gain_i already accounts for i's neighbors. Let me think more carefully.
                            #
                            # Actually, the gain formula 2*same_count[v] - degree[v] gives the change in cut
                            # when flipping v alone. When flipping multiple vertices, we need to account for
                            # interactions. The net change is:
                            # sum of individual gains - 2 * (number of edges between flipped vertices on different sides)
                            # + 2 * (number of edges between flipped vertices on same side that become different)
                            #
                            # This is getting complex. Let me just compute it directly for the four vertices.
                            # Actually, for a 4-swap, let's just compute the cut change directly.

                            # Compute cut change by checking all edges incident to i, j, k, l
                            # An edge contributes to cut if its endpoints are on different sides.
                            # Before flip:
                            #   i: 0, j: 0, k: 1, l: 1
                            # After flip:
                            #   i: 1, j: 1, k: 0, l: 0

                            # Edges to check: all edges from {i,j,k,l} to any vertex
                            # But we can simplify: only edges incident to i, j, k, or l change status.

                            # Let's compute the delta directly
                            delta = 0
                            # Edges from i to others
                            for u in neighbors[i]:
                                if u == j or u == k or u == l:
                                    continue
                                if labels[u] == 0:
                                    # Before: i(0)-u(0) non-cut. After: i(1)-u(0) cut. +1
                                    delta += 1
                                else:
                                    # Before: i(0)-u(1) cut. After: i(1)-u(1) non-cut. -1
                                    delta -= 1
                            # Edges from j to others
                            for u in neighbors[j]:
                                if u == i or u == k or u == l:
                                    continue
                                if labels[u] == 0:
                                    delta += 1
                                else:
                                    delta -= 1
                            # Edges from k to others
                            for u in neighbors[k]:
                                if u == i or u == j or u == l:
                                    continue
                                if labels[u] == 0:
                                    # Before: k(1)-u(0) cut. After: k(0)-u(0) non-cut. -1
                                    delta -= 1
                                else:
                                    # Before: k(1)-u(1) non-cut. After: k(0)-u(1) cut. +1
                                    delta += 1
                            # Edges from l to others
                            for u in neighbors[l]:
                                if u == i or u == j or u == k:
                                    continue
                                if labels[u] == 0:
                                    delta -= 1
                                else:
                                    delta += 1

                            # Edges between the four vertices
                            # i-j: both 0 -> both 1. Before: non-cut. After: non-cut. No change.
                            # k-l: both 1 -> both 0. Before: non-cut. After: non-cut. No change.
                            # i-k: 0-1 -> 1-0. Before: cut. After: non-cut. -1
                            if adj[i, k]:
                                delta -= 1
                            # i-l: 0-1 -> 1-0. Before: cut. After: non-cut. -1
                            if adj[i, l]:
                                delta -= 1
                            # j-k: 0-1 -> 1-0. Before: cut. After: non-cut. -1
                            if adj[j, k]:
                                delta -= 1
                            # j-l: 0-1 -> 1-0. Before: cut. After: non-cut. -1
                            if adj[j, l]:
                                delta -= 1

                            if delta > 0:
                                # Perform the 4-swap
                                labels[i] = 1
                                labels[j] = 1
                                labels[k] = 0
                                labels[l] = 0
                                improved = True

                                # Update same_count
                                for v in [i, j, k, l]:
                                    old_same = same_count[v]
                                    same_count[v] = degrees[v] - old_same

                                for v in [i, j]:
                                    for u in neighbors[v]:
                                        if u in (i, j, k, l):
                                            continue
                                        if labels[u] == labels[v]:
                                            same_count[u] += 1
                                        else:
                                            same_count[u] -= 1
                                for v in [k, l]:
                                    for u in neighbors[v]:
                                        if u in (i, j, k, l):
                                            continue
                                        if labels[u] == labels[v]:
                                            same_count[u] += 1
                                        else:
                                            same_count[u] -= 1

                                # Update same_count for the four vertices themselves
                                for v in [i, j, k, l]:
                                    same_count[v] = 0
                                    for u in neighbors[v]:
                                        if labels[u] == labels[v]:
                                            same_count[v] += 1

            if not improved:
                break

        return labels

    def spectral_partition(k=32, num_random=5):
        """Spectral partitioning with multiple random hyperplanes."""
        try:
            D = np.diag(degrees.astype(np.float64))
            L = D - adj.astype(np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(L)

            k = min(k, n - 1)
            embedding = eigenvectors[:, 1:k + 1]

            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms

            best_labels = None
            best_cut = -1

            for _ in range(num_random):
                random_vec = np.random.randn(k)
                random_vec /= np.linalg.norm(random_vec)
                projection = embedding @ random_vec
                labels = (projection > 0).astype(np.int8)
                cut = compute_cut(labels)
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels

            if best_labels is not None:
                return best_labels
        except Exception:
            pass
        return np.random.randint(0, 2, size=n, dtype=np.int8)

    def fiedler_partition():
        """Partition using the Fiedler vector (2nd eigenvector of Laplacian)."""
        try:
            D = np.diag(degrees.astype(np.float64))
            L = D - adj.astype(np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler = eigenvectors[:, 1]
            return (fiedler > 0).astype(np.int8)
        except Exception:
            return np.random.randint(0, 2, size=n, dtype=np.int8)

    def greedy_partition():
        """Greedy partition: assign vertices one by one to maximize cut."""
        labels = np.zeros(n, dtype=np.int8)
        order = np.argsort(-degrees)
        for v in order:
            count_0 = 0
            count_1 = 0
            for u in neighbors[v]:
                if labels[u] == 0:
                    count_0 += 1
                else:
                    count_1 += 1
            if count_0 < count_1:
                labels[v] = 1
            elif count_1 < count_0:
                labels[v] = 0
            else:
                labels[v] = random.randint(0, 1)
        return labels

    best_cut = -1
    best_labels = None

    def try_solution(labels):
        nonlocal best_cut, best_labels
        # 1-flip greedy
        labels1 = flip_greedy(labels, max_passes=60)
        cut1 = compute_cut(labels1)
        if cut1 > best_cut:
            best_cut = cut1
            best_labels = labels1.copy()

        # Cerny tabu (the key escape mechanism)
        if time.time() < deadline:
            labels2 = cerny_tabu(labels1, max_iters=4000, tabu_tenure=18)
            cut2 = compute_cut(labels2)
            if cut2 > best_cut:
                best_cut = cut2
                best_labels = labels2.copy()

        # 2-swap
        if time.time() < deadline:
            labels3 = two_swap_search(labels2, max_passes=10)
            cut3 = compute_cut(labels3)
            if cut3 > best_cut:
                best_cut = cut3
                best_labels = labels3.copy()

        # 4-swap (new: escape 2-swap local optima)
        if time.time() < deadline:
            labels4 = four_swap_search(labels3, max_passes=4)
            cut4 = compute_cut(labels4)
            if cut4 > best_cut:
                best_cut = cut4
                best_labels = labels4.copy()

        # Another round: 1-flip + tabu + 2-swap + 4-swap on the result
        if time.time() < deadline:
            labels5 = flip_greedy(labels4, max_passes=40)
            labels6 = cerny_tabu(labels5, max_iters=2500, tabu_tenure=14)
            labels7 = two_swap_search(labels6, max_passes=8)
            labels8 = four_swap_search(labels7, max_passes=3)
            cut8 = compute_cut(labels8)
            if cut8 > best_cut:
                best_cut = cut8
                best_labels = labels8.copy()

    # Generate diverse initial solutions
    init_solutions = []
    init_solutions.append(spectral_partition(k=32, num_random=5))
    init_solutions.append(spectral_partition(k=16, num_random=5))
    init_solutions.append(spectral_partition(k=8, num_random=5))
    init_solutions.append(fiedler_partition())
    init_solutions.append(greedy_partition())
    init_solutions.append(np.random.randint(0, 2, size=n, dtype=np.int8))
    init_solutions.append(np.arange(n) % 2)

    for init in init_solutions:
        if time.time() > deadline:
            break
        try_solution(init)

    # Additional random restarts
    num_restarts = 12
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        if r % 3 == 0:
            labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        elif r % 3 == 1:
            labels = spectral_partition(k=16, num_random=1)
        else:
            labels = greedy_partition()

        try_solution(labels)

    # Final polish: multiple rounds of local search on best solution
    if best_labels is not None and time.time() < deadline:
        for _ in range(3):
            if time.time() > deadline:
                break
            labels = flip_greedy(best_labels, max_passes=40)
            labels = cerny_tabu(labels, max_iters=2000, tabu_tenure=12)
            labels = two_swap_search(labels, max_passes=10)
            labels = four_swap_search(labels, max_passes=3)
            cut = compute_cut(labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
