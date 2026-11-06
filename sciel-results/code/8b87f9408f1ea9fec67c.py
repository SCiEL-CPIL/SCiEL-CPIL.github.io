"""Maximum Cut via Spectral Fiedler Vector Initialization and Multi-Restart Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import heapq

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses spectral (Fiedler vector of Laplacian) initialization, vectorized
    1-flip local search, and multi-restart with diverse spectral projections.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Symmetrize
    adj_sym = (adj + adj.T) // 2
    degrees = adj_sym.sum(axis=1).astype(np.int32)
    neighbors = [np.where(adj_sym[v] == 1)[0] for v in range(n)]

    def compute_cut(labels):
        """Compute the total cut size using matrix ops."""
        x = 2.0 * labels.astype(np.float64) - 1.0  # +1/-1
        # cut = (n_edges - sum_same)/2 ... actually:
        # cut = sum over edges (i,j) where labels differ
        # = 0.25 * (sum x_i) * ... no, simpler:
        # For binary 0/1 labels: cut = sum_{i in 0, j in 1} adj[i,j]
        in0 = (labels == 0).astype(np.int32)
        in1 = (labels == 1).astype(np.int32)
        return int(in0 @ adj_sym @ in1)

    def compute_same(labels):
        """Vectorized: same[v] = number of neighbors of v with same label as v."""
        label0 = (labels == 0).astype(np.int32)
        label1 = (labels == 1).astype(np.int32)
        same = np.zeros(n, dtype=np.int32)
        # For label-0 vertices: count label-0 neighbors
        same[label0 == 1] = (adj_sym @ label0)[label0 == 1]
        # For label-1 vertices: count label-1 neighbors
        same[label1 == 1] = (adj_sym @ label1)[label1 == 1]
        return same

    def local_search(labels, time_limit):
        """
        Fast 1-flip local search with heap-based priority queue.
        Returns (labels, cut_size).
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        same = compute_same(labels)

        # Build initial heap
        heap = []
        for v in range(n):
            gain = 2 * same[v] - degrees[v]
            if gain > 0:
                heapq.heappush(heap, (-gain, v))

        while time.time() < time_limit:
            # Pop best valid flip
            flipped = False
            while heap:
                neg_gain, v = heapq.heappop(heap)
                gain = -neg_gain
                current_gain = 2 * same[v] - degrees[v]
                if current_gain != gain or current_gain <= 0:
                    continue

                # Perform flip
                old_label = labels[v]
                new_label = 1 - old_label
                labels[v] = new_label

                # Update same[v]
                if new_label == 0:
                    same[v] = int(adj_sym[v, labels == 0].sum())
                else:
                    same[v] = int(adj_sym[v, labels == 1].sum())

                # Update neighbors
                for u in neighbors[v]:
                    if u == v:
                        continue
                    if labels[u] == old_label:
                        same[u] -= 1
                    else:
                        same[u] += 1
                    u_gain = 2 * same[u] - degrees[u]
                    if u_gain > 0:
                        heapq.heappush(heap, (-u_gain, u))

                cut += current_gain
                flipped = True
                break

            if not flipped:
                # No more 1-flip improvements; try 2-flip escape
                if n >= 2:
                    # Try a random 2-flip
                    v1, v2 = np.random.choice(n, size=2, replace=False)
                    if v1 == v2:
                        continue
                    g1 = 2 * same[v1] - degrees[v1]
                    g2 = 2 * same[v2] - degrees[v2]
                    combined = g1 + g2
                    if combined > 0:
                        old1, old2 = labels[v1], labels[v2]
                        labels[v1] = 1 - old1
                        labels[v2] = 1 - old2
                        new1, new2 = labels[v1], labels[v2]

                        same[v1] = int(adj_sym[v1, labels == new1].sum())
                        same[v2] = int(adj_sym[v2, labels == new2].sum())

                        for u in neighbors[v1]:
                            if u in (v1, v2):
                                continue
                            if labels[u] == old1:
                                same[u] -= 1
                            else:
                                same[u] += 1
                            u_gain = 2 * same[u] - degrees[u]
                            if u_gain > 0:
                                heapq.heappush(heap, (-u_gain, u))

                        for u in neighbors[v2]:
                            if u in (v1, v2):
                                continue
                            if labels[u] == old2:
                                same[u] -= 1
                            else:
                                same[u] += 1
                            u_gain = 2 * same[u] - degrees[u]
                            if u_gain > 0:
                                heapq.heappush(heap, (-u_gain, u))

                        cut += combined
                        flipped = True
                        # After 2-flip, re-enter 1-flip loop
                        continue

                # No improvement possible
                break

        return labels, cut

    def local_search_kflip(labels, time_limit, k=3):
        """
        Local search with occasional k-flip moves to escape local optima.
        Returns (labels, cut_size).
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        same = compute_same(labels)

        # First do standard 1-flip local search
        labels, cut = local_search(labels, time_limit)
        same = compute_same(labels)

        # Now try k-flip moves
        kflip_attempts = 0
        max_attempts = 500
        improved = True

        while improved and time.time() < time_limit and kflip_attempts < max_attempts:
            improved = False
            kflip_attempts += 1

            # Try random k-flip
            if n >= k:
                vs = np.random.choice(n, size=k, replace=False)
                # Compute combined gain
                # For each vertex in vs, its gain if flipped alone
                # But when flipping multiple, interactions matter
                # Simple approach: compute cut before and after
                old_labels = labels[vs].copy()
                labels[vs] = 1 - old_labels
                new_cut = compute_cut(labels)
                delta = new_cut - cut
                if delta > 0:
                    cut = new_cut
                    same = compute_same(labels)
                    improved = True
                    # Continue 1-flip search after improvement
                    labels, cut = local_search(labels, time_limit)
                    same = compute_same(labels)
                else:
                    labels[vs] = old_labels  # revert

        return labels, cut

    best_labels = None
    best_cut = -1

    rng = np.random.default_rng(42)

    # === Phase 1: Spectral Initialization ===
    candidates = []

    # Fiedler vector (2nd smallest eigenvalue of Laplacian)
    try:
        D = np.diag(degrees.astype(np.float64))
        L = D - adj_sym.astype(np.float64)
        eigvals, eigvecs = np.linalg.eigh(L)

        # Fiedler vector: eigenvector for 2nd smallest eigenvalue
        # eigvecs columns are sorted by eigenvalue ascending
        fiedler = eigvecs[:, 1]  # 2nd smallest
        fiedler_mean = np.mean(fiedler)
        fiedler_labels = (fiedler > fiedler_mean).astype(np.int8)
        candidates.append(fiedler_labels)

        # Also try negative Fiedler
        neg_fiedler_labels = (fiedler < fiedler_mean).astype(np.int8)
        candidates.append(neg_fiedler_labels)

        # Use top-k eigenvectors for diverse projections
        k_eig = min(15, n)
        V = eigvecs[:, :k_eig]  # smallest k eigenvalues (most informative for partitioning)

        # Also try the largest eigenvalues (anti-Fiedler)
        V_large = eigvecs[:, -k_eig:]

        # Random hyperplane cuts in small-eigenvalue space
        seen = set()
        num_projections = 300
        for i in range(num_projections):
            if time.time() > deadline - 30:
                break
            u = rng.standard_normal(k_eig)
            u /= np.linalg.norm(u)
            proj = V @ u
            med = np.median(proj)
            labels = (proj > med).astype(np.int8)
            if np.all(labels == 0) or np.all(labels == 1):
                continue
            key = labels.tobytes()
            if key not in seen:
                seen.add(key)
                candidates.append(labels)

            # Also try in large-eigenvalue space
            u2 = rng.standard_normal(k_eig)
            u2 /= np.linalg.norm(u2)
            proj2 = V_large @ u2
            med2 = np.median(proj2)
            labels2 = (proj2 > med2).astype(np.int8)
            if np.all(labels2 == 0) or np.all(labels2 == 1):
                continue
            key2 = labels2.tobytes()
            if key2 not in seen:
                seen.add(key2)
                candidates.append(labels2)

    except Exception:
        pass

    # Add random and greedy starts
    random_labels = rng.integers(0, 2, size=n, dtype=np.int8)
    candidates.append(random_labels)

    # Greedy by degree
    greedy_labels = np.zeros(n, dtype=np.int8)
    order = np.argsort(-degrees)
    for v in order:
        if len(neighbors[v]) == 0:
            continue
        count_1 = np.sum(greedy_labels[neighbors[v]] == 1)
        count_0 = np.sum(greedy_labels[neighbors[v]] == 0)
        greedy_labels[v] = 1 if count_1 > count_0 else 0
    candidates.append(greedy_labels)

    # Deduplicate candidates
    unique_candidates = []
    seen_cand = set()
    for c in candidates:
        key = c.tobytes()
        if key not in seen_cand:
            seen_cand.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates

    # Sort by initial cut quality, keep top ones
    if len(candidates) > 20:
        scored = [(compute_cut(c), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [c for _, c in scored[:20]]

    # === Phase 2: Refine with Local Search ===
    for labels in candidates:
        if time.time() > deadline - 10:
            break
        ls_labels, ls_cut = local_search(labels, deadline - 10)
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels.copy()

    # === Phase 3: Multi-restart with k-flip on best ===
    if best_labels is not None and time.time() < deadline - 5:
        for restart in range(5):
            if time.time() > deadline - 3:
                break
            # Perturb best solution
            perturbed = best_labels.copy()
            num_flips = max(1, n // 10)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            perturbed[flip_idx] = 1 - perturbed[flip_idx]

            # Try k-flip local search
            kf_labels, kf_cut = local_search_kflip(perturbed, deadline - 3, k=3)
            if kf_cut > best_cut:
                best_cut = kf_cut
                best_labels = kf_labels.copy()

            # Also try standard local search on perturbed
            ls2_labels, ls2_cut = local_search(perturbed, deadline - 3)
            if ls2_cut > best_cut:
                best_cut = ls2_cut
                best_labels = ls2_labels.copy()

    # === Phase 4: Final polish ===
    if best_labels is not None and time.time() < deadline - 1:
        final_labels, final_cut = local_search(best_labels, deadline - 1)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)

    return best_labels.astype(np.int8)

# EVOLVE-BLOCK-END
