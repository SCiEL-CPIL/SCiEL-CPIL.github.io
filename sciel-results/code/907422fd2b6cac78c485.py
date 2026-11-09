"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def try_labels(labels):
        nonlocal best_labels, best_cut
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # --- Spectral initialization (Goemans-Williamson style) ---
    def spectral_init():
        degree = adj.sum(axis=1).astype(np.float64)
        D = np.diag(degree)
        L = D - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            k = min(5, n - 1)
            if k >= 2:
                Y = eigvecs[:, 1:k+1]
                norms = np.linalg.norm(Y, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                Y = Y / norms
                u = np.random.randn(k)
                u = u / np.linalg.norm(u)
                proj = Y @ u
                labels = (proj > 0).astype(np.int8)
            else:
                labels = np.random.randint(0, 2, size=n)
        except:
            labels = np.random.randint(0, 2, size=n)
        return labels

    # --- Goemans-Williamson randomized rounding ---
    def goemans_williamson_init():
        degree = adj.sum(axis=1).astype(np.float64)
        D = np.diag(degree)
        L = D - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            k = min(5, n - 1)
            if k >= 2:
                Y = eigvecs[:, 1:k+1]
                norms = np.linalg.norm(Y, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                Y = Y / norms
                # Random hyperplane
                u = np.random.randn(k)
                u = u / np.linalg.norm(u)
                proj = Y @ u
                labels = (proj > 0).astype(np.int8)
            else:
                labels = np.random.randint(0, 2, size=n)
        except:
            labels = np.random.randint(0, 2, size=n)
        return labels

    # --- Greedy 1-flip local search (fast, vectorized per pass) ---
    def greedy_1flip(labels, max_passes=15):
        labels = labels.copy()
        for _ in range(max_passes):
            if time.time() > deadline - 2.0:
                break
            changed = False
            # Compute for all vertices at once: sum of neighbors in same vs opposite
            same_count = (adj * (labels[None, :] == labels[:, None])).sum(axis=1)
            opp_count = (adj * (labels[None, :] != labels[:, None])).sum(axis=1)
            flip_mask = same_count > opp_count
            if not np.any(flip_mask):
                break
            labels[flip_mask] = 1 - labels[flip_mask]
            changed = True
        return labels

    # --- 2-swap local search: flip pairs of vertices ---
    def fast_2swap(labels, max_passes=10):
        labels = labels.copy()
        for _ in range(max_passes):
            if time.time() > deadline - 2.0:
                break
            improved = False
            # Try a random subset of pairs each pass
            num_pairs = min(200, n * (n - 1) // 2)
            pairs = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(num_pairs)]
            for i, j in pairs:
                if i == j:
                    continue
                if labels[i] == labels[j]:
                    continue
                # Compute delta for flipping both i and j
                # Current: i and j are on opposite sides
                # After flip: i and j swap sides
                # Edges from i to same side as i: become crossing (+1)
                # Edges from i to opp side: become non-crossing (-1)
                # But j also flips, so edge (i,j) changes
                # Let's compute directly
                old_cut = compute_cut(labels)
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
                new_cut = compute_cut(labels)
                if new_cut > old_cut:
                    improved = True
                else:
                    # Revert
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
            if not improved:
                break
        return labels

    # --- Simulated Annealing with incremental updates ---
    def simulated_annealing(labels, T_initial, T_final, steps):
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_local_cut = current_cut
        best_local_labels = labels.copy()

        # Precompute sum_same and sum_opp
        sum_same = np.zeros(n, dtype=np.int32)
        sum_opp = np.zeros(n, dtype=np.int32)
        for v in range(n):
            same_mask = labels == labels[v]
            opp_mask = labels != labels[v]
            sum_same[v] = int(adj[v, same_mask].sum())
            sum_opp[v] = int(adj[v, opp_mask].sum())

        # Precompute neighbor lists
        neighbors = [np.where(adj[v] == 1)[0] for v in range(n)]

        if T_initial > 0 and T_final > 0:
            cf = (T_final / T_initial) ** (1.0 / steps)
        else:
            cf = 0.999

        T = T_initial
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break

            v = random.randint(0, n - 1)
            delta = sum_same[v] - sum_opp[v]

            accept = False
            if delta > 0:
                accept = True
            else:
                if T > 0:
                    ratio = np.exp(delta / T)
                    if random.random() < ratio:
                        accept = True

            if accept:
                old_label = labels[v]
                new_label = 1 - old_label
                labels[v] = new_label
                current_cut += delta

                sum_same[v], sum_opp[v] = sum_opp[v], sum_same[v]

                for u in neighbors[v]:
                    if labels[u] == new_label:
                        sum_same[u] += 1
                        sum_opp[u] -= 1
                    else:
                        sum_same[u] -= 1
                        sum_opp[u] += 1

                if current_cut > best_local_cut:
                    best_local_cut = current_cut
                    best_local_labels = labels.copy()

            T *= cf

        return best_local_labels, best_local_cut

    # --- Multi-start with diverse initializations ---
    # Phase 1: Spectral/GW init + greedy polish
    for r in range(8):
        if time.time() > deadline - 30.0:
            break
        if r % 2 == 0:
            labels = goemans_williamson_init()
        else:
            labels = spectral_init()
        polished = greedy_1flip(labels, max_passes=10)
        try_labels(polished)

    # Phase 2: SA from diverse starts
    num_sa = 15
    for r in range(num_sa):
        if time.time() > deadline - 15.0:
            break

        if r % 4 == 0:
            labels = np.random.randint(0, 2, size=n)
        elif r % 4 == 1:
            labels = goemans_williamson_init()
        elif r % 4 == 2:
            labels = np.arange(n) % 2
            flip_mask = np.random.random(n) < 0.3
            labels[flip_mask] = 1 - labels[flip_mask]
        else:
            labels = spectral_init()

        T_initial = n / 5.0
        T_final = 0.1
        steps = 1500

        sa_labels, sa_cut = simulated_annealing(labels, T_initial, T_final, steps)
        try_labels(sa_labels)

        # Polish with greedy
        if time.time() < deadline - 10.0:
            polished = greedy_1flip(sa_labels, max_passes=8)
            try_labels(polished)

    # Phase 3: 2-swap refinement on best
    if best_labels is not None and time.time() < deadline - 5.0:
        refined = fast_2swap(best_labels, max_passes=10)
        try_labels(refined)

        # Final greedy polish
        final = greedy_1flip(refined, max_passes=15)
        try_labels(final)

    # Final safety net
    if best_labels is None:
        labels = np.arange(n) % 2
        return labels

    return best_labels
# EVOLVE-BLOCK-END
