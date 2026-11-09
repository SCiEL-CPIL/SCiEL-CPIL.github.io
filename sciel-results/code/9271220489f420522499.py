"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    # Ensure symmetric (just in case)
    adj = 0.5 * (adj + adj.T)

    deadline = time.time() + 110.0  # safety margin before 120s

    best_labels = None
    best_cut = -1

    def cut_size(labels):
        # labels: 0/1 array
        # cut = number of edges crossing
        # = sum over v of (neighbors of v in opposite partition)
        # = sum_v sum_u adj[v,u] * (labels[v] != labels[u])
        # Efficient: for each v, count neighbors NOT in same set
        # = sum_v (row_sum[v] - sum_u where labels[u]==labels[v] adj[v,u])
        # But simpler vectorized:
        # cut = 0.5 * (sum over v,u of adj[v,u] * (labels[v] - labels[u])^2) ... no
        # Actually: cut = sum_{v<u} adj[v,u] * |labels[v]-labels[u]|
        # = 0.5 * sum_{v,u} adj[v,u] * (labels[v] != labels[u])
        # Using: (labels @ adj @ (1-labels)) counts each crossing edge twice? 
        # labels @ adj @ (1-labels) = sum_v labels[v] * sum_u adj[v,u]*(1-labels[u])
        # = sum over directed edges v->u where label differs = 2 * cut
        # So cut = 0.5 * labels @ adj @ (1-labels)
        return 0.5 * float(labels @ adj @ (1 - labels))

    def spectral_init(rng):
        # Use eigenvector of adjacency (largest eigenvalue) for initial partition
        try:
            # For max cut, we want to maximize x^T A x with x in {+1,-1}
        # Spectral: use eigenvector corresponding to largest eigenvalue of A
            w, V = np.linalg.eigh(adj)
            # Largest eigenvalue is last
            vec = V[:, -1]
            # Threshold at median to balance partitions
            med = np.median(vec)
            labels = (vec > med).astype(np.float64)
            # If all same, randomize
            if labels.sum() == 0 or labels.sum() == n:
                labels = (rng.random(n) > 0.5).astype(np.float64)
            return labels
        except Exception:
            return (rng.random(n) > 0.5).astype(np.float64)

    def random_init(rng):
        return (rng.random(n) > 0.5).astype(np.float64)

    def greedy_flip(labels):
        # One pass of greedy single-flip improvements
        labels = labels.copy()
        # Compute for each vertex: gain from flipping
        # gain[v] = (neighbors in opposite) - (neighbors in same)
        # = 2*(neighbors in opposite) - degree[v]
        # neighbors in opposite for v = sum_u adj[v,u]*(1-labels[u]) if labels[v]=0
        # Let's compute: for v, same = sum_u adj[v,u]*labels[u] (if labels[v]=1) 
        # Actually: 
        #   if labels[v]=0: opposite neighbors = sum_u adj[v,u]*labels[u]
        #   if labels[v]=1: opposite neighbors = sum_u adj[v,u]*(1-labels[u])
        # gain from flipping = opposite - same = 2*opposite - degree
        # Let's just do iterative:
        changed = True
        max_iter = 50
        for it in range(max_iter):
            if time.time() > deadline:
                break
            any_change = False
            # Vectorized: compute for each v the number of neighbors in each partition
            # same_count[v] = sum_u adj[v,u] * (labels[u] == labels[v])
            # We can compute: for each v, sum of adj[v] over u where labels[u]==labels[v]
            # = (adj @ labels)[v] if labels[v]=1, else (adj @ (1-labels))[v]
            # Let's compute both:
            sum_in_1 = adj @ labels       # for each v: sum of adj[v,u] for u with label 1
            sum_in_0 = adj @ (1 - labels)  # for each v: sum of adj[v,u] for u with label 0
            # For vertex v with label 1: same = sum_in_1[v], opposite = sum_in_0[v]
            # For vertex v with label 0: same = sum_in_0[v], opposite = sum_in_1[v]
            # gain[v] = opposite - same
            gain = np.where(labels == 1, sum_in_0 - sum_in_1, sum_in_1 - sum_in_0)
            # Flip vertices where gain > 0
            to_flip = gain > 0
            if not np.any(to_flip):
                break
            labels[to_flip] = 1 - labels[to_flip]
            any_change = True
            if not any_change:
                break
        return labels

    def tabu_search(labels, rng, tabu_len):
        # Tabu search with aspiration
        labels = labels.copy()
        cur_cut = cut_size(labels)
        best_local_cut = cur_cut
        best_local_labels = labels.copy()
        # Tabu list: store iteration at which a vertex was last flipped
        tabu_until = np.zeros(n, dtype=int)
        aspiration = True
        it = 0
        max_iters = 2000
        while it < max_iters:
            if time.time() > deadline:
                break
            it += 1
            # Compute gain for flipping each vertex
            sum_in_1 = adj @ labels
            sum_in_0 = adj @ (1 - labels)
            gain = np.where(labels == 1, sum_in_0 - sum_in_1, sum_in_1 - sum_in_0)
            # Tabu mask: vertices that are tabu (can't flip unless aspiration)
            is_tabu = tabu_until > it
            # Aspiration: allow tabu move if it improves best
            # For simplicity, allow if gain > (best_local_cut - cur_cut) threshold
            # Actually standard: allow if new cut > best_local_cut
            # Let's compute new cut for each candidate flip:
            # If we flip v: new_cut = cur_cut + gain[v]
            # Aspiration: allow tabu v if cur_cut + gain[v] > best_local_cut
            cand_gain = gain.copy()
            # Mask tabu vertices that don't meet aspiration
            if np.any(is_tabu):
                aspiration_ok = (cur_cut + gain) > best_local_cut
                cand_gain[is_tabu & ~aspiration_ok] = -np.inf
            # Choose best candidate
            best_v = int(np.argmax(cand_gain))
            if cand_gain[best_v] <= 0 and not (aspiration and cur_cut + gain[best_v] > best_local_cut):
                # No improving or aspirating move; try random
                candidates = np.where(np.isfinite(cand_gain))[0]
                if len(candidates) == 0:
                    break
                best_v = int(rng.choice(candidates))
            # Perform flip
            labels[best_v] = 1 - labels[best_v]
            tabu_until[best_v] = it + tabu_len
            new_cut = cur_cut + gain[best_v]
            cur_cut = new_cut
            if cur_cut > best_local_cut:
                best_local_cut = cur_cut
                best_local_labels = labels.copy()
        return best_local_labels, best_local_cut

    # Multi-start
    rng = np.random.default_rng(42)
    num_starts = 8

    # First start: spectral
    labels = spectral_init(rng)
    labels = greedy_flip(labels)
    c = cut_size(labels)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()

    # Tabu search from best
    if time.time() < deadline:
        tl, tc = tabu_search(best_labels, rng, tabu_len=12)
        if tc > best_cut:
            best_cut = tc
            best_labels = tl

    # More starts with varied tabu lengths
    tabu_lengths = [8, 10, 12, 14, 16, 18, 21]
    for i in range(1, num_starts):
        if time.time() > deadline:
            break
        if i % 2 == 0:
            labels = random_init(rng)
        else:
            labels = spectral_init(rng)
        labels = greedy_flip(labels)
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
        if time.time() < deadline:
            tl = tabu_lengths[i % len(tabu_lengths)]
            tl_labels, tc = tabu_search(labels, rng, tabu_len=tl)
            if tc > best_cut:
                best_cut = tc
                best_labels = tl_labels

    # Final polish: greedy from best
    if time.time() < deadline:
        polished = greedy_flip(best_labels)
        pc = cut_size(polished)
        if pc > best_cut:
            best_cut = pc
            best_labels = polished

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
