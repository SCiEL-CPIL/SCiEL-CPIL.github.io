"""Rewritten program for Maximum Cut task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using fast multi-start local search.

    Key optimizations:
    - O(n) incremental cut evaluation via per-vertex neighbor-sum tracking
      (avoids the O(n^2) boolean-matrix recomputation per step).
    - Multi-pass greedy (2-opt) sweep to convergence for strong local optima.
    - Many cheap random restarts within the time budget.
    - Simulated annealing (with fast delta) only when time remains.
    """
    start_time = time.time()
    DEADLINE = 145.0  # evaluator kills at 150s; leave 5s safety margin

    def _elapsed():
        return time.time() - start_time

    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    # Convert to numpy bool matrix; build integer adjacency for fast sums.
    adj = np.array(adjacency, dtype=bool)
    adj_int = adj.astype(np.int32)
    # Symmetrize defensively (in case input is directed representation).
    if not np.array_equal(adj, adj.T):
        adj = adj | adj.T
        adj_int = adj.astype(np.int32)

    def _init_neighbor_sum(labels, adj_int):
        """For each vertex, count neighbors with the SAME label as the vertex."""
        # sum over neighbors of (neighbor_label == vertex_label)
        # = (adj_int @ labels) where labels is 0/1, but only same-side:
        # same-side neighbors = adj & (labels[None,:] == labels[:,None])
        # Fast: for vertex v, same-side neighbor count =
        #   sum_u adj[v,u] * (labels[u] == labels[v])
        # = (adj_int @ labels)[v] if labels[v]==1 else (deg[v] - (adj_int@labels)[v])
        deg = adj_int.sum(axis=1)
        cross_to_one = adj_int @ labels  # sum of labels[u] over neighbors u
        # same-side count:
        same = np.where(labels == 1, cross_to_one, deg - cross_to_one)
        return same.astype(np.int32)

    def _cut_from_neighbor_sum(same):
        """Cut size = total_edges - sum(same)/2, since each same-side edge counted twice."""
        total_edges = int(adj_int.sum()) // 2
        return total_edges - int(same.sum()) // 2

    def _greedy_converge(labels, adj_int, deg, same):
        """
        Multi-pass greedy sweep to convergence.
        Flips vertex v if it increases the cut.
        Returns (labels, same, cut).
        """
        labels = labels.copy()
        same = same.copy()
        cut = _cut_from_neighbor_sum(same)
        for _ in range(50):  # cap passes; convergence usually much faster
            if _elapsed() > DEADLINE:
                break
            improved = False
            for v in range(n):
                # delta = gain - loss = same[v] - (deg[v] - same[v]) = 2*same[v] - deg[v]
                # Wait: flipping v changes which side v is on.
                # Currently same[v] = neighbors on same side as v.
                # After flip, those same-side neighbors become crossing (gain = same[v]),
                # and the other-side neighbors (deg[v]-same[v]) become same (loss).
                delta = same[v] - (deg[v] - same[v])  # = 2*same[v] - deg[v]
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    cut += delta
                    # Update same for v and its neighbors.
                    new_same_v = deg[v] - same[v]
                    same[v] = new_same_v
                    # For each neighbor u of v, u's same count changes by
                    # -1 if u was on same side as old v (now different), +1 otherwise.
                    # After flip, v is on opposite side. For neighbor u:
                    #   before: same-side relation (labels[u]==labels[v_old])
                    #   after:  (labels[u]==labels[v_new]) = not previous
                    # So same[u] += 1 if they were same before, else -1.
                    nbrs = np.where(adj_int[v] > 0)[0]
                    for u in nbrs:
                        if labels[u] == labels[v]:  # now same side
                            same[u] += 1
                        else:
                            same[u] -= 1
                    improved = True
            if not improved:
                break
        return labels, same, cut

    def _sa(labels, same, cut, adj_int, deg, iters=200):
        """Simulated annealing using fast delta. Returns best found."""
        labels = labels.copy()
        same = same.copy()
        cur_cut = cut
        best_labels = labels.copy()
        best_same = same.copy()
        best_cut = cut
        T0, T1 = 2.0, 0.05
        for i in range(iters):
            if _elapsed() > DEADLINE:
                break
            T = T0 * (T1 / T0) ** (i / max(iters, 1))
            v = np.random.randint(n)
            delta = same[v] - (deg[v] - same[v])
            if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-12)):
                # accept
                labels[v] = 1 - labels[v]
                cur_cut += delta
                new_same_v = deg[v] - same[v]
                same[v] = new_same_v
                nbrs = np.where(adj_int[v] > 0)[0]
                for u in nbrs:
                    if labels[u] == labels[v]:
                        same[u] += 1
                    else:
                        same[u] -= 1
                if cur_cut > best_cut:
                    best_cut = cur_cut
                    best_labels = labels.copy()
                    best_same = same.copy()
            # else reject (no state change)
        return best_labels, best_same, best_cut

    # Multi-start.
    best_labels = None
    best_cut = -1

    rng = np.random.default_rng(12345)

    def _try(labels0):
        nonlocal best_labels, best_cut
        if _elapsed() > DEADLINE:
            return
        same = _init_neighbor_sum(labels0, adj_int)
        labels, same, cut = _greedy_converge(labels0, adj_int, deg, same)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Deterministic starts.
    _try(np.arange(n) % 2)
    _try(np.ones(n, dtype=int))
    _try(np.zeros(n, dtype=int))

    # Random restarts (cheap due to fast greedy).
    while _elapsed() < DEADLINE - 2.0:
        labels0 = rng.integers(0, 2, size=n).astype(int)
        _try(labels0)

    # If time remains, polish the best solution with simulated annealing.
    if best_labels is not None and _elapsed() < DEADLINE - 1.0:
        same = _init_neighbor_sum(best_labels, adj_int)
        cut = _cut_from_neighbor_sum(same)
        sa_labels, sa_same, sa_cut = _sa(best_labels, same, cut, adj_int, deg, iters=2000)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()

    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
