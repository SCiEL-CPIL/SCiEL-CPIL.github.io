"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def priority(instance):
    """Return one priority score per activity.

    Uses a critical-path / longest-remaining-tail heuristic: activities that
    have the most work downstream (longest path from the activity to a sink,
    including its own duration) get the highest priority. Ties are broken by
    larger raw duration, then by smaller index, to give a deterministic order.
    """
    t_start = time.time()
    DEADLINE = 140.0  # safety margin under 150s wall-clock kill

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # Build predecessor / successor lists.
    # instance["predecessors"] may be a list of lists (preds of each activity)
    # or a list of (i, j) edges. Handle common shapes defensively.
    preds_raw = instance.get("predecessors", None)

    succ = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)

    if preds_raw is not None:
        # Case A: list of lists where preds_raw[j] = list of predecessors of j
        try:
            # Detect if it's list of lists vs list of tuples
            first = preds_raw[0]
            is_list_of_lists = isinstance(first, (list, tuple, np.ndarray)) and not (
                isinstance(first, tuple) and len(first) == 2
                and all(isinstance(x, (int, np.integer)) for x in first)
                and len(preds_raw) > 0
                and isinstance(preds_raw[0], tuple) and len(preds_raw[0]) == 2
            )
        except Exception:
            is_list_of_lists = True

        if is_list_of_lists:
            for j in range(n):
                for p in preds_raw[j]:
                    p = int(p)
                    if 0 <= p < n and p != j:
                        succ[p].append(j)
                        in_degree[j] += 1
        else:
            # list of (pred, succ) edges
            for edge in preds_raw:
                p, s = int(edge[0]), int(edge[1])
                if 0 <= p < n and 0 <= s < n and p != s:
                    succ[p].append(s)
                    in_degree[s] += 1

    # Topological order via Kahn's algorithm.
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    indeg = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    # If topo doesn't cover all nodes (cycle), fall back to range(n).
    if len(topo) < n:
        topo = list(range(n))

    # Longest tail: longest path from node to any sink, including its own duration.
    tail = np.zeros(n, dtype=float)
    # Process in reverse topological order.
    for u in reversed(topo):
        d = durations[u]
        best_child = 0.0
        for v in succ[u]:
            if tail[v] > best_child:
                best_child = tail[v]
        tail[u] = d + best_child

    # Also compute head (longest path from any source to u, excluding u's duration)
    # to form a critical-path score. head[u] = max over preds of (head[p] + dur[p]).
    head = np.zeros(n, dtype=float)
    for u in topo:
        best_pred = 0.0
        for p in (preds_raw[u] if is_list_of_lists and preds_raw is not None and len(preds_raw) > u else []):
            p = int(p)
            cand = head[p] + durations[p]
            if cand > best_pred:
                best_pred = cand
        head[u] = best_pred

    # Priority: critical-path length = head + duration + tail.
    # Higher is more critical -> higher priority.
    crit = head + durations + tail

    # Build a stable, deterministic ordering key:
    # primary: crit descending, secondary: duration descending, tertiary: index ascending.
    # We produce a score array; the scheduler picks the highest score among eligible.
    # To make index break ties deterministically without distorting, subtract a tiny
    # epsilon scaled by index.
    max_crit = float(np.max(crit)) if n > 0 else 1.0
    if max_crit <= 0:
        max_crit = 1.0
    eps = max_crit * 1e-9
    score = crit.copy()
    # subtract tiny amount by index so lower index wins ties
    score = score - np.arange(n) * eps

    # Deadline guard: if we somehow overshot, return crit (already computed cheaply).
    if time.time() - t_start > DEADLINE:
        return np.asarray(crit, dtype=float)

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
