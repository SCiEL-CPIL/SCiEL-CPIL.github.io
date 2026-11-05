"""RCPSP priority heuristic: combine CPM total float, bottom level, and resource pressure."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_MARGIN = 10.0  # seconds of safety margin before the 150s wall-clock kill


def _build_graph(instance):
    """Return n, preds (list of predecessor lists), succs (list of successor lists)."""
    n = int(instance["n"]) if "n" in instance else int(np.max(instance["durations"].shape) and instance["durations"].shape[0])
    preds = [[] for _ in range(n)]
    succs = [[] for _ in range(n)]
    if "predecessors" in instance:
        pred_data = instance["predecessors"]
    else:
        pred_data = instance.get("preds", instance.get("precedence", None))
    if pred_data is not None:
        for i in range(n):
            for p in pred_data[i]:
                p = int(p)
                if p != i:
                    preds[i].append(p)
                    succs[p].append(i)
    return n, preds, succs


def _compute_float_and_levels(instance, n, preds, succs):
    """Compute durations, bottom level, top level, and total float per activity."""
    durations = np.asarray(instance["durations"], dtype=float)
    # Bottom level: longest path from activity i to any sink (including its own duration).
    # Top level: longest path from any source to activity i (including its own duration).
    bottom = np.zeros(n, dtype=float)
    top = np.zeros(n, dtype=float)

    # Top level via topological order (Kahn).
    in_deg = np.array([len(preds[i]) for i in range(n)], dtype=int)
    from collections import deque
    q = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    indeg_work = in_deg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                q.append(v)
    # If graph has a cycle, fall back to natural order.
    if len(topo) != n:
        topo = list(range(n))

    # top[i] = max over preds of (top[p] + dur[p]) + dur[i]; sources have top = dur[i]
    for u in topo:
        best = 0.0
        for p in preds[u]:
            cand = top[p] + durations[p]
            if cand > best:
                best = cand
        top[u] = best + durations[u]

    # bottom[i] = max over succs of (bottom[s]) + dur[i]; sinks have bottom = dur[i]
    for u in reversed(topo):
        best = 0.0
        for s in succs[u]:
            cand = bottom[s]
            if cand > best:
                best = cand
        bottom[u] = best + durations[u]

    # Project makespan (critical path length ignoring resources).
    makespan_cpm = float(np.max(top)) if n > 0 else 0.0

    # Total float = makespan_cpm - top[i] - bottom[i] + dur[i]
    float_arr = makespan_cpm - top - bottom + durations

    return durations, top, bottom, float_arr, makespan_cpm


def priority(instance):
    start = time.time()
    try:
        n, preds, succs = _build_graph(instance)
        durations, top, bottom, float_arr, makespan_cpm = _compute_float_and_levels(instance, n, preds, succs)

        # Resource demand per activity (vector across resources).
        if "demands" in instance:
            demands = np.asarray(instance["demands"], dtype=float)
            if demands.ndim == 1:
                demands = demands.reshape(n, -1)
        else:
            demands = np.zeros((n, 1), dtype=float)

        if "capacities" in instance:
            capacities = np.asarray(instance["capacities"], dtype=float)
            if capacities.ndim == 0:
                capacities = np.full(demands.shape[1], float(capacities))
            else:
                capacities = np.asarray(capacities, dtype=float).ravel()
            # Ensure capacity vector matches demand columns.
            if capacities.size < demands.shape[1]:
                capacities = np.resize(capacities, demands.shape[1])
        else:
            capacities = np.ones(demands.shape[1], dtype=float)
            if capacities.size < demands.shape[1]:
                capacities = np.ones(demands.shape[1])

        # Normalized resource pressure: max demand-to-capacity ratio per activity.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = demands / capacities
            ratio = np.where(np.isfinite(ratio), ratio, 0.0)
        pressure = np.max(ratio, axis=1)

        # Normalize components to comparable scales.
        def _norm(x):
            x = np.asarray(x, dtype=float)
            mn, mx = np.min(x), np.max(x)
            if mx - mn < 1e-12:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        # Total float: lower float = more critical = higher priority.
        # We want high priority for low float, so invert.
        float_score = 1.0 - _norm(float_arr)
        # Bottom level: larger = closer to end, more critical.
        bottom_score = _norm(bottom)
        # Top level: larger = later start on critical path.
        top_score = _norm(top)
        # Resource pressure: higher demand pressure = schedule sooner to free resources.
        pressure_score = _norm(pressure)
        # Duration: longer activities benefit from early start (to overlap).
        dur_score = _norm(durations)

        # Weighted combination emphasizing float (critical path) and bottom level.
        score = (
            0.45 * float_score
            + 0.25 * bottom_score
            + 0.15 * top_score
            + 0.10 * pressure_score
            + 0.05 * dur_score
        )

        # Tie-break: add a tiny deterministic jitter based on index to avoid
        # arbitrary ordering among ties (stable, reproducible).
        score = score + 1e-9 * np.arange(n, dtype=float)

        if time.time() - start > _DEADLINE_MARGIN:
            # Fallback: return float-based priority quickly.
            return float_score

        return np.asarray(score, dtype=float)
    except Exception:
        # Safe fallback: longest duration first.
        try:
            return np.asarray(instance["durations"], dtype=float)
        except Exception:
            return np.zeros(1, dtype=float)
# EVOLVE-BLOCK-END
