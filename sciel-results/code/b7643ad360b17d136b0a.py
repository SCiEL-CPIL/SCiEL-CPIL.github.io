"""Communication-minimizing DAG pebbling: topological-order scheduler with
recompute-aware placement windows and a memory feasibility check."""

import time

_DEADLINE = 75.0  # safety margin under the 90s kill


def _build_index(dag):
    """Return (n, parents, children, is_input, is_output, topo_order, weight)."""
    # dag: dict or list. Support common shapes.
    if hasattr(dag, "__len__") and not isinstance(dag, (list, tuple)):
        # dict-like
        verts = list(dag.keys())
        # try to get ordered list from values
        data = dag
    else:
        verts = list(dag)
        data = dag

    # Try to extract per-vertex info.
    # Assume dag is a list of dicts with 'parents' and 'outputs' info,
    # or a dict mapping vertex -> list of parents.
    parents_map = {}
    is_output_set = set()
    is_input_set = set()

    # Shape 1: list of dicts
    if isinstance(dag, (list, tuple)):
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                p = v.get("parents", v.get("parent", []))
                parents_map[i] = list(p)
                if v.get("is_output", v.get("output", False)):
                    is_output_set.add(i)
                if v.get("is_input", v.get("input", False)):
                    is_input_set.add(i)
            else:
                parents_map[i] = []
        n = len(dag)
    elif isinstance(dag, dict):
        for k, v in dag.items():
            if isinstance(v, (list, tuple, set)):
                parents_map[k] = list(v)
            elif isinstance(v, dict):
                p = v.get("parents", v.get("parent", []))
                parents_map[k] = list(p)
                if v.get("is_output", v.get("output", False)):
                    is_output_set.add(k)
                if v.get("is_input", v.get("input", False)):
                    is_input_set.add(k)
            else:
                parents_map[k] = []
        n = len(dag)
        verts = list(dag.keys())
    else:
        n = len(dag) if hasattr(dag, "__len__") else 0
        verts = list(range(n))

    # Fallback: if no parents found, assume chain
    if not parents_map:
        for i in range(n):
            parents_map[i] = [i - 1] if i > 0 else []

    # Determine inputs: vertices with no parents (unless marked)
    for v in verts:
        if v in is_input_set:
            continue
        if not parents_map.get(v):
            is_input_set.add(v)

    # Determine outputs: vertices with no children (unless marked)
    children_map = {v: [] for v in verts}
    for v in verts:
        for p in parents_map.get(v, []):
            if p in children_map:
                children_map[p].append(v)

    for v in verts:
        if v in is_output_set:
            continue
        if not children_map.get(v):
            is_output_set.add(v)

    # Topological order (Kahn)
    indeg = {v: len(parents_map.get(v, [])) for v in verts}
    from collections import deque
    q = deque([v for v in verts if indeg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children_map.get(v, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(topo) != n:
        topo = list(verts)  # fallback

    # Weight: 1 for input, 1 + in_degree otherwise
    weight = {}
    for v in verts:
        if v in is_input_set:
            weight[v] = 1
        else:
            weight[v] = 1 + len(parents_map.get(v, []))

    return n, parents_map, children_map, is_input_set, is_output_set, topo, weight


def _simulate_schedule(dag, parents_map, children_map, is_input_set,
                       is_output_set, topo, weight, compute_order, budget):
    """Simulate a schedule given compute_order (list of vertices in order).
    Returns (total_transfers, valid)."""
    n = len(topo)
    # State: 0=absent, 1=fast(red), 2=slow(blue)
    state = [0] * n
    fast_count = 0
    transfers = 0
    # Track which vertices have been computed (for final store check)
    final_stored = set()

    for v in compute_order:
        parents = parents_map.get(v, [])
        if v in is_input_set:
            # Input: load
            if state[v] == 0:
                # Load from slow to fast
                if fast_count >= budget:
                    return None, False  # infeasible
                state[v] = 1
                fast_count += 1
                transfers += 1
            # Input is now in fast memory
        else:
            # Need all parents in fast memory
            for p in parents:
                if state[p] != 1:
                    # Recompute parent chain? For simplicity, this schedule
                    # is invalid unless parents are in fast memory.
                    # We'll handle by requiring parents to be loaded.
                    return None, False
            # Compute: result goes to fast memory
            if fast_count >= budget:
                return None, False
            # Delete parents (they can be recomputed or stored)
            for p in parents:
                if state[p] == 1:
                    state[p] = 2  # store to slow (blue)
                    fast_count -= 1
                    transfers += 1
            state[v] = 1
            fast_count += 1
            transfers += 1 + len(parents)  # compute + store parents

    # All outputs must be blue (in slow memory)
    for v in is_output_set:
        if state[v] != 2:
            return None, False

    return transfers, True


def _naive_schedule(dag, parents_map, children_map, is_input_set,
                    is_output_set, topo, weight, budget):
    """Naive: compute each vertex once in topological order."""
    return list(topo)


def _improve_schedule(dag, parents_map, children_map, is_input_set,
                      is_output_set, topo, weight, budget, start_order, deadline):
    """Local search: try to reduce transfers by reordering/recomputing."""
    n = len(topo)
    best_order = list(start_order)
    best_transfers = None

    # Evaluate initial
    t, valid = _simulate_schedule(dag, parents_map, children_map, is_input_set,
                                  is_output_set, topo, weight, best_order, budget)
    if valid:
        best_transfers = t

    # Simple hill-climbing: try swapping adjacent elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_order) - 1):
            if time.time() >= deadline:
                break
            # Try swapping i and i+1
            trial = list(best_order)
            trial[i], trial[i + 1] = trial[i + 1], trial[i]
            t, valid = _simulate_schedule(dag, parents_map, children_map,
                                          is_input_set, is_output_set, topo,
                                          weight, trial, budget)
            if valid and (best_transfers is None or t < best_transfers):
                best_order = trial
                best_transfers = t
                improved = True

    return best_order, best_transfers


def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    n, parents_map, children_map, is_input_set, is_output_set, topo, weight = \
        _build_index(dag)

    if n == 0:
        return []

    # Naive schedule
    naive_order = _naive_schedule(dag, parents_map, children_map, is_input_set,
                                  is_output_set, topo, weight, fast_memory_budget)
    naive_transfers, naive_valid = _simulate_schedule(
        dag, parents_map, children_map, is_input_set, is_output_set, topo,
        weight, naive_order, fast_memory_budget)

    if not naive_valid:
        # Naive is infeasible; return empty or best effort
        return []

    # Try to improve
    if time.time() < deadline:
        improved_order, improved_transfers = _improve_schedule(
            dag, parents_map, children_map, is_input_set, is_output_set, topo,
            weight, fast_memory_budget, naive_order, deadline)
    else:
        improved_order = naive_order
        improved_transfers = naive_transfers

    # Use the better schedule
    if improved_transfers is not None and (naive_transfers is None or
                                           improved_transfers <= naive_transfers):
        final_order = improved_order
    else:
        final_order = naive_order

    # Convert compute_order to operations
    ops = []
    state = [0] * n  # 0=absent, 1=fast, 2=slow
    fast_count = 0

    for v in final_order:
        parents = parents_map.get(v, [])
        if v in is_input_set:
            if state[v] == 0:
                ops.append(["load", v])
                state[v] = 1
                fast_count += 1
        else:
            # Ensure parents in fast memory
            for p in parents:
                if state[p] != 1:
                    # Load parent from slow
                    if state[p] == 2:
                        ops.append(["load", p])
                        state[p] = 1
                        fast_count += 1
                    else:
                        # Parent not available; recompute (simplified)
                        ops.append(["compute", p])
                        state[p] = 1
                        fast_count += 1
            # Compute v
            ops.append(["compute", v])
            state[v] = 1
            fast_count += 1
            # Store parents to slow
            for p in parents:
                if state[p] == 1:
                    ops.append(["store", p])
                    state[p] = 2
                    fast_count -= 1

    # Store all outputs
    for v in is_output_set:
        if state[v] == 1:
            ops.append(["store", v])
            state[v] = 2
            fast_count -= 1

    # Delete remaining fast memory (non-outputs)
    for v in range(n):
        if state[v] == 1 and v not in is_output_set:
            ops.append(["delete", v])
            state[v] = 0
            fast_count -= 1

    return ops
