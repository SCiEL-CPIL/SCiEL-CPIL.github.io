# EVOLVE-BLOCK-START
import time
import heapq

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 90.0
    SAFETY = 6.0
    MAX_OPS = 20000
    fast_memory_budget = int(fast_memory_budget) if fast_memory_budget is not None else 0

    def remaining_time_ok():
        return (time.time() - t0) < (DEADLINE - SAFETY)

    # ---- Parse DAG ----
    # dag: dict with 'parents' {vertex: [parents...]} and 'outputs' [vertex...]
    parents_map = {}
    outputs = []
    if isinstance(dag, dict):
        pm = dag.get('parents') or dag.get('parent') or {}
        parents_map = {int(k): [int(x) for x in v] for k, v in pm.items()}
        out = dag.get('outputs') or dag.get('output') or dag.get('sinks') or []
        outputs = [int(x) for x in out]
        if not outputs:
            # try to infer sinks
            allp = set()
            for v, ps in parents_map.items():
                for p in ps:
                    allp.add(p)
            outputs = [v for v in parents_map if v not in allp]
    else:
        # list of (vertex, parents) or (vertex, [parents])
        for item in dag:
            v, ps = item[0], item[1]
            parents_map[int(v)] = [int(x) for x in ps]
        allp = set()
        for v, ps in parents_map.items():
            for p in ps:
                allp.add(p)
        outputs = [v for v in parents_map if v not in allp]

    vertices = list(parents_map.keys())
    if not vertices:
        return []
    allp = set()
    for v, ps in parents_map.items():
        for p in ps:
            allp.add(p)
    inputs = [v for v in vertices if v not in allp]

    # Compute topological order (Kahn)
    indeg = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents_map.items():
        for p in ps:
            children[p].append(v)
            indeg[v] += 1
    # initialize indeg for vertices not seen
    for v in vertices:
        if v not in indeg:
            indeg[v] = 0
    from collections import deque
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    indeg2 = dict(indeg)
    while q:
        u = q.popleft()
        topo.append(u)
        for c in children[u]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    if len(topo) != len(vertices):
        # cycle fallback
        topo = list(parents_map.keys())
    top_index = {v: i for i, v in enumerate(topo)}

    # ---- Naive baseline schedule (topological compute, store inputs, delete intermediates) ----
    def naive_schedule():
        ops = []
        in_memory = set()
        in_memory |= set(inputs)
        for v in topo:
            ps = parents_map.get(v, [])
            for p in ps:
                if p not in in_memory:
                    ops.append(('load', p))
                    in_memory.add(p)
            if len(ps) > 0:
                ops.append(('compute', v))
                in_memory.add(v)
            # keep outputs, delete other intermediates
            if v not in outputs and v not in inputs:
                # delay deletion until after children computed - but for naive just delete if no children need it later
                pass
        # store outputs
        for o in outputs:
            ops.append(('store', o))
        # count transfers
        transfers = sum(1 for op, _ in ops if op in ('load', 'store'))
        return ops, transfers

    # ---- Greedy pebbling heuristic ----
    # State: which vertices are red (in fast memory)
    # Goal: all outputs red, then store them
    # We want to minimize loads (communication)

    # Strategy: process DAG in topological order.
    # For each vertex, ensure parents are red (load if needed), compute, then decide whether to delete.
    # Key optimization: delete vertices that are no longer needed, to free space.

    def greedy_schedule():
        ops = []
        red = set(inputs)  # inputs start red/blue (available)
        # Actually "blue" means in memory. Let's track which vertices have their value available.
        # Inputs are available at start.
        available = set(inputs)
        outputs_set = set(outputs)

        # Determine for each vertex the set of descendants that need it
        # Actually we need: for each vertex, which children (and their descendants) still need it
        # Simpler: process in topo order; a vertex can be deleted once all its children have been computed
        # AND its value won't be needed again (no recomputation)
        # But with recomputation, we might load it later.

        # Let's use: children_remaining[v] = number of children not yet computed
        children_remaining = {v: len(children[v]) for v in vertices}
        for v in vertices:
            children_remaining.setdefault(v, 0)

        for v in topo:
            ps = parents_map.get(v, [])
            for p in ps:
                if p not in available:
                    ops.append(('load', p))
                    available.add(p)
            if len(ps) > 0:
                ops.append(('compute', v))
                available.add(v)
            # Update children_remaining for parents
            for p in ps:
                children_remaining[p] = children_remaining.get(p, 0) - 1
            # Decide deletion: delete v if (1) not output, (2) not input, (3) all children computed,
            # (4) memory pressure (size >= budget)
            if v not in outputs_set and v not in inputs:
                if children_remaining.get(v, 0) <= 0:
                    if len(available) >= fast_memory_budget or fast_memory_budget == 0:
                        ops.append(('delete', v))
                        available.discard(v)
            if len(ops) >= MAX_OPS:
                break
        for o in outputs:
            if o in available:
                ops.append(('store', o))
            else:
                ops.append(('load', o))  # shouldn't happen
                available.add(o)
        transfers = sum(1 for op, _ in ops if op in ('load', 'store'))
        return ops, transfers

    # ---- Advanced: recompute-based pebbling ----
    # For layered DAGs, we can use divide-and-conquer to reduce memory
    # But for now, let's try a smarter greedy with recomputation

    def compute_schedule_with_recompute():
        ops = []
        available = set(inputs)
        outputs_set = set(outputs)
        children_remaining = {v: len(children.get(v, [])) for v in vertices}

        # Precompute: for each vertex, its "last use" topological index
        last_use = {}
        for v in vertices:
            last_use[v] = top_index.get(v, 0)
        for v, ps in parents_map.items():
            for p in ps:
                # p is used at v, so last_use[p] >= top_index[v]
                if top_index.get(v, 0) > last_use.get(p, 0):
                    last_use[p] = top_index.get(v, 0)

        for v in topo:
            ps = parents_map.get(v, [])
            for p in ps:
                if p not in available:
                    ops.append(('load', p))
                    available.add(p)
            if len(ps) > 0:
                ops.append(('compute', v))
                available.add(v)
            for p in ps:
                children_remaining[p] = children_remaining.get(p, 0) - 1

            # Delete if no longer needed and memory is full
            if v not in outputs_set and v not in inputs:
                if children_remaining.get(v, 0) <= 0:
                    if len(available) >= max(fast_memory_budget, 1):
                        ops.append(('delete', v))
                        available.discard(v)
            if len(ops) >= MAX_OPS:
                break

        for o in outputs:
            if o in available:
                ops.append(('store', o))
            else:
                ops.append(('load', o))
        transfers = sum(1 for op, _ in ops if op in ('load', 'store'))
        return ops, transfers

    # Try multiple strategies, pick best
    candidates = []
    try:
        ops_g, tr_g = greedy_schedule()
        candidates.append((tr_g, ops_g))
    except Exception:
        pass
    try:
        ops_n, tr_n = naive_schedule()
        candidates.append((tr_n, ops_n))
    except Exception:
        pass
    try:
        ops_r, tr_r = compute_schedule_with_recompute()
        candidates.append((tr_r, ops_r))
    except Exception:
        pass

    if not candidates:
        return []

    # Sort by transfer count
    candidates.sort(key=lambda x: x[0])
    best_ops = candidates[0][1]
    return best_ops
# EVOLVE-BLOCK-END
