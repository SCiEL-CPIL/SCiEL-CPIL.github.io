"""DAG pebbling: communication-minimizing schedule with bounded fast memory."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    import time
    from collections import deque

    t0 = time.time()
    DEADLINE = 85.0  # safety margin before the 90s wall-clock kill
    MAX_OPS = 20000

    # ---------- Parse the DAG ----------
    parents = {}
    if isinstance(dag, dict):
        for v, p in dag.items():
            if isinstance(p, (list, tuple, set)):
                parents[v] = list(p)
            elif p is None:
                parents[v] = []
            else:
                parents[v] = [p]
        vertices = list(parents.keys())
    elif isinstance(dag, (list, tuple)):
        # Could be [(v, [parents]), ...] or (vertices, edges, ...)
        if len(dag) == 2 and isinstance(dag[0], (list, tuple)) and isinstance(dag[1], (list, tuple)):
            # (vertices, edges) where edges is list of (parent, child)
            verts, edges = dag
            parents = {v: [] for v in verts}
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    a, b = e
                    if b not in parents:
                        parents[b] = []
                    if a not in parents:
                        parents[a] = []
                    parents[b].append(a)
            vertices = list(parents.keys())
        else:
            for item in dag:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    v, p = item
                    if isinstance(p, (list, tuple, set)):
                        parents[v] = list(p)
                    elif p is None:
                        parents[v] = []
                    else:
                        parents[v] = [p]
            vertices = list(parents.keys())
    else:
        # Last resort: empty
        vertices = []

    if fast_memory_budget is None or fast_memory_budget < 1:
        fast_memory_budget = max(1, len(vertices))

    # ---------- Build children ----------
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p not in children:
                children[p] = []
            children[p].append(v)

    inputs = [v for v in vertices if not parents[v]]
    outputs = [v for v in vertices if not children[v]]
    output_set = set(outputs)

    # ---------- Topological order ----------
    in_deg = {v: len(parents[v]) for v in vertices}
    q = deque([v for v in vertices if in_deg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    if len(topo) != len(vertices):
        seen = set(topo)
        topo = topo + [v for v in vertices if v not in seen]
    topo_index = {v: i for i, v in enumerate(topo)}

    # Children sorted by topological position (for next-use lookup)
    children_topo = {
        v: sorted(children[v], key=lambda c: topo_index.get(c, 10 ** 9))
        for v in vertices
    }

    # ---------- Pebbling state ----------
    in_memory = set(inputs)
    # If inputs exceed budget, we keep them all initially (they start blue);
    # eviction logic will trim as needed.
    ptr = {v: 0 for v in vertices}          # children computed so far
    ref = {v: len(children_topo[v]) for v in vertices}  # uncomputed children
    ops = []
    computed = set()
    current_pos = 0

    BIG = 10 ** 9

    def next_use_dist(v):
        # Outputs must persist to the end -> keep them (huge distance).
        if v in output_set:
            return BIG * 10
        if ptr[v] >= len(children_topo[v]):
            return BIG
        c = children_topo[v][ptr[v]]
        d = topo_index.get(c, BIG) - current_pos
        return max(0, d)

    def try_evict(exclude):
        # Evict the in-memory value with the furthest next use.
        cand = [x for x in in_memory if x not in exclude]
        if not cand:
            return None
        cand.sort(key=lambda x: next_use_dist(x), reverse=True)
        ev = cand[0]
        ops.append(['store', ev])
        in_memory.discard(ev)
        return ev

    # ---------- Main schedule ----------
    for v in topo:
        if time.time() - t0 > DEADLINE or len(ops) >= MAX_OPS:
            break
        current_pos = topo_index.get(v, 0)

        # Ensure all parents are resident (blue)
        for p in parents[v]:
            if p not in in_memory:
                if len(in_memory) >= fast_memory_budget:
                    try_evict(exclude=set(parents[v]) | {v})
                ops.append(['load', p])
                in_memory.add(p)

        # Compute v (result lands in fast memory)
        ops.append(['compute', v])
        in_memory.add(v)
        computed.add(v)

        # Update parents: decrement ref, delete when no longer needed
        for p in parents[v]:
            ptr[p] += 1
            ref[p] -= 1
            if ref[p] == 0 and p not in output_set and p in in_memory:
                ops.append(['delete', p])
                in_memory.discard(p)

    # ---------- Ensure all outputs finish blue ----------
    for o in outputs:
        if time.time() - t0 > DEADLINE or len(ops) >= MAX_OPS:
            break
        if o not in in_memory:
            if len(in_memory) >= fast_memory_budget:
                try_evict(exclude=output_set)
            ops.append(['load', o])
            in_memory.add(o)

    return ops
# EVOLVE-BLOCK-END
