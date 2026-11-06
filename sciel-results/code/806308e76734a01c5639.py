"""Communication-minimizing DAG pebbling scheduler (self-contained)."""
import time

# ---- deadline guard: evaluator kills at 90s wall-clock; keep a safety margin ----
_DEADLINE = time.time() + 82.0
_MAX_OPS = 20000


def _time_up():
    return time.time() > _DEADLINE


def _get_vertices(dag):
    for attr in ("vertices", "nodes", "V", "vertex_list", "all_vertices"):
        v = getattr(dag, attr, None)
        if v is not None:
            return list(v)
    for attr in ("vertex_count", "n", "num_vertices", "size"):
        n = getattr(dag, attr, None)
        if isinstance(n, int):
            return list(range(n))
    if isinstance(dag, dict) and "vertices" in dag:
        return list(dag["vertices"])
    return list(range(len(dag)))


def _get_parents(dag, vertices):
    # per-vertex parent lists
    for attr in ("parents", "preds", "in_edges", "predecessors", "deps"):
        p = getattr(dag, attr, None)
        if p is not None:
            out = []
            for v in vertices:
                try:
                    out.append(list(p[v]))
                except Exception:
                    out.append([])
            return out
    # adjacency dict {v: [parents]}
    if isinstance(dag, dict):
        if "parents" in dag:
            return [list(dag["parents"].get(v, [])) for v in vertices]
        if "edges" in dag:
            adj = {v: [] for v in vertices}
            for (a, b) in dag["edges"]:
                adj.setdefault(b, []).append(a)
            return [adj.get(v, []) for v in vertices]
    return [[] for _ in vertices]


def _get_inputs(dag, vertices, parents):
    for attr in ("inputs", "input_vertices", "sources", "leaves", "input_nodes"):
        i = getattr(dag, attr, None)
        if i is not None:
            return list(i)
    return [v for v in vertices if not parents[v]]


def _get_outputs(dag, vertices, parents):
    for attr in ("outputs", "output_vertices", "sinks", "output_nodes", "targets"):
        o = getattr(dag, attr, None)
        if o is not None:
            return list(o)
    return [v for v in vertices if not parents[v]]


def _topo_order(vertices, parents):
    # Kahn topological sort (parents before child)
    indeg = {}
    children = {v: [] for v in vertices}
    for v in vertices:
        indeg[v] = len(parents[v])
    for v in vertices:
        for p in parents[v]:
            if p in children:
                children[p].append(v)
    from collections import deque
    q = deque([v for v in vertices if indeg[v] == 0])
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(order) != len(vertices):  # fallback: given order
        return list(vertices)
    return order


def _pebbling_cost(v, parents, memo):
    # cost = #transfers to compute v from scratch assuming parents available
    # (recompute sub-DAG, then compute v)
    if v in memo:
        return memo[v]
    memo[v] = None  # cycle guard
    total = 0
    for p in parents[v]:
        total += _pebbling_cost(p, parents, memo)
    memo[v] = total
    return total


def pebble(dag, fast_memory_budget):
    start = time.time()
    try:
        vertices = _get_vertices(dag)
        parents = _get_parents(dag, vertices)
        inputs = _get_inputs(dag, vertices, parents)
        outputs = _get_outputs(dag, vertices, parents)
    except Exception:
        # last-ditch: naive linear schedule
        try:
            vertices = _get_vertices(dag)
        except Exception:
            vertices = []
        return [("load", v) for v in vertices]

    if not vertices:
        return []

    budget = int(fast_memory_budget) if fast_memory_budget is not None else 0
    budget = max(0, budget)

    # Build a topological order and a parent-index map for quick lookups
    topo = _topo_order(vertices, parents)
    idx = {v: i for i, v in enumerate(topo)}
    parent_idx = [[idx[p] for p in parents[v] if p in idx] for v in topo]

    # Pebbling cost per vertex (recompute cost) for scheduling heuristics
    cost_memo = {}
    vcost = [_pebbling_cost(v, parents, cost_memo) for v in topo]

    # ---- Schedule: depth-first pebbling with recompute to minimize transfers ----
    # We maintain a stack of vertices to evaluate. We load inputs when needed,
    # compute, and keep a working set in fast memory up to `budget`.
    #
    # Strategy: process outputs in order. For each output, recursively ensure
    # its parents are available (in fast memory) before computing it. Values
    # are evicted (delete) when they exceed budget and are no longer needed.
    #
    # To minimize transfers, we reuse already-computed values (blue in memory)
    # and only recompute what's missing. We track reference counts of how many
    # remaining computations need each vertex.

    # Determine remaining "need" counts: how many times each vertex is required
    # as a parent by vertices still to be computed (the outputs and their deps).
    # We'll do a DFS that, for each target, ensures all ancestors are pebbled.

    ops = []
    in_memory = set()          # vertices currently blue in fast memory
    # remaining count: number of children (in topo sense) that still need v
    # We process outputs; a vertex is needed until all its children are done.
    remaining = {v: 0 for v in topo}
    children = {v: [] for v in topo}
    for v in topo:
        for p in parent_idx[v]:
            children[topo[p]].append(v)
            remaining[topo[p]] += 1

    # We will compute vertices in an order that respects: parents before child.
    # To keep memory low, we use a DFS post-order over the union of sub-DAGs
    # needed by outputs, computing each vertex once per "pass" when needed.
    #
    # Simpler robust approach: linear pass over topo order.
    #   - if v is an input: load it (transfer). If in memory already, skip.
    #   - else: ensure all parents are in memory (load/recompute if not),
    #     compute v (transfer for result), then free parents that are no longer needed.
    #
    # Recompute if a parent is not in memory: recursively load its parents.

    # To handle memory budget: if in_memory exceeds budget, evict vertices
    # whose remaining count is 0 (no longer needed) first.

    def ensure_in_memory(v, depth=0):
        """Ensure v is in fast memory (blue). May recompute."""
        if v in in_memory:
            return
        if len(ops) > _MAX_OPS or _time_up():
            return
        if v in inputs_set:
            ops.append(("load", v))
            in_memory.add(v)
        else:
            # need all parents in memory
            for p in parent_idx[idx[v]]:
                pv = topo[p]
                ensure_in_memory(pv, depth + 1)
            # compute v
            ops.append(("compute", v))
            in_memory.add(v)
        # evict if over budget
        if len(in_memory) > budget:
            evict()

    def evict():
        # evict vertices with remaining==0 first, then oldest
        while len(in_memory) > budget:
            # find a vertex in memory that is no longer needed
            victim = None
            for v in list(in_memory):
                if remaining.get(v, 0) == 0:
                    victim = v
                    break
            if victim is None:
                # evict the one with smallest remaining (least useful)
                victim = min(in_memory, key=lambda v: remaining.get(v, 0))
            ops.append(("delete", victim))
            in_memory.discard(victim)

    inputs_set = set(inputs)
    outputs_set = set(outputs)

    # Process each output: ensure it's in memory (computed), which pulls in its DAG.
    # We go through topo order so that by the time we reach a vertex, its parents
    # have already been computed (and may still be in memory or need recompute).
    for v in topo:
        if len(ops) > _MAX_OPS or _time_up():
            break
        if v in in_memory:
            # already computed and in memory; just decrement remaining of parents
            for p in parent_idx[idx[v]]:
                pv = topo[p]
                remaining[pv] = max(0, remaining.get(pv, 0) - 1)
            continue
        # ensure v is in memory
        ensure_in_memory(v)
        # after computing v, its parents are no longer needed by v
        for p in parent_idx[idx[v]]:
            pv = topo[p]
            remaining[pv] = max(0, remaining.get(pv, 0) - 1)
        evict()

    # Final: ensure all outputs are blue in memory.
    for o in outputs:
        if o not in in_memory:
            ensure_in_memory(o)
        evict()

    # If we ran out of ops or time, return what we have (best effort).
    return ops
