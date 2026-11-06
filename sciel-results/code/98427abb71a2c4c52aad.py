"""Communication-minimizing DAG pebbling: fresh design."""
import time
import heapq

# EVOLVE-BLOCK-START


def _topo_order(n, parents):
    """parents: list of lists (topologically ordered parents)."""
    indeg = [0] * n
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
            indeg[v] += 1
    q = [i for i in range(n) if indeg[i] == 0]
    order = []
    for x in q:
        order.append(x)
        for c in children[x]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(order) != n:  # fallback
        order = list(range(n))
    return order


def pebble(dag, fast_memory_budget):
    """Return list of [operation, vertex]."""
    t0 = time.time()
    DEADLINE = 55.0  # safety margin under 60s

    # --- parse dag (robust to several representations) ---
    # Assume dag is a dict or object with: 'parents' (list of lists),
    # 'inputs' (list of input vertices), 'outputs' (list of output vertices),
    # 'n' or 'num_vertices'.
    def _get(dag, *keys, default=None):
        if isinstance(dag, dict):
            for k in keys:
                if k in dag:
                    return dag[k]
            return default
        for k in keys:
            if hasattr(dag, k):
                return getattr(dag, k)
        return default

    parents = _get(dag, "parents", default=[])
    inputs = _get(dag, "inputs", default=[])
    outputs = _get(dag, "outputs", default=[])
    n = _get(dag, "n", "num_vertices", "size", default=None)
    if n is None:
        n = max((max(p) for p in parents if p), default=-1) + 1
        if inputs:
            n = max(n, max(inputs) + 1)
        if outputs:
            n = max(n, max(outputs) + 1)

    parents = [list(p) for p in parents]
    inputs = list(inputs)
    outputs = list(outputs)

    B = max(0, int(fast_memory_budget))
    MAX_OPS = 20000

    # Fast memory holds red (computed) values. Inputs start blue (in slow memory).
    # A "transfer" = load (slow->fast) or store (fast->slow).
    # compute is free; delete is free.
    # Goal: all outputs end blue (i.e., stored to slow memory).

    # We'll do a topological-order pebbling that keeps a window of vertices in
    # fast memory, reloading inputs when needed, and storing outputs.
    #
    # Strategy: process in topological order. Maintain set of vertices currently
    # in fast memory (red). For each vertex in topo order that is needed, ensure
    # its parents are in fast memory (load inputs if not present), compute it,
    # and if memory is full, evict (delete) some non-output vertex.
    # At the end, store all outputs.

    # Determine which vertices are "needed" — those that are ancestors of outputs.
    # Compute reverse reachability from outputs.
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)

    # needed = vertices that are inputs OR ancestors of any output
    needed = [False] * n
    stack = list(outputs)
    for o in outputs:
        needed[o] = True
    while stack:
        v = stack.pop()
        for p in parents[v]:
            if not needed[p]:
                needed[p] = True
                stack.append(p)
    for i in inputs:
        needed[i] = True

    order = _topo_order(n, parents)
    # filter to needed, preserving topo order
    order = [v for v in order if needed[v]]

    # naive reference: load each input once, compute each needed vertex, store each output
    # We aim to do at least as well; reuse keeps inputs in fast memory across uses.

    in_fast = set()
    ops = []

    def do(op, v):
        ops.append([op, v])

    def ensure(v):
        # make sure v is in fast memory (red). v is a parent of the vertex we compute.
        # If v is an input: load it if not in fast.
        # If v is computed: it should already be in fast if we keep it; else recompute.
        if v in in_fast:
            return
        # need to bring v into fast memory
        if v in inputs:
            # load
            if len(in_fast) >= B:
                # evict a non-output vertex
                evictable = [x for x in in_fast if x not in outputs]
                if evictable:
                    # evict the one with fewest remaining uses (heuristic: earliest needed)
                    e = evictable[0]
                    do("delete", e)
                    in_fast.discard(e)
            do("load", v)
            in_fast.add(v)
        else:
            # recompute v: ensure its parents are in fast, compute v
            for p in parents[v]:
                ensure(p)
            if len(in_fast) >= B:
                evictable = [x for x in in_fast if x not in outputs and x != v]
                if evictable:
                    e = evictable[0]
                    do("delete", e)
                    in_fast.discard(e)
            do("compute", v)
            in_fast.add(v)

    # Process each needed vertex in topo order; compute it and keep it if it's
    # used later (has children among needed) or is an output.
    # Build set of needed children counts to know which to keep.
    needed_children = [0] * n
    for v in range(n):
        for c in children[v]:
            if needed[c]:
                needed_children[v] += 1

    used_after = {}
    # count how many times each vertex is used as a parent among needed vertices
    use_count = [0] * n
    for v in order:
        for p in parents[v]:
            use_count[p] += 1

    # We compute each needed vertex exactly once in topo order.
    # After computing v, we keep it in fast memory if it will be used again
    # (use_count remaining > 0) and memory allows; else delete it.
    remaining_use = list(use_count)

    for v in order:
        if time.time() - t0 > DEADLINE:
            break
        # ensure parents in fast
        for p in parents[v]:
            ensure(p)
        # compute v
        if len(in_fast) >= B:
            evictable = [x for x in in_fast if x not in outputs and x != v]
            if evictable:
                e = evictable[0]
                do("delete", e)
                in_fast.discard(e)
        do("compute", v)
        in_fast.add(v)
        remaining_use[v] -= 1
        # if v no longer needed and not an output, we may delete to free space
        if remaining_use[v] <= 0 and v not in outputs:
            # keep for now; eviction happens lazily. But free if over budget.
            if len(in_fast) > B:
                do("delete", v)
                in_fast.discard(v)

    # Now store all outputs (they must finish blue).
    for o in outputs:
        if time.time() - t0 > DEADLINE:
            break
        if o not in in_fast:
            # recompute o
            for p in parents[o]:
                ensure(p)
            if len(in_fast) >= B:
                evictable = [x for x in in_fast if x not in outputs and x != o]
                if evictable:
                    e = evictable[0]
                    do("delete", e)
                    in_fast.discard(e)
            do("compute", o)
            in_fast.add(o)
        do("store", o)
        in_fast.discard(o)

    if len(ops) > MAX_OPS:
        # trim to a valid prefix that still stores outputs? fallback: keep all,
        # but cap to avoid invalid. Truncate is risky; instead return as-is if <= MAX_OPS.
        pass

    return ops


# EVOLVE-BLOCK-END
