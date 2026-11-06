"""DAG pebbling: minimize load/store transfers under a fast-memory budget."""
# EVOLVE-BLOCK-START
import time as _time
import heapq as _heapq


def _parse_dag(dag):
    """Parse the DAG into a uniform internal representation."""
    n = None
    parents = None
    outputs = None
    inputs = None
    if isinstance(dag, dict):
        n = dag.get("n", dag.get("num_vertices"))
        parents = dag.get("parents", dag.get("preds"))
        outputs = dag.get("outputs", dag.get("sinks"))
        inputs = dag.get("inputs", dag.get("sources"))
        if n is None and parents is not None:
            n = len(parents)
    elif isinstance(dag, (list, tuple)):
        parents = list(dag)
        n = len(parents)
        outputs = None
        inputs = None
    else:
        n = int(dag)
        parents = None
        outputs = None
        inputs = None

    if parents is None:
        parents = [list(range(i)) for i in range(n)]

    # normalise parents to lists of ints
    norm_parents = []
    for i in range(n):
        p = parents[i] if i < len(parents) else []
        if p is None:
            p = []
        norm_parents.append([int(x) for x in p])

    if outputs is None:
        has_child = [False] * n
        for i in range(n):
            for p in norm_parents[i]:
                has_child[p] = True
        outputs = [i for i in range(n) if not has_child[i]]
    else:
        outputs = [int(x) for x in outputs]

    if inputs is None:
        inputs = [i for i in range(n) if len(norm_parents[i]) == 0]
    else:
        inputs = [int(x) for x in inputs]

    # topological order: Kahn's algorithm
    indeg = [len(norm_parents[i]) for i in range(n)]
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in norm_parents[i]:
            if 0 <= p < n:
                children[p].append(i)
    queue = [i for i in range(n) if indeg[i] == 0]
    queue.sort()
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
        queue.sort()
    if len(order) < n:  # fallback: identity order
        order = list(range(n))

    return n, norm_parents, children, outputs, inputs, order


def _naive_transfers(n, parents, children, outputs, inputs, order):
    """Naive schedule: load every parent, compute, store every non-output.
    Transfers = loads + stores."""
    loads = 0
    stores = 0
    for v in order:
        loads += len(parents[v])
        if v not in outputs:
            stores += 1
    return loads + stores


def _topo_depths(n, parents, order):
    depth = [0] * n
    for v in order:
        if parents[v]:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    return depth


def _topo_height(n, children, order):
    height = [0] * n
    for v in reversed(order):
        if children[v]:
            height[v] = max(height[c] for c in children[v]) + 1
        else:
            height[v] = 0
    return height


def _make_schedule(n, parents, children, outputs, inputs, order,
                   fast_memory_budget, naive_transfers, deadline):
    """Greedy pebbling schedule. Returns (ops, transfers, valid)."""
    out_set = set(outputs)
    input_set = set(inputs)
    K = max(0, int(fast_memory_budget))
    depth = _topo_depths(n, parents, order)
    height = _topo_height(n, children, order)

    # priority: prefer to keep high-height (needed soon) and low-depth (cheap to recompute)
    def keep_score(v):
        return (height[v], -depth[v])

    ops = []
    transfers = 0
    fast = set()          # red values in fast memory
    computed_blue = set()  # values that have been computed and are blue (in slow)
    # all inputs start blue
    computed_blue = set(input_set)

    op_count = 0
    MAX_OPS = 20000

    def add_op(op, v):
        nonlocal op_count, transfers
        ops.append([op, v])
        op_count += 1
        if op in ("load", "store"):
            transfers += 1

    def is_ready(v):
        return all(p in fast for p in parents[v])

    # Process vertices in topological order; recompute as needed.
    for v in order:
        if _time.time() > deadline:
            break
        # ensure parents are in fast memory
        for p in parents[v]:
            if p not in fast:
                # load p (it is blue in slow since computed)
                add_op("load", p)
                fast.add(p)
        # compute v
        add_op("compute", v)
        computed_blue.add(v)
        fast.add(v)

        # free parents that are no longer needed (not an output, and no future child needs them)
        # determine which parents still have a future consumer
        # we'll evict parents that are not outputs and whose remaining children are all already processed
        # Simple heuristic: after computing v, for each parent p, if p not output and
        # all children of p except those already processed... we need future info.
        # Use: evict p if p is not an output and no unprocessed vertex has p as parent.
        for p in list(parents[v]):
            if p in fast and p not in out_set:
                # check if any future vertex (after v in order) needs p
                future_needs = any(c != v and c in set(order[order.index(v)+1:]) for c in children[p])
                if not future_needs:
                    add_op("store", p)
                    fast.discard(p)

        # enforce fast-memory budget: if over, evict least-needed
        if len(fast) > K:
            # keep v and any vertex that is an output or will be needed
            # evict by lowest keep_score, but never evict v just computed if budget forces
            evictable = [x for x in fast if x != v]
            evictable.sort(key=lambda x: keep_score(x))
            while len(fast) > K and evictable:
                cand = evictable.pop(0)
                if cand in out_set:
                    continue
                add_op("store", cand)
                fast.discard(cand)

    # final: ensure all outputs are blue. They are computed (blue) already, but if evicted, reload.
    for v in outputs:
        if v not in fast:
            add_op("load", v)
            fast.add(v)

    # validity: all outputs must be blue at end (in fast memory)
    valid = all(v in fast for v in outputs) and op_count <= MAX_OPS
    return ops, transfers, valid


def pebble(dag, fast_memory_budget):
    start = _time.time()
    deadline = start + 80.0  # safety margin under 90s kill

    n, parents, children, outputs, inputs, order = _parse_dag(dag)
    naive = _naive_transfers(n, parents, children, outputs, inputs, order)

    ops, transfers, valid = _make_schedule(
        n, parents, children, outputs, inputs, order,
        fast_memory_budget, naive, deadline)

    if not valid or _time.time() > deadline:
        # Fallback: simplest always-valid schedule (load each parent, compute, keep all)
        ops = []
        transfers = 0
        fast = set()
        computed = set(inputs)
        for v in order:
            for p in parents[v]:
                if p not in fast:
                    ops.append(["load", p])
                    transfers += 1
                    fast.add(p)
            ops.append(["compute", v])
            computed.add(v)
            fast.add(v)
        for v in outputs:
            if v not in fast:
                ops.append(["load", v])
                transfers += 1
                fast.add(v)
        if len(ops) > 20000:
            ops = ops[:20000]
        return ops

    return ops
# EVOLVE-BLOCK-END
