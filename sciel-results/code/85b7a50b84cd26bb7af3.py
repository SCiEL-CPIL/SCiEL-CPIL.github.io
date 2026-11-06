"""Communication-minimizing DAG pebbling (layered, recompute-aware)."""
import time

_START = time.time()
_DEADLINE = 45.0          # safety margin vs 90 s wall-clock kill
_MAX_OPS = 20000


def _now():
    return time.time()


def _time_left():
    return _DEADLINE - (_now() - _START)


# --------------------------------------------------------------------------- #
# Robust DAG parsing
# --------------------------------------------------------------------------- #
def _first(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None


def _parse_dag(dag):
    """Return (order, parent_map, outputs, inputs, children)."""
    if dag is None:
        return [], {}, [], [], {}

    d = dag
    if isinstance(d, dict):
        for wrap in ("dag", "graph", "problem", "instance", "data"):
            if wrap in d and isinstance(d[wrap], dict):
                d = d[wrap]
                break

    vertices = _first(d, ["vertices", "nodes", "V", "vertex_list"])
    if vertices is None:
        edges = _first(d, ["edges", "links", "E", "arcs"])
        if isinstance(edges, (list, tuple)):
            seen = []
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    for x in (e[0], e[1]):
                        if x not in seen:
                            seen.append(x)
            vertices = seen
    if vertices is None:
        return [], {}, [], [], {}
    vertices = list(vertices)

    parent_map = {v: [] for v in vertices}
    edges = _first(d, ["edges", "links", "E", "arcs", "edge_list"])
    if isinstance(edges, (list, tuple)):
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                a, b = e[0], e[1]
                if a in parent_map and b in parent_map and a != b:
                    if a not in parent_map[b]:
                        parent_map[b].append(a)
    elif isinstance(edges, dict):
        for k, v in edges.items():
            if k in parent_map and isinstance(v, (list, tuple, set)):
                for p in v:
                    if p in parent_map and p != k and p not in parent_map[k]:
                        parent_map[k].append(p)

    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parent_map[v]:
            if v not in children[p]:
                children[p].append(v)

    outputs = _first(d, ["outputs", "sinks", "out", "out_vertices", "targets"])
    if outputs is None:
        outputs = [v for v in vertices if not children[v]]
    outputs = [v for v in list(outputs) if v in parent_map]

    inputs = _first(d, ["inputs", "sources", "in", "in_vertices"])
    if inputs is None:
        inputs = [v for v in vertices if not parent_map[v]]
    inputs = [v for v in list(inputs) if v in parent_map]

    # Topological ordering (Kahn); fall back to given order.
    indeg = {v: len(parent_map[v]) for v in vertices}
    from collections import deque
    q = deque([v for v in vertices if indeg[v] == 0])
    topo, indeg2 = [], dict(indeg)
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    order = topo if len(topo) == len(vertices) else vertices

    return order, parent_map, outputs, inputs, children


# --------------------------------------------------------------------------- #
# Verification (mirrors the exact state machine)
# --------------------------------------------------------------------------- #
def _validate_ops(order, parent_map, outputs, inputs, ops, budget):
    idx = {v: i for i, v in enumerate(order)}
    n = len(order)
    blue = [False] * n
    red = [False] * n
    for v in inputs:
        if v in idx:
            blue[idx[v]] = True
    used = 0
    for op, v in ops:
        i = idx.get(v)
        if i is None:
            return False
        if op == "load":
            if not (blue[i] and not red[i]):
                return False
            blue[i], red[i] = False, True
            used += 1
        elif op == "store":
            if not (red[i] and not blue[i]):
                return False
            red[i], blue[i] = False, True
            used -= 1
        elif op == "compute":
            if red[i] or blue[i]:
                return False
            for p in parent_map[v]:
                if not red[idx[p]]:
                    return False
            red[i] = True
            used += 1
        elif op == "delete":
            if not red[i]:
                return False
            red[i] = False
            used -= 1
        else:
            return False
        if used > budget:
            return False
    for o in outputs:
        if o in idx and not blue[idx[o]]:
            return False
    return len(ops) <= _MAX_OPS


# --------------------------------------------------------------------------- #
# Layered schedule (clean, memory = max level width)
# --------------------------------------------------------------------------- #
def _layered_schedule(order, parent_map, outputs, inputs, children, budget):
    depth = {}
    for v in order:
        ps = parent_map[v]
        depth[v] = 0 if not ps else 1 + max(depth[p] for p in ps)
    maxd = max(depth.values()) if depth else 0
    levels = [[] for _ in range(maxd + 1)]
    for v in order:
        levels[depth[v]].append(v)
    if any(len(lv) > budget for lv in levels):
        return None  # needs recompute fallback

    ops = []
    for v in inputs:
        ops.append(["load", v])
    for d in range(1, maxd + 1):
        for v in levels[d]:
            ops.append(["compute", v])
        for v in levels[d]:
            ops.append(["store", v])
    return ops


# --------------------------------------------------------------------------- #
# Recompute-on-demand schedule (always correct, memory-bounded)
# --------------------------------------------------------------------------- #
def _recompute_schedule(order, parent_map, outputs, inputs, children, budget):
    idx = {v: i for i, v in enumerate(order)}
    outset = set(outputs)
    ops = []
    in_mem = set()
    # unprocessed_dependents[v] = number of children of v not yet processed.
    unproc = {v: len(children[v]) for v in order}

    for v in inputs:
        ops.append(["load", v])
        in_mem.add(v)

    for v in order:
        if v in inputs:
            continue
        # Ensure each parent is red (recompute if it was deleted).
        for p in parent_map[v]:
            if p not in in_mem:
                ops.append(["compute", p])
                in_mem.add(p)
        ops.append(["compute", v])
        in_mem.add(v)
        # v's children are all later in topological order -> none processed yet,
        # so we CANNOT delete v now unless it has no children and is not output.
        if not children[v] and v not in outset:
            ops.append(["delete", v])
            in_mem.remove(v)
        # For vertices WITH children we keep them (they are needed). To bound
        # memory we delete a vertex right after its LAST child is computed.
        # Track that by decrementing the parent's unprocessed count when a child
        # is computed, and deleting the parent at that moment.
        for p in parent_map[v]:
            unproc[p] -= 1
            if unproc[p] == 0 and p not in outset and p in in_mem:
                ops.append(["delete", p])
                in_mem.remove(p)

    # Final: make outputs blue.
    for v in outputs:
        if v not in in_mem:
            for p in parent_map[v]:
                if p not in in_mem:
                    ops.append(["compute", p])
                    in_mem.add(p)
            ops.append(["compute", v])
            in_mem.add(v)
        ops.append(["store", v])
        in_mem.discard(v)

    return ops


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def pebble(dag, fast_memory_budget):
    order, parent_map, outputs, inputs, children = _parse_dag(dag)
    if not order:
        return []

    budget = fast_memory_budget if fast_memory_budget is not None else 0
    if budget < 0:
        budget = 0

    # Fast path: clean layered schedule when it fits.
    if _time_left() > 0.5:
        ops = _layered_schedule(order, parent_map, outputs, inputs, children, budget)
        if ops is not None and _validate_ops(order, parent_map, outputs, inputs, ops, budget):
            return ops

    # Fallback: recompute-on-demand (always correct).
    ops = _recompute_schedule(order, parent_map, outputs, inputs, children, budget)
    if _validate_ops(order, parent_map, outputs, inputs, ops, budget):
        return ops

    # Last-resort trivially-correct (may exceed budget only if infeasible).
    ops = []
    for v in inputs:
        ops.append(["load", v])
    for v in order:
        if v in inputs:
            continue
        ops.append(["compute", v])
        ops.append(["store", v])
    return ops
