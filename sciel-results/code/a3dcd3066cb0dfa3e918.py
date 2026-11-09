"""Communication-minimizing DAG pebbling.

pebble(dag, fast_memory_budget) -> list of [op, vertex]
Ops: "load", "store", "compute", "delete".
"""
# EVOLVE-BLOCK-START
import time
from collections import deque

_TIME_BUDGET = 55.0
_OP_BUDGET = 20000
_START = None


def _elapsed():
    return time.perf_counter() - _START if _START is not None else 0.0


def _topo_order(n, parents):
    indeg = [0] * n
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
            indeg[v] += 1
    q = deque([v for v in range(n) if indeg[v] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for c in children[u]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    return order if len(order) == n else None


def _parse_dag(dag):
    if isinstance(dag, dict):
        if "parents" in dag and ("vertices" in dag or "n" in dag):
            n = int(dag.get("vertices", dag.get("n")))
            parents = [list(dag["parents"][i]) for i in range(n)]
            outputs = list(dag.get("outputs", []))
            inputs = list(dag.get("inputs", []))
            topo = _topo_order(n, parents)
            if topo is not None:
                return n, parents, outputs, inputs, topo
        # dict keyed by vertex -> parent list
        if dag and all(isinstance(k, (int, str)) for k in dag.keys()):
            keys = list(dag.keys())
            idx = {k: i for i, k in enumerate(keys)}
            n = len(keys)
            parents = [[idx[p] for p in dag[k] if p in idx] for k in keys]
            outputs = [idx[v] for v in dag.get("__outputs__", []) if v in idx]
            inputs = [idx[v] for v in dag.get("__inputs__", []) if v in idx]
            topo = _topo_order(n, parents)
            if topo is not None:
                return n, parents, outputs, inputs, topo
    if isinstance(dag, (list, tuple)):
        n = len(dag)
        parents = [list(p) for p in dag]
        outputs = [n - 1] if n else []
        inputs = [i for i in range(n) if not parents[i]]
        topo = _topo_order(n, parents)
        if topo is not None:
            return n, parents, outputs, inputs, topo
    return None


def _valid(ops, n, parents, outputs, in_set, budget):
    if len(ops) > _OP_BUDGET:
        return False
    red = set()
    for op, v in ops:
        if len(red) > budget:
            return False
        if op == "load":
            red.add(v)
        elif op == "store":
            red.discard(v)
        elif op == "compute":
            for p in parents[v]:
                if p not in red:
                    return False
            red.add(v)
        elif op == "delete":
            red.discard(v)
        else:
            return False
    return all(v not in red for v in outputs)


def _schedule(n, parents, outputs, in_set, topo, budget):
    """Recursive pebbling.

    pebble(v): make v red.
      - if input: load v (once; inputs are cheap and few).
      - else: for each parent p: pebble(p); compute v; store each parent.
    This keeps live set = (inputs kept red) + (1 per recursion level).
    We keep inputs red for the whole run to avoid reload cost; for layered
    DAGs inputs are the first layer and few.
    """
    ops = []
    red = set()

    def pebble(v):
        if _elapsed() > _TIME_BUDGET:
            return
        if v in red:
            return
        if v in in_set:
            ops.append(["load", v])
            red.add(v)
            return
        for p in parents[v]:
            pebble(p)
        ops.append(["compute", v])
        red.add(v)
        for p in parents[v]:
            ops.append(["store", p])
            red.discard(p)

    for v in topo:
        if v in in_set:
            pebble(v)
    # Compute each non-input; after computing, store it if it is not an
    # output and has no remaining uncomputed dependents, to bound memory.
    # Simpler: compute all, storing parents as we go (done in pebble).
    computed = set()
    for v in topo:
        if v in in_set:
            continue
        pebble(v)
        computed.add(v)

    for v in outputs:
        if v in red:
            ops.append(["store", v])
            red.discard(v)
    return ops


def pebble(dag, fast_memory_budget):
    global _START
    _START = time.perf_counter()
    budget = int(fast_memory_budget)

    parsed = _parse_dag(dag)
    if parsed is None:
        return []
    n, parents, outputs, inputs, topo = parsed
    in_set = set(inputs)

    # Generous budget -> single pass (minimal transfers).
    if n <= budget:
        ops = []
        for v in topo:
            if v in in_set:
                ops.append(["load", v])
            else:
                ops.append(["compute", v])
        for v in outputs:
            ops.append(["store", v])
        if _valid(ops, n, parents, outputs, in_set, budget):
            return ops

    ops = _schedule(n, parents, outputs, in_set, topo, budget)
    if _valid(ops, n, parents, outputs, in_set, budget):
        return ops

    # Fallback: reload-everything schedule, always memory-safe.
    ops = []
    red = set()
    for v in topo:
        if v in in_set:
            ops.append(["load", v]); red.add(v)
        else:
            for p in parents[v]:
                if p not in red:
                    ops.append(["load", p]); red.add(p)
            ops.append(["compute", v]); red.add(v)
            for p in parents[v]:
                ops.append(["store", p]); red.discard(p)
    for v in outputs:
        if v in red:
            ops.append(["store", v]); red.discard(v)
    return ops[:_OP_BUDGET]
# EVOLVE-BLOCK-END
