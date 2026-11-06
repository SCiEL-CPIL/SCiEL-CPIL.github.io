"""Rewritten program: communication-minimizing DAG pebbling with next-use eviction."""
# EVOLVE-BLOCK-START
import time
from collections import deque

_START = time.time()
_DEADLINE = 72.0

def _elapsed():
    return time.time() - _START

def _hit():
    return _elapsed() > _DEADLINE

def _parents_of(v, dag):
    parents = getattr(dag, "parents", None)
    if parents is None:
        return []
    return list(parents.get(v, parents.get(int(v), [])))

def _children_of(v, dag):
    children = getattr(dag, "children", None)
    if children is None:
        return []
    return list(children.get(v, children.get(int(v), [])))

def _is_input(v, dag):
    return len(_parents_of(v, dag)) == 0

def _outputs(dag):
    out = getattr(dag, "outputs", None)
    if out is not None:
        return list(out)
    children = getattr(dag, "children", None)
    if children is not None:
        return [v for v in children if not children.get(v)]
    return []

def _all_vertices(dag):
    parents = getattr(dag, "parents", None)
    if parents is not None:
        return list(parents.keys())
    children = getattr(dag, "children", None)
    if children is not None:
        return list(children.keys())
    return []

def pebble(dag, fast_memory_budget):
    try:
        return _pebble(dag, fast_memory_budget)
    except Exception:
        return _naive(dag)

def _naive(dag):
    ops = []
    outs = set(_outputs(dag))
    for v in _all_vertices(dag):
        if _is_input(v, dag):
            ops.append(["load", v])
        else:
            ops.append(["compute", v])
        if v in outs:
            ops.append(["store", v])
    return ops

def _pebble(dag, budget):
    verts = _all_vertices(dag)
    if not verts:
        return []
    outs = set(_outputs(dag))
    children = {v: _children_of(v, dag) for v in verts}
    parents = {v: _parents_of(v, dag) for v in verts}

    indeg = {v: len(parents[v]) for v in verts}
    q = deque([v for v in verts if indeg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(topo) != len(verts):
        return _naive(dag)

    # next-use index for each vertex
    pos = {v: i for i, v in enumerate(topo)}
    next_use = {}
    for v in verts:
        nxt = None
        for c in children[v]:
            if c in pos:
                if nxt is None or pos[c] < nxt:
                    nxt = pos[c]
        next_use[v] = nxt

    red = set()
    ops = []

    def evict_one():
        cands = [r for r in red if r not in outs]
        if not cands:
            cands = list(red)
        if not cands:
            return
        def key(r):
            nu = next_use.get(r)
            return (nu is None, nu if nu is not None else 0)
        victim = max(cands, key=key)
        red.discard(victim)

    for v in topo:
        if _is_input(v, dag):
            if v not in red:
                ops.append(["load", v])
                red.add(v)
            if len(red) > budget:
                evict_one()
        else:
            for p in parents[v]:
                if p not in red:
                    ops.append(["load", p])
                    red.add(p)
                    if len(red) > budget:
                        evict_one()
            ops.append(["compute", v])
            red.add(v)
            if len(red) > budget:
                evict_one()
            if v in outs:
                ops.append(["store", v])
                red.discard(v)
                if len(red) > budget:
                    evict_one()
        if _hit():
            break

    return ops
# EVOLVE-BLOCK-END
