# EVOLVE-BLOCK-START
import time

def _extract(dag):
    verts = getattr(dag, "vertices", None)
    if verts is None:
        verts = getattr(dag, "nodes", None)
    if verts is None:
        verts = getattr(dag, "V", None)
    if verts is None and isinstance(dag, dict):
        verts = list(dag.keys())
    if verts is None:
        raise ValueError("no vertices")
    verts = list(verts)

    parent_map = getattr(dag, "parent_map", None)
    parents = {}
    for v in verts:
        p = None
        if isinstance(dag, dict) and v in dag and isinstance(dag[v], dict):
            p = dag[v].get("parents")
        if p is None and parent_map is not None:
            p = parent_map.get(v, [])
        if p is None:
            p = getattr(dag, "parents", None)
            if callable(p):
                p = p(v)
        if p is None:
            p = []
        parents[v] = list(p)

    outputs = getattr(dag, "outputs", None)
    if outputs is None:
        outputs = getattr(dag, "sinks", None)
    if outputs is None and isinstance(dag, dict):
        outputs = dag.get("outputs", [])
    if outputs is None:
        outputs = []
    outputs = list(outputs)

    inputs = [v for v in verts if len(parents[v]) == 0]

    indeg = {v: len(parents[v]) for v in verts}
    from collections import deque
    q = deque(v for v in verts if indeg[v] == 0)
    topo = []
    child = {v: [] for v in verts}
    for v in verts:
        for p in parents[v]:
            child[p].append(v)
    while q:
        u = q.popleft()
        topo.append(u)
        for w in child[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(verts):
        topo = verts
    return verts, parents, outputs, inputs, topo

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    try:
        verts, parents, outputs, inputs, topo = _extract(dag)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

    if fast_memory_budget is None or fast_memory_budget < 1:
        fast_memory_budget = 1
    budget = int(fast_memory_budget)

    input_set = set(inputs)
    output_set = set(outputs)

    ops = []
    in_mem = set()
    for v in inputs:
        if len(in_mem) < budget:
            in_mem.add(v)

    remaining = [v for v in topo if v not in input_set]
    need = {v: 0 for v in verts}
    for v in remaining:
        for p in parents[v]:
            need[p] += 1
    for o in outputs:
        need[o] = max(need[o], 1)

    def evict_one():
        for v in list(in_mem):
            if need[v] == 0 and v not in output_set:
                in_mem.discard(v)
                ops.append(["store", v])
                return
        best = None
        for v in in_mem:
            if v in output_set:
                continue
            if best is None or need[v] < need[best]:
                best = v
        if best is not None:
            in_mem.discard(best)
            ops.append(["store", best])
            return
        for v in list(in_mem):
            in_mem.discard(v)
            ops.append(["store", v])
            return

    for v in remaining:
        if time.time() - t0 > 50:
            break
        for p in parents[v]:
            if p not in in_mem:
                while len(in_mem) >= budget and p not in in_mem:
                    evict_one()
                ops.append(["load", p])
                in_mem.add(p)
        while len(in_mem) >= budget and v not in in_mem:
            evict_one()
        ops.append(["compute", v])
        in_mem.add(v)
        for p in parents[v]:
            need[p] -= 1

    for o in outputs:
        if o not in in_mem:
            while len(in_mem) >= budget and o not in in_mem:
                evict_one()
            ops.append(["load", o])
            in_mem.add(o)

    if len(ops) > 20000:
        ops = ops[:20000]
    return ops
# EVOLVE-BLOCK-END
