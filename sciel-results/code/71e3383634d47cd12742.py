# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    start = time.time()
    DEADLINE = 85.0

    # Parse DAG
    if isinstance(dag, dict):
        parents = dag.get('parents', None)
        inputs = dag.get('inputs', None)
        outputs = dag.get('outputs', None)
        vertices = dag.get('vertices', None)
    else:
        raise ValueError("dag must be a dict")

    if parents is None:
        parents = {k: list(v) for k, v in dag.items()}
        if inputs is None:
            inputs = [v for v, ps in parents.items() if not ps]
        if outputs is None:
            all_parents = set()
            for ps in parents.values():
                all_parents.update(ps)
            outputs = [v for v in parents if v not in all_parents]
        if vertices is None:
            vertices = list(parents.keys())
    else:
        if vertices is None:
            vertices = list(parents.keys())
        if inputs is None:
            inputs = [v for v in vertices if not parents.get(v, [])]
        if outputs is None:
            all_parents = set()
            for ps in parents.values():
                all_parents.update(ps)
            outputs = [v for v in vertices if v not in all_parents]

    if not isinstance(inputs, list):
        inputs = list(inputs)
    if not isinstance(outputs, list):
        outputs = list(outputs)

    budget = max(0, int(fast_memory_budget))

    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)

    # Topological order
    in_deg = {v: len(parents.get(v, [])) for v in vertices}
    q = deque([v for v in vertices if in_deg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for c in children.get(u, []):
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    if len(topo) != len(vertices):
        topo = list(vertices)

    topo_idx = {v: i for i, v in enumerate(topo)}

    fast = set()
    slow = set()
    computed = set()
    ops = []

    # Initialize inputs in fast memory
    for inp in inputs:
        fast.add(inp)
    while len(fast) > budget:
        def next_use(v):
            if not children.get(v):
                return float('inf')
            return min(topo_idx[c] for c in children[v])
        evict_v = max(fast, key=next_use)
        ops.append(['store', evict_v])
        fast.discard(evict_v)
        slow.add(evict_v)

    def evict_one():
        if not fast:
            return
        def next_use(v):
            if not children.get(v):
                return float('inf')
            return min(topo_idx[c] for c in children[v])
        evict_v = max(fast, key=next_use)
        ops.append(['store', evict_v])
        fast.discard(evict_v)
        slow.add(evict_v)

    def ensure_available(v):
        if v in fast:
            return
        if v in slow:
            if len(fast) >= budget:
                evict_one()
            ops.append(['load', v])
            slow.discard(v)
            fast.add(v)
        else:
            for p in parents.get(v, []):
                ensure_available(p)
            ops.append(['compute', v])
            computed.add(v)
            if len(fast) >= budget:
                evict_one()
            fast.add(v)

    for v in topo:
        if v in inputs:
            continue
        if time.time() - start > DEADLINE:
            break
        ensure_available(v)

    for out in outputs:
        if out not in fast:
            ensure_available(out)

    return ops
# EVOLVE-BLOCK-END
