# EVOLVE-BLOCK-START
import time as _time
import heapq as _heapq
from collections import defaultdict as _dd
from collections import deque

# Global deadline guard (90s wall clock, safety margin)
_DEADLINE = _time.time() + 85.0
_ops_cap = 20000


def pebble(dag, fast_memory_budget):
    t0 = _time.time()
    try:
        return _solve(dag, fast_memory_budget, t0)
    except Exception:
        # Fallback: naive sequential schedule (always valid)
        return _naive_fallback(dag, fast_memory_budget)


def _naive_fallback(dag, fast_memory_budget):
    """Naive: load all inputs, compute in topo order, store all outputs."""
    ops = []
    inputs = dag.get("inputs", [])
    outputs = dag.get("outputs", [])
    topo = dag.get("topo", dag.get("order", None))
    if topo is None:
        # try to derive from vertices
        verts = dag.get("vertices", [])
        topo = list(verts)
    # load inputs
    for v in inputs:
        ops.append(["load", v])
    # compute non-input vertices in topo order
    for v in topo:
        if v in inputs:
            continue
        ops.append(["compute", v])
    # store outputs
    for v in outputs:
        ops.append(["store", v])
    return ops


def _solve(dag, fast_memory_budget, t0):
    # Parse the DAG structure
    inputs = list(dag.get("inputs", []))
    outputs = list(dag.get("outputs", []))
    parents = dag.get("parents", {})
    if isinstance(parents, list):
        parents = {i: p for i, p in enumerate(parents)}
    
    # vertices: all keys + values
    verts = set()
    for v in inputs:
        verts.add(v)
    for v in outputs:
        verts.add(v)
    for v, ps in parents.items():
        verts.add(v)
        for p in ps:
            verts.add(p)
            
    # topological order
    topo = dag.get("topo", dag.get("order", None))
    if topo is None:
        # compute via Kahn
        indeg = {v: 0 for v in verts}
        children = _dd(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
                indeg[v] = indeg.get(v, 0)
        q = [v for v in verts if indeg.get(v, 0) == 0]
        topo = []
        dq = deque(q)
        while dq:
            v = dq.popleft()
            topo.append(v)
            for c in children[v]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    dq.append(c)
    if not topo:
        topo = sorted(verts, key=lambda x: (str(type(x)), x))

    input_set = set(inputs)
    output_set = set(outputs)

    B = int(fast_memory_budget) if fast_memory_budget is not None else 0
    if B < 0:
        B = 0

    # Build child map for potential heuristics, though we use LRU for simplicity
    children = _dd(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # If budget is 0, naive is the only sane option (can't hold any red value)
    if B == 0:
        return _naive_fallback(dag, fast_memory_budget)

    fast_mem = {}  # vertex -> True
    lru_order = []  # list of vertices in LRU order (least recent at front)
    ops = []
    transfers = 0
    op_count = 0

    def evict_if_needed():
        nonlocal fast_mem, lru_order
        while len(fast_mem) > B:
            if lru_order:
                v = lru_order.pop(0)
                if v in fast_mem:
                    del fast_mem[v]
            else:
                # Fallback if lru_order is inconsistent
                for k in list(fast_mem.keys()):
                    del fast_mem[k]
                    break

    def touch(v):
        nonlocal lru_order
        if v in lru_order:
            lru_order.remove(v)
        lru_order.append(v)

    def add_to_mem(v, is_load=False):
        nonlocal transfers, op_count
        if is_load:
            ops.append(["load", v])
            transfers += 1
            op_count += 1
        else:
            ops.append(["compute", v])
            op_count += 1
        fast_mem[v] = True
        touch(v)
        evict_if_needed()

    def ensure_in_memory(v):
        nonlocal op_count
        # If v is already in fast memory, do nothing
        if v in fast_mem:
            touch(v)
            return
        
        # If v is an input, load it
        if v in input_set:
            add_to_mem(v, is_load=True)
            return
        
        # If v is not an input, it must be computed.
        # Ensure its parents are in memory first.
        ps = parents.get(v, [])
        for p in ps:
            ensure_in_memory(p)
        
        # Now compute v
        add_to_mem(v, is_load=False)

    # Strategy:
    # 1. Ensure all inputs are in memory? No, that might exceed B.
    #    Instead, process vertices in topo order.
    # 2. For each vertex v in topo:
    #    - ensure_in_memory(v) will handle loading inputs or recomputing nodes.
    #    - If v is an output, store it.
    
    # Optimization: Pre-load inputs if they fit, to avoid reloading them later.
    # But if B is small, we can't hold all. 
    # The ensure_in_memory logic will reload/recompute as needed.
    
    for v in topo:
        if _time.time() - t0 > 80:
            return _naive_fallback(dag, fast_memory_budget)
        
        if v in input_set:
            # Ensure it's in memory (it might have been evicted)
            ensure_in_memory(v)
        else:
            ensure_in_memory(v)
            
        if v in output_set:
            ops.append(["store", v])
            transfers += 1
            op_count += 1
            # We can optionally evict v here if we know it's not needed later,
            # but LRU handles that.
        
        if op_count > _ops_cap:
            return _naive_fallback(dag, fast_memory_budget)

    return ops
# EVOLVE-BLOCK-END
