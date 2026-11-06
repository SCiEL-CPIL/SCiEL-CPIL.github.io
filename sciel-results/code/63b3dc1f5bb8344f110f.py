# EVOLVE-BLOCK-START
import time
import sys

sys.setrecursionlimit(10000)

_START = time.time()
_DEADLINE = _START + 82.0   # safety margin under the 90s wall-clock kill


def _norm(dag):
    """Return (parents, children, inputs, outputs) as plain lists/sets."""
    parents = list(dag.get("parents", []))
    children = dag.get("children")
    if children is None:
        n = len(parents)
        children = [[] for _ in range(n)]
        for i, ps in enumerate(parents):
            for p in ps:
                children[p].append(i)
        children = [list(c) for c in children]
    inputs = dag.get("inputs")
    if inputs is None:
        inputs = [i for i, ps in enumerate(parents) if len(ps) == 0]
    outputs = dag.get("outputs")
    if outputs is None:
        outputs = [i for i, ch in enumerate(children) if len(ch) == 0]
    return parents, children, list(inputs), list(outputs)


def _get_topo_order(n, parents, children):
    in_degree = [len(p) for p in parents]
    queue = [i for i in range(n) if in_degree[i] == 0]
    order = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    if len(order) != n:
        return list(range(n))
    return order


def _naive_transfers(parents, children, inputs, outputs):
    """Reference: load each input once, compute each vertex once, store each output once."""
    t = 0
    for i in inputs:
        t += 1
    for o in outputs:
        t += 1
    return max(t, 1)


def _count_transfers(ops):
    return sum(1 for op, _ in ops if op in ("load", "store"))


def pebble(dag, fast_memory_budget):
    if time.time() > _DEADLINE:
        parents, children, inputs, outputs = _norm(dag)
        ops = []
        for i in inputs:
            ops.append(["load", i])
        for i in range(len(parents)):
            ops.append(["compute", i])
        for o in outputs:
            ops.append(["store", o])
        return ops

    parents, children, inputs, outputs = _norm(dag)
    n = len(parents)

    if n == 0:
        return []

    budget = int(fast_memory_budget) if fast_memory_budget else 1
    if budget < 1:
        budget = 1

    topo_order = _get_topo_order(n, parents, children)
    topo_index = {v: i for i, v in enumerate(topo_order)}
    inputs_set = set(inputs)
    
    # Identify relevant nodes (ancestors of outputs)
    relevant = set(outputs)
    queue = list(outputs)
    while queue:
        v = queue.pop()
        for p in parents[v]:
            if p not in relevant:
                relevant.add(p)
                queue.append(p)

    ops = []
    fast = set()
    avail = set() # Vertices that have been computed at least once (non-inputs)

    def ensure_blue(v):
        nonlocal fast, avail, ops
        if v in fast:
            return
        
        if v in avail:
            ops.append(["load", v])
            fast.add(v)
            return
        
        if v in inputs_set:
            ops.append(["load", v])
            fast.add(v)
            return
            
        for p in parents[v]:
            ensure_blue(p)
            
        ops.append(["compute", v])
        fast.add(v)
        avail.add(v)
        
        if len(fast) > budget:
            victim = None
            min_idx = float('inf')
            for u in fast:
                if u == v:
                    continue
                idx = topo_index.get(u, 0)
                if idx < min_idx:
                    min_idx = idx
                    victim = u
            
            if victim is not None:
                fast.discard(victim)
                ops.append(["delete", victim])

    for v in topo_order:
        if v in relevant:
            ensure_blue(v)
            
    for o in outputs:
        if o not in fast:
            if o in avail:
                ops.append(["load", o])
                fast.add(o)
            else:
                ensure_blue(o)
        ops.append(["store", o])
        
    if len(ops) > 20000:
        ops = ops[:20000]
        
    return ops
# EVOLVE-BLOCK-END
