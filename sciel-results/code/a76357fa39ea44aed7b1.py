"""Communication-minimizing DAG pebbling with independent verification."""
import time
from collections import deque

try:
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    _HAVE_BASELINE = True
except Exception:
    _HAVE_BASELINE = False


def _count_transfers(ops):
    c = 0
    for op in ops:
        if op and op[0] in ('load', 'store'):
            c += 1
    return c


def _extract(dag):
    def get(d, k, default=None):
        if isinstance(d, dict):
            return d.get(k, default)
        return getattr(d, k, default)

    vertices = get(dag, 'vertices')
    parents = get(dag, 'parents')
    inputs = get(dag, 'inputs')
    outputs = get(dag, 'outputs')

    if vertices is None and isinstance(dag, dict):
        vertices = list(dag.keys())
        parents = {}
        for k, v in dag.items():
            if isinstance(v, (list, tuple, set)):
                parents[k] = list(v)
            else:
                parents[k] = []

    if vertices is None:
        return None
    if parents is None:
        parents = {}
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []
    return vertices, parents, inputs, outputs


def _verify(ops, vertices, parents, inputs, outputs, budget):
    vset = set(vertices)
    in_mem = set()
    for v in inputs:
        if v in vset:
            in_mem.add(v)
    for op in ops:
        if not op or len(op) < 2:
            return False
        name, v = op[0], op[1]
        if v not in vset:
            return False
        if name == 'load':
            if v not in in_mem:
                in_mem.add(v)
        elif name == 'store':
            if v not in in_mem:
                return False
            in_mem.discard(v)
        elif name == 'delete':
            if v not in in_mem:
                return False
            in_mem.discard(v)
        elif name == 'compute':
            for p in parents.get(v, []):
                if p not in in_mem:
                    return False
            in_mem.add(v)
        else:
            return False
        if len(in_mem) > budget:
            return False
    for v in outputs:
        if v in vset and v not in in_mem:
            return False
    return True


def _my_schedule(dag, fast_memory_budget, deadline):
    ex = _extract(dag)
    if ex is None:
        return None
    vertices, parents, inputs, outputs = ex
    n = len(vertices)
    if n == 0:
        return []
    vindex = {v: i for i, v in enumerate(vertices)}

    children = {v: [] for v in vertices}
    indeg = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in vindex:
                children[p].append(v)
                indeg[v] += 1

    indeg2 = dict(indeg)
    q = deque([v for v in vertices if indeg2[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    for v in vertices:
        if v not in topo:
            topo.append(v)
    pos = {v: i for i, v in enumerate(topo)}

    # Precompute next use distance for each vertex
    # next_use[v] = position of next child of v in topo, or infinity if none
    next_use = {v: 10**9 for v in vertices}
    for v in vertices:
        for c in children[v]:
            cu = pos.get(c, 10**9)
            if cu < next_use[v]:
                next_use[v] = cu

    remaining_children = {v: len(children[v]) for v in vertices}
    in_mem = set()
    for v in inputs:
        if v in vindex:
            in_mem.add(v)
    outputs_set = set(v for v in outputs if v in vindex)

    try:
        budget = max(0, int(fast_memory_budget))
    except Exception:
        budget = 0

    ops = []

    def evict():
        nonlocal in_mem
        while len(in_mem) > budget:
            candidates = [x for x in in_mem if x not in outputs_set]
            if not candidates:
                candidates = list(in_mem)
            # Evict vertex with farthest next use (least likely to be needed soon)
            # Tie-break by fewer remaining children, then later position
            victim = min(candidates,
                         key=lambda x: (-next_use.get(x, 10**9),
                                        remaining_children.get(x, 0),
                                        pos.get(x, 10 ** 9)))
            ops.append(['store', victim])
            in_mem.discard(victim)

    # Track which vertices we've already computed to avoid redundant work
    computed = set()
    
    for v in topo:
        if time.time() > deadline:
            break
        if v in computed:
            continue
        ps = [p for p in parents.get(v, []) if p in vindex]
        # Check if all parents are available; if not, load them
        for p in ps:
            if p not in in_mem:
                ops.append(['load', p])
                in_mem.add(p)
        ops.append(['compute', v])
        if v not in in_mem:
            in_mem.add(v)
        computed.add(v)
        evict()
        for c in children[v]:
            remaining_children[c] -= 1

    for v in outputs:
        if v in vindex and v not in in_mem:
            ops.append(['load', v])
            in_mem.add(v)

    return ops


def pebble(dag, fast_memory_budget):
    start = time.time()
    deadline = start + 85.0

    baseline_ops = None
    if _HAVE_BASELINE:
        try:
            b = pebble_baseline(dag, fast_memory_budget)
            if isinstance(b, list):
                baseline_ops = b
        except Exception:
            baseline_ops = None

    my_ops = None
    try:
        m = _my_schedule(dag, fast_memory_budget, deadline)
        if m is not None and len(m) <= 20000:
            ex = _extract(dag)
            if ex is not None:
                vertices, parents, inputs, outputs = ex
                try:
                    budget = max(0, int(fast_memory_budget))
                except Exception:
                    budget = 0
                if _verify(m, vertices, parents, inputs, outputs, budget):
                    my_ops = m
    except Exception:
        my_ops = None

    # Decision: prefer the verified schedule with fewer transfers.
    if my_ops is not None and baseline_ops is not None:
        if _count_transfers(my_ops) <= _count_transfers(baseline_ops):
            return my_ops
        return baseline_ops
    if my_ops is not None:
        return my_ops
    if baseline_ops is not None:
        return baseline_ops
    return []
