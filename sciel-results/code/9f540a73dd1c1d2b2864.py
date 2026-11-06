"""Pebling DAG with communication minimization."""
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 55.0

    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))

    if not vertices:
        return []

    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)

    layer = {}
    for v in vertices:
        if v in inputs:
            layer[v] = 0
        else:
            ps = parents.get(v, [])
            if not ps:
                layer[v] = 0
            else:
                layer[v] = max(layer[p] for p in ps)

    max_layer = max(layer.values()) if layer else 0
    layers = defaultdict(list)
    for v in vertices:
        layers[layer[v]].append(v)

    fast_mem = set()
    ops = []
    computed = set()

    def elapsed():
        return time.time() - start_time

    def is_needed(v):
        if v in outputs:
            return True
        for c in children.get(v, []):
            if c not in computed:
                return True
        return False

    def evict_one():
        """Evict one vertex from fast_mem, preferring unneeded non-outputs."""
        # First: unneeded non-outputs
        for fm_v in list(fast_mem):
            if not is_needed(fm_v) and fm_v not in outputs:
                ops.append(['delete', fm_v])
                fast_mem.discard(fm_v)
                return
        # Second: unneeded outputs
        for fm_v in list(fast_mem):
            if not is_needed(fm_v):
                ops.append(['delete', fm_v])
                fast_mem.discard(fm_v)
                return
        # Third: any non-output
        for fm_v in list(fast_mem):
            if fm_v not in outputs:
                ops.append(['delete', fm_v])
                fast_mem.discard(fm_v)
                return
        # Fourth: any
        if fast_mem:
            fm_v = next(iter(fast_mem))
            ops.append(['delete', fm_v])
            fast_mem.discard(fm_v)

    for L in range(max_layer + 1):
        if elapsed() > DEADLINE:
            break
        layer_vertices = layers[L]

        for v in layer_vertices:
            if elapsed() > DEADLINE:
                break
            if v in inputs:
                continue

            for p in parents.get(v, []):
                if p not in fast_mem and p not in inputs:
                    if len(fast_mem) >= fast_memory_budget:
                        evict_one()
                    ops.append(['load', p])
                    fast_mem.add(p)

            ops.append(['compute', v])
            fast_mem.add(v)
            computed.add(v)

            while len(fast_mem) > fast_memory_budget:
                evict_one()

        for prev_L in range(L):
            for v in layers[prev_L]:
                if v in fast_mem and not is_needed(v):
                    ops.append(['delete', v])
                    fast_mem.discard(v)

    for v in outputs:
        if v in fast_mem:
            ops.append(['store', v])
            fast_mem.discard(v)
        elif v in computed:
            if len(fast_mem) >= fast_memory_budget:
                evict_one()
            ops.append(['load', v])
            fast_mem.add(v)
            ops.append(['store', v])
            fast_mem.discard(v)
        else:
            ops.append(['compute', v])
            ops.append(['store', v])

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
