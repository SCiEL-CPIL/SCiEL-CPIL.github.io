"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85  # seconds

    # Parse DAG
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        edges = dag.get('edges', [])
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        vertices = list(getattr(dag, 'vertices', []))
        edges = getattr(dag, 'edges', [])
        inputs = set(getattr(dag, 'inputs', []))
        outputs = set(getattr(dag, 'outputs', []))

    # Build parent/child relationships
    children = defaultdict(list)
    parents = defaultdict(list)
    for edge in edges:
        u, v = edge[0], edge[1]
        children[u].append(v)
        parents[v].append(u)

    # Topological layers
    in_degree = {v: len(parents[v]) for v in vertices}
    q = deque([v for v in vertices if in_degree[v] == 0])
    layer_of = {}
    layer = 0
    current = list(q)
    while current:
        for v in current:
            layer_of[v] = layer
        next_curr = []
        for v in current:
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    next_curr.append(c)
        current = next_curr
        layer += 1

    layers = defaultdict(list)
    for v in vertices:
        layers[layer_of[v]].append(v)
    max_layer = max(layer_of.values()) if layer_of else 0

    # Descendant count for eviction priority
    desc_count = {}
    def get_desc(v, memo={}):
        if v in memo:
            return memo[v]
        total = sum(1 + get_desc(c) for c in children[v])
        memo[v] = total
        return total
    for v in vertices:
        desc_count[v] = get_desc(v)

    # State
    in_mem = set()  # vertices in fast memory (red)
    ops = []

    # Process layer by layer
    for L in range(max_layer + 1):
        if time.time() - start_time > DEADLINE:
            break
        layer_verts = layers[L]

        # Load parents that aren't in memory
        for v in layer_verts:
            for p in parents[v]:
                if p not in in_mem:
                    ops.append(['load', p])
                    in_mem.add(p)

        # Compute layer vertices
        for v in layer_verts:
            ops.append(['compute', v])
            in_mem.add(v)

        # Evict if over budget
        # Must keep: parents of next layer + outputs (if in this layer)
        must_keep = set()
        if L + 1 <= max_layer:
            for v in layers[L + 1]:
                must_keep.update(parents[v])
        must_keep.update(outputs & set(layer_verts))  # outputs in this layer must be kept

        while len(in_mem) > fast_memory_budget:
            evictable = [v for v in in_mem if v not in must_keep]
            if not evictable:
                break
            ev = min(evictable, key=lambda x: desc_count.get(x, 0))
            ops.append(['delete', ev])
            in_mem.discard(ev)

    # Store all outputs to blue
    for o in outputs:
        if o in in_mem:
            ops.append(['store', o])
            in_mem.discard(o)
        else:
            # Output was evicted; need to reload or recompute
            # For now, just load (assumes it was stored before)
            # This is incomplete; a full solution would recompute
            ops.append(['load', o])
            ops.append(['store', o])

    # Truncate if needed
    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

# EVOLVE-BLOCK-END
