"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        has_baseline = True
    except ImportError:
        has_baseline = False

    try:
        if isinstance(dag, dict):
            parents = dict(dag.get('parents', {}))
            outputs = list(dag.get('outputs', []))
            vertices = dag.get('vertices', None)
            inputs = dag.get('inputs', None)
        else:
            parents = dict(getattr(dag, 'parents', {}))
            outputs = list(getattr(dag, 'outputs', []))
            vertices = getattr(dag, 'vertices', None)
            inputs = getattr(dag, 'inputs', None)
    except Exception:
        if has_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        raise

    if vertices is not None:
        all_vertices = set(vertices)
    else:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        for o in outputs:
            all_vertices.add(o)

    if inputs is not None:
        input_vertices = set(inputs)
    else:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    # Compute layer for each vertex
    layer = {}
    for v in input_vertices:
        layer[v] = 0

    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    queue = deque(list(input_vertices))
    while queue:
        v = queue.popleft()
        for c in children[v]:
            if c not in layer:
                all_p_done = all(p in layer for p in parents.get(c, []))
                if all_p_done:
                    layer[c] = max(layer[p] for p in parents.get(c, [])) + 1
                    queue.append(c)

    for v in all_vertices:
        if v not in layer:
            layer[v] = 0

    layers = defaultdict(list)
    for v in all_vertices:
        layers[layer[v]].append(v)

    max_L = max(layers.keys()) if layers else 0

    # Precompute for each vertex, the earliest layer it's needed (as a parent)
    next_use_layer = {}
    for v in all_vertices:
        next_use_layer[v] = max_L + 1  # default: never needed
        for c in children.get(v, []):
            cl = layer.get(c, 0)
            if cl < next_use_layer[v]:
                next_use_layer[v] = cl

    # Also track how many times each vertex is used (degree in children)
    use_count = {v: len(children.get(v, [])) for v in all_vertices}

    # Precompute for each layer, which vertices are needed in that layer as parents
    needed_in_layer = defaultdict(set)
    for L in range(max_L + 1):
        for v in layers.get(L, []):
            if v in input_vertices:
                continue
            for p in parents.get(v, []):
                needed_in_layer[L].add(p)

    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []

    for L in range(max_L + 1):
        if time_check():
            break
        lv = layers.get(L, [])
        if not lv:
            continue

        # Determine which parents we need for this layer
        needed_parents = needed_in_layer.get(L, set())

        # Load needed parents that are in slow memory
        for p in needed_parents:
            if p not in fast_mem and p in slow_mem:
                ops.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)

        # Compute vertices in this layer
        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break
            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)

        # Evict if over budget - use smarter eviction based on next use and use count
        while len(fast_mem) > fast_memory_budget and not time_check():
            best = None
            best_score = float('inf')
            for u in fast_mem:
                # Score: lower is better to evict
                # 1. If it's an output, we must keep it (or store it)
                if u in outputs:
                    score = 100000
                # 2. If it's an input, it's cheap to reload
                elif u in input_vertices:
                    score = 1
                else:
                    # Score based on when it's next needed and how many times
                    nu = next_use_layer.get(u, max_L + 1)
                    uc = use_count.get(u, 0)
                    # If not needed again, evict it
                    if nu > max_L:
                        score = 2
                    else:
                        # Earlier next use = higher score (keep longer)
                        # More uses = higher score (keep longer)
                        score = 10000 - (nu * 100 + uc * 10)
                        # If it's needed in the current or next layer, keep it
                        if nu <= L + 1:
                            score += 50000

                if score < best_score:
                    best_score = score
                    best = u

            if best is None:
                break

            # Evict the best candidate
            if best in outputs:
                ops.append(['store', best])
                slow_mem.add(best)
            else:
                ops.append(['delete', best])
            fast_mem.discard(best)

    # Store outputs
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)
        elif o in slow_mem:
            # Already stored, nothing to do
            pass
        else:
            # Should not happen, but handle gracefully
            pass

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

# EVOLVE-BLOCK-END
