"""Improved pebbling algorithm with better eviction and layer-aware scheduling."""
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    def time_check():
        return time.time() - start_time > DEADLINE

    # Parse the DAG
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

    # Build children mapping and compute layers
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Compute topological layers (longest path from inputs)
    layer = {}
    for v in input_vertices:
        layer[v] = 0

    in_deg = {v: len(parents.get(v, [])) for v in all_vertices}
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

    # For each vertex, compute:
    # 1. next_use_layer: earliest layer where this vertex is needed as a parent
    # 2. use_count: how many children use this vertex
    # 3. last_use_layer: latest layer where this vertex is needed
    next_use_layer = {}
    last_use_layer = {}
    use_count = {}
    for v in all_vertices:
        cl = children.get(v, [])
        use_count[v] = len(cl)
        if cl:
            child_layers = [layer.get(c, 0) for c in cl]
            next_use_layer[v] = min(child_layers)
            last_use_layer[v] = max(child_layers)
        else:
            next_use_layer[v] = max_L + 1
            last_use_layer[v] = max_L + 1

    # Count how many times each layer needs to load a given parent
    # This helps prioritize loading parents that will be reused

    # Strategy: Process layer by layer. For each layer:
    # 1. Determine which parents are needed
    # 2. Load only those parents that are in slow memory and will be used
    # 3. Compute vertices
    # 4. Evict vertices that are no longer needed (last_use < current layer)
    # 5. If still over budget, evict based on a scoring function

    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []

    # Pre-compute for each layer, which parents are needed
    layer_parents_needed = {}
    for L in range(max_L + 1):
        needed = set()
        for v in layers.get(L, []):
            if v not in input_vertices:
                for p in parents.get(v, []):
                    needed.add(p)
        layer_parents_needed[L] = needed

    # Track which vertices are "live" (needed in future layers)
    for L in range(max_L + 1):
        if time_check():
            break
        lv = layers.get(L, [])
        if not lv:
            continue

        # Determine which parents we need for this layer
        needed_parents = layer_parents_needed[L]

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

        # Evict vertices that are no longer needed
        # First pass: evict vertices whose last_use_layer < L (completely done)
        while len(fast_mem) > fast_memory_budget and not time_check():
            # Find a vertex to evict
            best = None
            best_score = float('inf')

            for u in fast_mem:
                # Score: lower is better to evict
                if u in outputs:
                    # Must keep or store outputs
                    score = 1000000
                elif u in input_vertices:
                    # Cheap to reload
                    score = 1
                else:
                    nu = next_use_layer.get(u, max_L + 1)
                    lu = last_use_layer.get(u, max_L + 1)
                    uc = use_count.get(u, 0)

                    if lu < L:
                        # No longer needed at all
                        score = 2
                    elif nu > max_L:
                        # Never needed again
                        score = 3
                    else:
                        # Still needed; score based on urgency
                        # Earlier next use = keep longer
                        # More uses = keep longer
                        # Later last use = keep longer
                        score = 100000 - (nu * 1000 + uc * 50 + lu * 10)
                        # If needed very soon, keep it
                        if nu <= L + 1:
                            score += 500000

                if score < best_score:
                    best_score = score
                    best = u

            if best is None:
                break

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
            # Already in slow memory, nothing to do
            pass
        else:
            # Output not in memory - this shouldn't happen but handle gracefully
            pass

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
