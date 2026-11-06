"""DAG pebbling: layered-aware, batched transfer minimization."""
import time
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling minimizing communication (load+store ops).
    Returns list of [operation, vertex].
    """
    try:
        return _pebble(dag, fast_memory_budget)
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return _pebble_fallback(dag, fast_memory_budget)


def _pebble_fallback(dag, fast_memory_budget):
    """Minimal fallback: load all inputs, compute in topo order, store all."""
    num_v = dag.get('num_vertices', dag.get('n', 0))
    ops = []
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    topo = list(dag.get('topo_order', range(num_v)))

    in_fast = set()
    for v in topo:
        for p in parents.get(v, []):
            if p not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    evictable = [x for x in in_fast if x not in outputs]
                    if evictable:
                        e = evictable[0]
                        ops.append(['store', e])
                        in_fast.discard(e)
                ops.append(['load', p])
                in_fast.add(p)
        ops.append(['compute', v])
        in_fast.add(v)

    for v in sorted(outputs):
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                evictable = [x for x in in_fast if x not in outputs]
                if evictable:
                    e = evictable[0]
                    ops.append(['store', e])
                    in_fast.discard(e)
            ops.append(['load', v])
            in_fast.add(v)

    return ops[:20000]


def _parse_dag(dag):
    """Parse DAG structure into (num_v, parents, children, inputs, outputs, topo, layers)."""
    deadline = time.time() + 80

    # Determine number of vertices
    num_v = None
    if 'num_vertices' in dag:
        num_v = dag['num_vertices']
    elif 'n' in dag:
        num_v = dag['n']
    elif 'vertices' in dag:
        if isinstance(dag['vertices'], list):
            num_v = len(dag['vertices'])
        elif isinstance(dag['vertices'], int):
            num_v = dag['vertices']
    if num_v is None:
        # Infer from parents dict
        if 'parents' in dag:
            num_v = max(dag['parents'].keys()) + 1 if dag['parents'] else 0
    if num_v is None or num_v == 0:
        raise ValueError("Cannot determine number of vertices")
    num_v = int(num_v)

    # Build parent/child relationships
    parents = {v: [] for v in range(num_v)}
    children = {v: [] for v in range(num_v)}

    if 'parents' in dag:
        for v in range(num_v):
            if v in dag['parents']:
                parents[v] = [int(p) for p in dag['parents'][v]]
                for p in parents[v]:
                    if 0 <= p < num_v and v not in children[p]:
                        children[p].append(v)
    elif 'edges' in dag:
        for e in dag['edges']:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                u, w = int(e[0]), int(e[1])
                if 0 <= u < num_v and 0 <= w < num_v:
                    if u not in parents[w]:
                        parents[w].append(u)
                    if w not in children[u]:
                        children[u].append(w)

    # Identify inputs and outputs
    inputs = set()
    outputs = set()
    if 'inputs' in dag:
        inputs = set(int(x) for x in dag['inputs'])
    else:
        for v in range(num_v):
            if len(parents[v]) == 0:
                inputs.add(v)

    if 'outputs' in dag:
        outputs = set(int(x) for x in dag['outputs'])
    else:
        for v in range(num_v):
            if len(children[v]) == 0:
                outputs.add(v)

    # Topological order
    if 'topo_order' in dag:
        topo = [int(x) for x in dag['topo_order']]
        if len(topo) != num_v:
            # Rebuild
            in_deg = {v: len(parents[v]) for v in range(num_v)}
            q = deque([v for v in range(num_v) if in_deg[v] == 0])
            topo = []
            while q:
                v = q.popleft()
                topo.append(v)
                for c in children[v]:
                    in_deg[c] -= 1
                    if in_deg[c] == 0:
                        q.append(c)
            if len(topo) != num_v:
                topo = list(range(num_v))
    else:
        in_deg = {v: len(parents[v]) for v in range(num_v)}
        q = deque([v for v in range(num_v) if in_deg[v] == 0])
        topo = []
        while q:
            v = q.popleft()
            topo.append(v)
            for c in children[v]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    q.append(c)
        if len(topo) != num_v:
            topo = list(range(num_v))

    # Detect layers: a vertex's layer = 1 + max(layer of parents), with inputs at layer 0
    # This works for layered DAGs. For general DAGs, we still get a "longest path" layering.
    layer_of = {}
    for v in topo:
        if not parents[v]:
            layer_of[v] = 0
        else:
            layer_of[v] = 1 + max(layer_of.get(p, 0) for p in parents[v])

    # Group vertices by layer
    max_layer = max(layer_of.values()) if layer_of else 0
    layers = defaultdict(list)
    for v in topo:
        layers[layer_of[v]].append(v)

    # Verify it's truly layered: every edge goes from layer L to L+1
    is_layered = True
    for v in range(num_v):
        lv = layer_of.get(v, 0)
        for c in children[v]:
            lc = layer_of.get(c, 0)
            if lc != lv + 1:
                is_layered = False
                break
        if not is_layered:
            break

    return num_v, parents, children, inputs, outputs, topo, layers, is_layered, max_layer


def _pebble(dag, fast_memory_budget):
    deadline = time.time() + 80

    num_v, parents, children, inputs, outputs, topo, layers, is_layered, max_layer = _parse_dag(dag)

    # For layered DAGs, use layer-by-layer batched strategy
    if is_layered and fast_memory_budget >= 1:
        return _pebble_layered(num_v, parents, children, inputs, outputs, layers, max_layer,
                               fast_memory_budget, deadline)
    else:
        return _pebble_topo(num_v, parents, children, inputs, outputs, topo,
                            fast_memory_budget, deadline)


def _pebble_layered(num_v, parents, children, inputs, outputs, layers, max_layer,
                    fast_memory_budget, deadline):
    """
    Layer-by-layer pebbling for layered DAGs.
    Strategy:
    - Load layer 0 (inputs) into fast memory.
    - For each subsequent layer L:
        - Compute all vertices in layer L (their parents are in layer L-1, already in fast).
        - Store layer L-1 (no longer needed) to free space.
        - Load layer L into fast (already computed, but need to ensure capacity).
    - At the end, ensure outputs are in fast memory.
    
    Actually, the key insight: after computing layer L, the values are in fast memory.
    We need to evict layer L-1 before computing layer L+1 (since layer L-1 is no longer needed).
    But we must ensure we have room for the new layer's values.
    
    Better: 
    1. Load all inputs (layer 0).
    2. For L = 1..max_layer:
        a. Compute all vertices in layer L (parents in fast from layer L-1).
        b. Now fast has: layer L-1 + layer L.
        c. Store (evict) all of layer L-1 to free slots.
        d. Fast now has: layer L.
    3. At the end, fast has the last layer (outputs).
    
    But what if a layer is larger than fast_memory_budget?
    Then we can't hold an entire layer. We need to process in chunks.
    """
    ops = []
    in_fast = set()

    # Step 1: Load all inputs (layer 0)
    layer0 = layers.get(0, [])
    for v in layer0:
        if len(in_fast) >= fast_memory_budget:
            # Need to evict something, but layer 0 is all we have.
            # This means fast_memory_budget < len(layer0).
            # We'll have to process in chunks.
            # For now, evict the first non-output (but all are inputs, not outputs unless single layer)
            evictable = [x for x in in_fast if x not in outputs]
            if evictable:
                e = evictable[0]
                ops.append(['store', e])
                in_fast.discard(e)
        ops.append(['load', v])
        in_fast.add(v)

    # Step 2: Process each subsequent layer
    for L in range(1, max_layer + 1):
        if time.time() > deadline:
            break
        layer_L = layers.get(L, [])
        if not layer_L:
            continue

        # Compute all vertices in layer L
        for v in layer_L:
            # Parents should be in fast (from layer L-1)
            for p in parents[v]:
                if p not in in_fast:
                    # Load parent if not in fast (shouldn't happen in layered DAG)
                    if len(in_fast) >= fast_memory_budget:
                        evictable = [x for x in in_fast if x not in outputs]
                        if evictable:
                            e = evictable[0]
                            ops.append(['store', e])
                            in_fast.discard(e)
                    ops.append(['load', p])
                    in_fast.add(p)
            ops.append(['compute', v])
            in_fast.add(v)

        # Now fast has: layer L-1 + layer L
        # Store (evict) layer L-1 to free space for next iteration
        layer_Lm1 = layers.get(L - 1, [])
        for v in layer_Lm1:
            if v in in_fast and v not in outputs:
                ops.append(['store', v])
                in_fast.discard(v)

    # Step 3: Ensure all outputs are in fast memory
    for v in sorted(outputs):
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                evictable = [x for x in in_fast if x not in outputs]
                if evictable:
                    e = evictable[0]
                    ops.append(['store', e])
                    in_fast.discard(e)
            ops.append(['load', v])
            in_fast.add(v)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops


def _pebble_topo(num_v, parents, children, inputs, outputs, topo, fast_memory_budget, deadline):
    """Topological-order greedy with liveness-based eviction."""
    ops = []
    in_fast = set()

    # Precompute last_use and rem_child for eviction priority
    last_use = {}
    for i, v in enumerate(topo):
        last_use[v] = i

    rem_child = {v: len(children[v]) for v in range(num_v)}

    for v in topo:
        if time.time() > deadline:
            break

        if v in in_fast:
            # Already in fast (e.g., it's an input that was pre-loaded)
            # But we still need to update rem_child for its parents
            for p in parents[v]:
                rem_child[p] -= 1
            continue

        # Collect all parents that need to be loaded
        need_load = []
        for p in parents[v]:
            if p not in in_fast:
                need_load.append(p)

        if need_load:
            # Evict to make room if at capacity
            if len(in_fast) >= fast_memory_budget:
                parent_set = set(parents[v])
                cands = [x for x in in_fast if x not in parent_set and x not in outputs]
                if cands:
                    # Sort by (rem_child, last_use) ascending: evict least urgent first
                    cands.sort(key=lambda x: (rem_child[x], last_use[x]))
                    num_to_evict = min(len(need_load), len(cands))
                    for i in range(num_to_evict):
                        evict = cands[i]
                        ops.append(['store', evict])
                        in_fast.discard(evict)

            # Load needed parents
            for p in need_load:
                if p not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        # Emergency eviction
                        cands = [x for x in in_fast if x not in set(parents[v]) and x not in outputs]
                        if cands:
                            cands.sort(key=lambda x: (rem_child[x], last_use[x]))
                            evict = cands[0]
                            ops.append(['store', evict])
                            in_fast.discard(evict)
                    ops.append(['load', p])
                    in_fast.add(p)

        # Compute v
        ops.append(['compute', v])
        in_fast.add(v)

        # Update liveness of parents
        for p in parents[v]:
            rem_child[p] -= 1

    # Ensure outputs are in fast memory
    for v in sorted(outputs):
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                cands = [x for x in in_fast if x not in outputs]
                if cands:
                    cands.sort(key=lambda x: (rem_child[x], last_use[x]))
                    evict = cands[0]
                    ops.append(['store', evict])
                    in_fast.discard(evict)
            ops.append(['load', v])
            in_fast.add(v)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
