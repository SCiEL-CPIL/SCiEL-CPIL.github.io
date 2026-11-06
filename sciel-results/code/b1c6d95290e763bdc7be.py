"""DAG pebbling: communication-minimizing schedule via bottom-up pebbling."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    t_start = time.time()
    DEADLINE = 80.0  # safety margin under 90s wall-clock

    # ---- Parse DAG structure ----
    # dag: dict with 'parents' (topo-ordered parents list per vertex) and 'outputs'
    parents = dag.get('parents', dag.get('parent', None))
    outputs = dag.get('outputs', dag.get('output', None))

    # Build adjacency: children[v] = list of vertices that have v as a parent
    # Determine vertex set
    if parents is None:
        # fallback: try 'vertices' or infer
        vertices = list(dag.get('vertices', []))
        parents = {v: [] for v in vertices}
    else:
        # parents may be dict or list
        if isinstance(parents, dict):
            pmap = parents
        else:
            pmap = {i: list(parents[i]) for i in range(len(parents))}

    all_vertices = set(pmap.keys())
    for v, ps in pmap.items():
        for p in ps:
            all_vertices.add(p)
    # vertices with no outgoing edges (sinks) that aren't in pmap as keys
    # Actually pmap keys are vertices that have parents; leaves (inputs) may not have parents
    # Let's ensure all vertices are keys
    for v in list(all_vertices):
        if v not in pmap:
            pmap[v] = []

    if outputs is None:
        # outputs are vertices with no children (sinks)
        has_child = set()
        for v, ps in pmap.items():
            for p in ps:
                has_child.add(p)
        outputs = [v for v in pmap.keys() if v not in has_child]
    else:
        if isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
        elif isinstance(outputs, set):
            outputs = list(outputs)

    outputs_set = set(outputs)

    # children map
    children = {v: [] for v in pmap}
    for v, ps in pmap.items():
        for p in ps:
            children[p].append(v)

    # in-degree for topological order (already given, but let's be safe)
    # pmap is topologically ordered per problem statement

    # ---- Bottom-up pebbling schedule ----
    # Strategy: process vertices in topological order (inputs first).
    # For each vertex, load its parents (if not in fast memory), compute, store result.
    # Keep results in fast memory until all children are computed, then delete.
    #
    # To minimize transfers, we want to keep as many intermediate results in fast memory
    # as possible. The budget limits how many red values we can hold simultaneously.
    #
    # Greedy approach: process in topological order. For each vertex v:
    #   - For each parent p of v: if p is not in fast memory, load it (transfer +1)
    #   - Compute v (no transfer cost for compute itself)
    #   - Store v in fast memory (transfer +1)
    #   - If all children of v have been processed, delete v (free, no transfer)
    #
    # But we may exceed fast_memory_budget. If so, we must evict (store to slow) some value.
    #
    # Better approach: Use a reference-counting scheme.
    # Process vertices in topological order. Maintain a set of vertices in fast memory.
    # When adding v, if len(fast_memory) >= budget, evict a vertex that won't be needed soon.
    #
    # Actually, the simplest optimal-ish strategy for layered DAGs:
    # Process layer by layer. Within a layer, compute all vertices.
    # Between layers, we need the previous layer's outputs to be in fast memory.
    #
    # Let me use a straightforward greedy with eviction based on "last use" time.

    # Compute topological order (should already be given, but let's verify/use)
    # Since parents are topologically ordered, we can process in the order of pmap keys
    # but we need a proper topo order. Let's do Kahn's algorithm.
    in_deg = {v: len(pmap[v]) for v in pmap}
    from collections import deque
    queue = deque([v for v in pmap if in_deg[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)

    # If topo_order doesn't cover all vertices (cycles?), fall back to pmap order
    if len(topo_order) != len(pmap):
        topo_order = list(pmap.keys())

    # Compute "last use" for each vertex: the last position in topo_order where v is needed
    # v is needed when computing any of its children
    last_use = {v: -1 for v in pmap}
    for i, v in enumerate(topo_order):
        for p in pmap[v]:
            last_use[p] = max(last_use[p], i)
    # Outputs are used at the end (they must finish blue)
    for v in outputs:
        last_use[v] = len(topo_order) - 1

    # Greedy pebbling with LRU-like eviction
    fast_memory = {}  # vertex -> True (in fast memory)
    operations = []
    num_transfers = 0

    for i, v in enumerate(topo_order):
        if time.time() - t_start > DEADLINE:
            # Return what we have; if incomplete, the verifier will reject,
            # but we need to return something. Pad with valid ops if possible.
            break

        # Load parents that are not in fast memory
        for p in pmap[v]:
            if p not in fast_memory:
                # Check if we need to evict
                if len(fast_memory) >= fast_memory_budget:
                    # Evict the vertex with the farthest last_use that is not an output
                    # and is not a parent of v
                    evict_candidates = [x for x in fast_memory if x != v and x not in pmap[v]]
                    if evict_candidates:
                        evict_v = min(evict_candidates, key=lambda x: last_use[x])
                        del fast_memory[evict_v]
                        operations.append(['delete', evict_v])
                    else:
                        # All in fast memory are parents of v; evict the one with farthest last_use
                        evict_v = min(pmap[v], key=lambda x: last_use[x])
                        del fast_memory[evict_v]
                        operations.append(['delete', evict_v])
                operations.append(['load', p])
                num_transfers += 1
                fast_memory[p] = True

        # Compute v
        operations.append(['compute', v])

        # Store v in fast memory
        if len(fast_memory) >= fast_memory_budget and v not in fast_memory:
            # Need to evict something
            evict_candidates = [x for x in fast_memory if x != v]
            if evict_candidates:
                evict_v = min(evict_candidates, key=lambda x: last_use[x])
                del fast_memory[evict_v]
                operations.append(['delete', evict_v])
        fast_memory[v] = True
        operations.append(['store', v])
        num_transfers += 1

        # Delete parents that are no longer needed
        for p in pmap[v]:
            if p in fast_memory and last_use[p] <= i:
                del fast_memory[p]
                operations.append(['delete', p])

    # Ensure outputs are in fast memory (blue) at the end
    for v in outputs:
        if v not in fast_memory:
            # Load and compute if needed
            if time.time() - t_start > DEADLINE:
                break
            # Simple: just load and store (recompute if necessary)
            # This is a fallback; the main loop should have handled it
            for p in pmap[v]:
                if p not in fast_memory:
                    if len(fast_memory) >= fast_memory_budget:
                        evict_v = min([x for x in fast_memory if x != v], key=lambda x: last_use[x])
                        del fast_memory[evict_v]
                        operations.append(['delete', evict_v])
                    operations.append(['load', p])
                    num_transfers += 1
                    fast_memory[p] = True
            operations.append(['compute', v])
            fast_memory[v] = True
            operations.append(['store', v])
            num_transfers += 1

    return operations
# EVOLVE-BLOCK-END
