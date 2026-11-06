"""Communication-minimizing DAG pebbling with DP-based cost optimization."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    # If we can't parse, return empty
    if vertices is None or parents is None:
        return []

    # Normalize to lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children_map = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children_map[p].append(v)
        outputs = set(v for v in vertex_list if not children_map[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children_map = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children_map[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Precompute usage counts and next-use patterns
    # For each vertex, track which children use it
    uses = {v: [] for v in needed_vertices}
    for v in needed_vertices:
        for child in children_map.get(v, []):
            if child in needed_set:
                uses[v].append(child)
        uses[v].sort()

    # Compute in-degree within needed subgraph
    in_degree = {v: 0 for v in needed_vertices}
    for v in needed_vertices:
        for p in parent_dict.get(v, []):
            if p in needed_set:
                in_degree[v] += 1

    # Topological order (already given by vertex_list order, but let's verify)
    topo_order = needed_vertices

    # DP-based pebbling: compute minimal communication cost
    # For each vertex, we compute the optimal strategy considering:
    # 1. Whether to keep it in fast memory after computation
    # 2. The cost of recomputing vs reloading
    # 
    # Key insight: For layered DAGs, we can use a recursive cost function
    # that considers the entire subtree of uses.

    # Precompute: for each vertex, the "span" - how many operations until last use
    # This helps decide whether to keep in fast memory or recompute

    # Build a cost model:
    # - load(v): cost 1, brings v from slow to fast
    # - store(v): cost 1, moves v from fast to slow
    # - compute(v): cost 0 (but requires parents in fast)
    # - delete(v): cost 0

    # Strategy: Process in topological order, but use lookahead to decide
    # whether to evict or keep vertices based on future usage patterns.

    # For layered DAGs, we can exploit the layer structure.
    # Let's detect layers: vertices at same "depth" from inputs.
    
    # Compute depth (longest path from any input)
    depth = {v: 0 for v in needed_vertices}
    for v in topo_order:
        ps = [p for p in parent_dict.get(v, []) if p in needed_set]
        if ps:
            depth[v] = max(depth[p] for p in ps) + 1
        elif v in inputs:
            depth[v] = 0

    # Group by depth
    layers = defaultdict(list)
    for v in topo_order:
        layers[depth[v]].append(v)

    # Max depth
    max_depth = max(layers.keys()) if layers else 0

    # For each layer, the "width" is the number of vertices
    # If width <= budget, we can keep all in fast memory
    # Otherwise, we need to evict/recompute

    # Optimized strategy for layered DAGs:
    # 1. Load/compute layer by layer
    # 2. For each layer, if width > budget, use a sliding window approach
    # 3. Keep only the vertices that will be needed soon in fast memory
    # 4. Recompute or reload as needed

    # Let's use a more sophisticated approach:
    # For each vertex, compute the "reuse distance" - how many layers until next use
    # If reuse distance is small, keep in fast memory
    # If reuse distance is large, consider recomputing

    # Precompute next use for each vertex
    next_use = {v: {} for v in needed_vertices}
    for v in needed_vertices:
        for i, child in enumerate(uses[v]):
            # The next use of v is at child
            next_use[v][i] = child

    # Main pebbling algorithm with cost-aware decisions
    in_fast = set()
    operations = []
    total_comm = 0

    def load(v):
        nonlocal total_comm
        operations.append(['load', v])
        in_fast.add(v)
        total_comm += 1

    def store(v):
        nonlocal total_comm
        operations.append(['store', v])
        in_fast.discard(v)
        total_comm += 1

    def compute(v):
        operations.append(['compute', v])
        computed_set.add(v)
        in_fast.add(v)

    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)

    computed_set = set()

    # Process layer by layer
    # For each layer, decide which vertices to keep in fast memory
    # based on their future usage patterns

    # First pass: load all inputs that are needed
    for v in topo_order:
        if time.time() > deadline:
            break
        if v in inputs and v in needed_set:
            if v not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    # Evict a vertex that won't be used soon
                    evict_one(topo_order.index(v))
                load(v)

    # Second pass: compute non-input vertices in topological order
    # with cost-aware eviction
    for v in topo_order:
        if time.time() > deadline:
            break
        if v in inputs:
            continue

        parents = [p for p in parent_dict.get(v, []) if p in needed_set]

        # Ensure all parents are in fast memory
        for p in parents:
            if p not in in_fast:
                # Need to bring p into fast memory
                # Check if p can be recomputed
                p_parents = [pp for pp in parent_dict.get(p, []) if pp in needed_set]
                can_recompute = all(pp in in_fast for pp in p_parents)

                if len(in_fast) >= fast_memory_budget:
                    # Need to evict something
                    # Evict the vertex with the furthest next use
                    # or the one with no future use
                    evict_for_compute(v, set(parents))

                if can_recompute:
                    compute(p)
                else:
                    load(p)

        # Compute v
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                evict_for_compute(v, set(parents))
            compute(v)

        # After computing v, decide whether to keep it in fast memory
        # or store it if it won't be used soon
        # Look ahead: check if v is used in the next few layers
        # If not, store it to free up space

        # Heuristic: if the next use of v is more than budget layers away,
        # consider storing it
        if uses[v]:
            next_child = uses[v][0]
            next_child_depth = depth.get(next_child, 0)
            current_depth = depth[v]
            gap = next_child_depth - current_depth

            # If gap is large and we're close to budget, store v
            if gap > 2 and len(in_fast) >= fast_memory_budget * 0.8:
                # Check if v has more uses
                if len(uses[v]) == 1:
                    # Only one use, and it's far away
                    # Store it now, reload later
                    store(v)

    # Ensure all outputs are in slow memory (blue)
    # "all outputs must finish blue" means they must be in slow memory
    # So if an output is in fast memory, we must store it
    for v in outputs:
        if v in needed_set and v in in_fast:
            store(v)

    # Final check: if any outputs are not computed, compute them
    for v in outputs:
        if v in needed_set and v not in computed_set:
            # This shouldn't happen, but just in case
            # Recompute from parents
            parents = [p for p in parent_dict.get(v, []) if p in needed_set]
            for p in parents:
                if p not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        evict_for_compute(v, set(parents))
                    p_parents = [pp for pp in parent_dict.get(p, []) if pp in needed_set]
                    can_recompute = all(pp in in_fast for pp in p_parents)
                    if can_recompute:
                        compute(p)
                    else:
                        load(p)
            if v not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    evict_for_compute(v, set(parents))
                compute(v)
            # Store output to make it blue
            if v in in_fast:
                store(v)

    return operations

    # Helper function to evict a vertex when space is needed
    def evict_one(current_idx):
        # Find the vertex in in_fast with the furthest next use
        best_v = None
        best_next_use = -1

        for u in in_fast:
            # Find the next use of u after current_idx
            next_use_idx = None
            for child in uses.get(u, []):
                child_idx = vertex_to_idx.get(child, -1)
                if child_idx > current_idx:
                    next_use_idx = child_idx
                    break

            if next_use_idx is None:
                # No future use, evict this one
                store(u)
                return
            else:
                if next_use_idx > best_next_use:
                    best_next_use = next_use_idx
                    best_v = u

        if best_v is not None:
            store(best_v)

    # Helper function to evict when computing a vertex
    def evict_for_compute(v, protected):
        # Find the best vertex to evict, excluding protected parents
        candidates = [u for u in in_fast if u not in protected]

        if not candidates:
            # Emergency: evict a protected vertex (shouldn't happen)
            for u in in_fast:
                store(u)
                return

        best_v = None
        best_next_use = -1

        for u in candidates:
            next_use_idx = None
            for child in uses.get(u, []):
                child_idx = vertex_to_idx.get(child, -1)
                if child_idx > vertex_to_idx.get(v, -1):
                    next_use_idx = child_idx
                    break

            if next_use_idx is None:
                store(u)
                return
            else:
                if next_use_idx > best_next_use:
                    best_next_use = next_use_idx
                    best_v = u

        if best_v is not None:
            store(best_v)
# EVOLVE-BLOCK-END
