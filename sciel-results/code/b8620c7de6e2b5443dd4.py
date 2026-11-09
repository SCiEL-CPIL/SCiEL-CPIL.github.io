"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 85.0

    # Parse the DAG structure
    if hasattr(dag, 'num_vertices'):
        n = dag.num_vertices
    elif hasattr(dag, 'n'):
        n = dag.n
    elif isinstance(dag, dict) and 'n' in dag:
        n = dag['n']
    else:
        if hasattr(dag, 'vertices'):
            n = len(dag.vertices)
        else:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)

    def get_parents(v):
        if hasattr(dag, 'parents'):
            return dag.parents[v]
        elif isinstance(dag, dict) and 'parents' in dag:
            return dag['parents'][v]
        else:
            return []

    def get_outputs():
        if hasattr(dag, 'outputs'):
            return set(dag.outputs)
        elif isinstance(dag, dict) and 'outputs' in dag:
            return set(dag['outputs'])
        else:
            return {n-1}

    outputs = get_outputs()

    # Build efficient data structures
    parents_list = [get_parents(v) for v in range(n)]

    # Compute children for each vertex
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents_list[v]:
            children[p].append(v)

    # Compute remaining uses (how many children still need this vertex)
    remaining_uses = [0] * n
    for v in range(n):
        remaining_uses[v] = len(children[v])

    # Compute depth (topological level) for hierarchical decomposition
    depth = [0] * n
    for v in range(n):
        if parents_list[v]:
            depth[v] = max(depth[p] for p in parents_list[v]) + 1

    # Compute topological order via Kahn's algorithm
    in_degree = [0] * n
    for v in range(n):
        for p in parents_list[v]:
            in_degree[v] += 1

    from collections import deque
    queue = deque()
    for v in range(n):
        if in_degree[v] == 0:
            queue.append(v)

    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    red = set()  # vertices currently in fast memory
    operations = []

    # Pre-compute next-use position for each vertex (earliest topological index where it's needed as parent)
    # next_use[v] = the smallest topo index i such that v is a parent of topo_order[i]
    # We'll compute this during the main loop by tracking remaining children in topo order.
    
    # Build a list of children in topological order for each vertex
    children_topo = [[] for _ in range(n)]
    for v in range(n):
        for p in parents_list[v]:
            children_topo[p].append(v)
    # Sort each children list by topological position
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    for v in range(n):
        children_topo[v].sort(key=lambda c: topo_pos[c])
    
    # next_use_idx[v] will be the index into children_topo[v] of the next child to be processed
    next_use_idx = [0] * n
    
    def get_next_use_position(v):
        """Get the topological position of the next child that needs v, or infinity if none"""
        idx = next_use_idx[v]
        if idx < len(children_topo[v]):
            return topo_pos[children_topo[v][idx]]
        return float('inf')
    
    def evict_lowest_priority():
        """Evict vertex with lowest priority: prefer to keep vertices with soonest next use"""
        if not red:
            return None
        best_vertex = None
        best_key = (float('inf'), float('inf'), float('inf'))
        for v in red:
            # Priority: keep vertices with soonest next use (lower position = keep)
            # Evict vertices with latest next use (highest position = evict)
            # Also consider remaining uses as tiebreaker
            next_pos = get_next_use_position(v)
            key = (next_pos, remaining_uses[v], -depth[v])
            # We want to evict the one with the LARGEST key (latest next use, fewest remaining uses)
            if best_vertex is None or key > best_key:
                best_key = key
                best_vertex = v
        if best_vertex is not None:
            operations.append(['store', best_vertex])
            red.discard(best_vertex)

    def ensure_space():
        """Ensure fast memory has space for one more vertex"""
        while len(red) >= fast_memory_budget:
            evict_lowest_priority()

    def load_vertex(v):
        """Load vertex v into fast memory if not already there"""
        if v not in red:
            ensure_space()
            if v not in red:
                operations.append(['load', v])
                red.add(v)

    # Process vertices in topological order
    for v in topo_order:
        if time.time() - start_time > deadline - 10:
            break

        if not parents_list[v]:
            # Input vertex: load from slow to fast
            load_vertex(v)
        else:
            # Has parents: need all parents in fast memory to compute
            for p in parents_list[v]:
                if p not in red:
                    load_vertex(p)

            # Compute v
            operations.append(['compute', v])
            red.add(v)

        # Decrement remaining uses for parents and update next_use_idx
        for p in parents_list[v]:
            remaining_uses[p] -= 1
            # Advance next_use_idx for parent p
            if next_use_idx[p] < len(children_topo[p]):
                next_use_idx[p] += 1

        # After each vertex, evict vertices with 0 remaining uses (they're dead)
        if len(red) > fast_memory_budget:
            zero_use = [v2 for v2 in red if remaining_uses[v2] == 0]
            for v2 in zero_use:
                operations.append(['store', v2])
                red.discard(v2)

            # If still over budget, evict lowest priority
            while len(red) >= fast_memory_budget:
                evict_lowest_priority()

    # Ensure all outputs are blue (stored back to slow memory)
    for v in outputs:
        if v in red:
            operations.append(['store', v])
            red.discard(v)

    # Check time and operation limit
    if len(operations) > 20000:
        # Truncate but try to keep outputs stored
        operations = operations[:20000]
        # Re-ensure outputs are stored
        for v in outputs:
            if v in red:
                operations.append(['store', v])
                red.discard(v)

    return operations
# EVOLVE-BLOCK-END
