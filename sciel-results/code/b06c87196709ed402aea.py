"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = 80.0  # safety margin before 90s kill

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
            return []

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
            return {n - 1}

    outputs = get_outputs()

    # Build efficient data structures
    parents_list = [list(get_parents(v)) for v in range(n)]

    # Compute children for each vertex
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents_list[v]:
            children[p].append(v)

    # Compute remaining uses (how many children still need this vertex)
    remaining_uses = [0] * n
    for v in range(n):
        remaining_uses[v] = len(children[v])

    # Compute depth (topological level)
    depth = [0] * n
    for v in range(n):
        if parents_list[v]:
            depth[v] = max(depth[p] for p in parents_list[v]) + 1

    # Compute topological order via Kahn's algorithm
    in_degree = [0] * n
    for v in range(n):
        for p in parents_list[v]:
            in_degree[v] += 1

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

    # If topological order doesn't cover all vertices (shouldn't happen for valid DAG),
    # add remaining vertices
    if len(topo_order) < n:
        for v in range(n):
            if v not in set(topo_order):
                topo_order.append(v)

    # Identify input vertices (no parents)
    input_vertices = set(v for v in range(n) if not parents_list[v])

    # Compute next_use and last_use for each vertex in topological order
    next_use = [n] * n
    last_use = [-1] * n
    for idx, v in enumerate(topo_order):
        for p in parents_list[v]:
            if next_use[p] == n:
                next_use[p] = idx
            last_use[p] = idx

    # For vertices with no children, next_use and last_use remain n / -1
    # These are "sink" vertices that don't need to be kept after computation

    # Improved greedy approach with better eviction strategy
    red = set()  # vertices currently in fast memory
    computed = set()  # vertices that have been computed at least once
    operations = []

    # Precompute a priority score for eviction
    # We want to keep vertices that will be needed soon and have high remaining uses
    def eviction_score(v):
        """Lower score = evict first. We want to keep high-score vertices."""
        # Vertices with 0 remaining uses should be evicted immediately
        if remaining_uses[v] == 0:
            return -1
        # Prefer to evict vertices with fewer remaining uses
        # Among same remaining uses, evict those needed further in the future
        # And those at lower depth (less critical)
        # Use a weighted combination
        return (remaining_uses[v], next_use[v], -depth[v])

    def ensure_space():
        """Ensure fast memory has space for one more vertex"""
        while len(red) >= fast_memory_budget:
            if not red:
                break
            # Find vertex with lowest eviction score
            best_v = min(red, key=eviction_score)
            operations.append(['store', best_v])
            red.discard(best_v)

    def load_vertex(v):
        """Load vertex v into fast memory if not already there"""
        if v not in red:
            ensure_space()
            if v not in red:
                # Check if v is available (input or computed)
                if v in input_vertices or v in computed:
                    operations.append(['load', v])
                    red.add(v)
                else:
                    # Should not happen if we process in topological order
                    operations.append(['load', v])
                    red.add(v)

    # Process vertices in topological order
    for v in topo_order:
        if time.time() - t0 > deadline - 10:
            break

        if not parents_list[v]:
            # Input vertex: load from slow to fast
            load_vertex(v)
            # Inputs are "computed" by definition
            computed.add(v)
        else:
            # Has parents: need all parents in fast memory to compute
            for p in parents_list[v]:
                if p not in red:
                    load_vertex(p)

            # Compute v
            operations.append(['compute', v])
            red.add(v)
            computed.add(v)

        # Decrement remaining uses for parents
        for p in parents_list[v]:
            remaining_uses[p] -= 1

        # After each vertex, evict vertices with 0 remaining uses
        if len(red) > fast_memory_budget:
            zero_use = [v2 for v2 in red if remaining_uses[v2] == 0]
            for v2 in zero_use:
                operations.append(['store', v2])
                red.discard(v2)

            # If still over budget, evict lowest priority
            while len(red) >= fast_memory_budget:
                if not red:
                    break
                best_v = min(red, key=eviction_score)
                operations.append(['store', best_v])
                red.discard(best_v)

    # Ensure all outputs are blue (stored back to slow memory)
    for v in outputs:
        if v in red:
            operations.append(['store', v])
            red.discard(v)

    # Check operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
        # Ensure outputs are still handled
        for v in outputs:
            if v in red:
                operations.append(['store', v])

    return operations
# EVOLVE-BLOCK-END
