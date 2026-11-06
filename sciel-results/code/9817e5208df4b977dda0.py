"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Optimize DAG pebbling by minimizing transfers.
    
    Strategy: Use a greedy approach that prioritizes:
    1. Computing nodes whose parents are already in fast memory
    2. Keeping nodes in fast memory that will be needed soon
    3. Minimizing redundant loads/stores
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Parse DAG structure
    # dag format: list of (vertex, parents_list, is_output)
    # Or some other format - need to handle generically
    
    # Let's first try to understand the dag structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        outputs = dag.get('outputs', [])
        if not vertices:
            # Try alternate keys
            vertices = dag.get('nodes', [])
            edges = dag.get('arcs', [])
            outputs = dag.get('sinks', [])
    elif isinstance(dag, (list, tuple)):
        # Assume list of (vertex, parents, is_output) tuples
        vertices = []
        edges = []
        outputs = []
        for item in dag:
            if isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    v, parents, is_out = item[0], item[1], item[2]
                    vertices.append(v)
                    outputs.append(v) if is_out else None
                    for p in parents:
                        edges.append((p, v))
                elif len(item) == 2:
                    v, parents = item[0], item[1]
                    vertices.append(v)
                    for p in parents:
                        edges.append((p, v))
    else:
        # Fallback to baseline
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # If we can't parse the structure, use baseline
    if not vertices:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Build adjacency structures
    parents_map = defaultdict(list)  # vertex -> list of parent vertices
    children_map = defaultdict(list)  # vertex -> list of child vertices
    all_vertices = set()
    output_set = set()
    
    for v in vertices:
        all_vertices.add(v)
    
    for edge in edges:
        if len(edge) == 2:
            p, c = edge[0], edge[1]
            parents_map[c].append(p)
            children_map[p].append(c)
            all_vertices.add(p)
            all_vertices.add(c)
    
    for v in outputs:
        output_set.add(v)
        all_vertices.add(v)
    
    if not all_vertices:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Topological sort
    in_degree = {v: 0 for v in all_vertices}
    for v in all_vertices:
        for p in parents_map.get(v, []):
            in_degree[v] += 1
    
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(all_vertices):
        # Cycle detected or parsing issue, use baseline
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Check time
    if time.time() > deadline:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Greedy pebbling strategy
    # State: fast_memory (set of vertices in fast memory), slow_memory (set of vertices in slow memory)
    # We want to minimize transfers (load, store operations)
    
    # Identify input vertices (no parents) and output vertices
    input_vertices = [v for v in topo_order if not parents_map.get(v)]
    output_vertices = [v for v in topo_order if v in output_set]
    
    # If no outputs specified, assume all sinks are outputs
    if not output_vertices:
        output_vertices = [v for v in topo_order if not children_map.get(v)]
    
    # Compute for each vertex how many times it's needed (fan-out consideration)
    # and its position in topological order
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Greedy algorithm:
    # Process in topological order, but use fast memory strategically
    operations = []
    fast_mem = set()  # vertices currently in fast memory
    slow_mem = set()  # vertices currently in slow memory (computed)
    computed = set()  # all vertices that have been computed
    
    # First, load all input vertices that we'll need
    # Strategy: Load inputs just-in-time based on when they're needed
    
    # Precompute: for each vertex, which children need it
    # and estimate when we should load/store
    
    # Simple greedy: process in topo order, load parent when needed, compute, keep in fast if budget allows
    
    # Better: Use a priority queue for which vertices to keep in fast memory
    # based on: (1) how soon they'll be needed again, (2) their fan-out
    
    # For now, implement a reasonable greedy:
    # - Load inputs when first needed
    # - Compute in topo order
    # - Keep results in fast memory if budget allows, prioritizing those needed soon
    
    # Track when each vertex is next needed
    next_need = {}
    for v in topo_order:
        # v is needed by its children
        for c in children_map.get(v, []):
            if v not in next_need or topo_pos[c] < next_need[v]:
                next_need[v] = topo_pos[c]
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
            
        # Load parents that are not in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                # Load from slow memory
                operations.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        fast_mem.add(v)
        
        # If over budget, evict something
        while len(fast_mem) > fast_memory_budget:
            # Find vertex to evict: least recently needed or lowest priority
            # Priority: vertices needed soonest should stay
            candidates = list(fast_mem)
            # Sort by next need time (later is better to evict)
            candidates.sort(key=lambda x: next_need.get(x, float('inf')), reverse=True)
            # Evict the one needed latest (or never)
            if candidates:
                to_evict = candidates[0]
                operations.append(['store', to_evict])
                fast_mem.discard(to_evict)
                slow_mem.add(to_evict)
    
    # Ensure all outputs are in fast memory at the end
    for v in output_vertices:
        if v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
            slow_mem.discard(v)
    
    # Final cleanup: store everything that's not an output
    for v in list(fast_mem):
        if v not in output_set:
            operations.append(['store', v])
            fast_mem.discard(v)
            slow_mem.add(v)
    
    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
