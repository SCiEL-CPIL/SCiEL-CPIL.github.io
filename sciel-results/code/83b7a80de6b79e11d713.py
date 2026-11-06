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
                    if is_out:
                        outputs.append(v)
                    for p in parents:
                        edges.append((p, v))
                elif len(item) == 2:
                    v, parents = item[0], item[1]
                    vertices.append(v)
                    for p in parents:
                        edges.append((p, v))
            elif isinstance(item, dict):
                v = item.get('vertex', item.get('node', item.get('id', None)))
                parents = item.get('parents', item.get('preds', []))
                is_out = item.get('is_output', item.get('output', False))
                if v is not None:
                    vertices.append(v)
                    if is_out:
                        outputs.append(v)
                    for p in parents:
                        edges.append((p, v))
    else:
        # Fallback: try to infer structure
        return _pebble_fallback(dag, fast_memory_budget)
    
    # If we can't parse the structure, use fallback
    if not vertices:
        return _pebble_fallback(dag, fast_memory_budget)
    
    # Build adjacency structures
    parents_map = defaultdict(list)  # vertex -> list of parent vertices
    children_map = defaultdict(list)  # vertex -> list of child vertices
    all_vertices = set()
    output_set = set()
    
    for v in vertices:
        all_vertices.add(v)
    
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            p, c = edge[0], edge[1]
            parents_map[c].append(p)
            children_map[p].append(c)
            all_vertices.add(p)
            all_vertices.add(c)
        elif isinstance(edge, dict):
            p = edge.get('from', edge.get('src', edge.get('parent', None)))
            c = edge.get('to', edge.get('dst', edge.get('child', None)))
            if p is not None and c is not None:
                parents_map[c].append(p)
                children_map[p].append(c)
                all_vertices.add(p)
                all_vertices.add(c)
    
    for v in outputs:
        output_set.add(v)
        all_vertices.add(v)
    
    if not all_vertices:
        return _pebble_fallback(dag, fast_memory_budget)
    
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
        # Cycle detected or parsing issue, use fallback
        return _pebble_fallback(dag, fast_memory_budget)
    
    # Check time
    if time.time() > deadline:
        return _pebble_fallback(dag, fast_memory_budget)
    
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
    
    # Precompute: for each vertex, which children need it
    # and estimate when we should load/store
    
    # Track when each vertex is next needed
    next_need = {}
    for v in topo_order:
        # v is needed by its children
        min_child_pos = float('inf')
        for c in children_map.get(v, []):
            if c in topo_pos and topo_pos[c] < min_child_pos:
                min_child_pos = topo_pos[c]
        if min_child_pos != float('inf'):
            next_need[v] = min_child_pos
    
    # Compute recompute cost: number of ancestors (rough proxy for recompute difficulty)
    recompute_cost = {}
    for v in topo_order:
        cost = 1  # at least itself
        for p in parents_map.get(v, []):
            cost += recompute_cost.get(p, 0)
        recompute_cost[v] = cost
    
    # Process in topological order
    operations = []
    fast_mem = set()  # vertices currently in fast memory (blue)
    slow_mem = set()  # vertices currently in slow memory (red/computed)
    computed = set()  # all vertices that have been computed
    
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
            # Find vertex to evict: consider recompute cost and next need
            # Priority: keep vertices that are needed soon AND expensive to recompute
            candidates = []
            for x in fast_mem:
                if x in output_set:
                    continue  # Don't evict outputs
                need_time = next_need.get(x, float('inf'))
                cost = recompute_cost.get(x, 1)
                # Score: we want to evict vertices that are needed later and cheap to recompute
                # Use a combined score: later need + lower cost = better to evict
                if need_time == float('inf'):
                    evict_score = -cost  # Never needed again, evict based on cost
                else:
                    evict_score = -need_time + cost * 0.1  # Need soon, expensive to recompute = keep
                candidates.append((evict_score, x))
            
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                to_evict = candidates[0][1]
                operations.append(['store', to_evict])
                fast_mem.discard(to_evict)
                slow_mem.add(to_evict)
            else:
                # All are outputs, can't evict
                break
    
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


def _pebble_fallback(dag, fast_memory_budget):
    """
    Fallback pebbling strategy when DAG structure is unclear.
    """
    start_time = time.time()
    deadline = start_time + 55
    
    # Try to extract any structure we can
    operations = []
    
    # If dag is a list of dicts with vertex info
    if isinstance(dag, (list, tuple)):
        vertices = []
        for item in dag:
            if isinstance(item, dict):
                v = item.get('vertex', item.get('node', item.get('id', None)))
                if v is not None:
                    vertices.append(v)
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                vertices.append(item[0])
        
        if vertices:
            # Simple sequential processing
            for v in vertices:
                operations.append(['compute', v])
    
    # If still no operations, return empty
    if not operations:
        return []
    
    if time.time() > deadline:
        return operations[:20000] if operations else []
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
