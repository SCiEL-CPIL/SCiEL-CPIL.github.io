"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Optimize DAG pebbling by minimizing transfers using fan-out persistence cost heuristic.
    
    Strategy: 
    1. Use topological order for computation
    2. Prioritize keeping high-fan-out vertices in fast memory
    3. Evict vertices with lowest persistence cost (few remaining dependents * distance to sink)
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        outputs = dag.get('outputs', [])
        if not vertices:
            vertices = dag.get('nodes', [])
            edges = dag.get('arcs', [])
            outputs = dag.get('sinks', [])
    elif isinstance(dag, (list, tuple)):
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
    else:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if not vertices:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Build adjacency structures
    parents_map = defaultdict(list)
    children_map = defaultdict(list)
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
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if time.time() > deadline:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Identify input and output vertices
    input_vertices = [v for v in topo_order if not parents_map.get(v)]
    output_vertices = [v for v in topo_order if v in output_set]
    
    if not output_vertices:
        output_vertices = [v for v in topo_order if not children_map.get(v)]
    
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Precompute persistence cost: for each vertex, count remaining dependents and distance to sink
    # We'll compute this dynamically as we process vertices
    
    # Initialize state
    fast_mem = set()
    slow_mem = set(all_vertices)  # All vertices start in slow memory
    operations = []
    computed = set()
    remaining_children = {v: len(children_map.get(v, [])) for v in all_vertices}
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        # Load parents that are not in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                operations.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        fast_mem.add(v)
        
        # If over budget, evict vertices with lowest persistence cost
        while len(fast_mem) > fast_memory_budget:
            # Calculate persistence cost for each vertex in fast memory
            # Cost = remaining_children * (distance to nearest sink)
            # We want to keep vertices with HIGH cost, evict those with LOW cost
            
            best_evict = None
            best_cost = float('inf')
            
            for candidate in fast_mem:
                # Don't evict if it's an output vertex that hasn't been finalized yet
                if candidate in output_set and candidate not in computed:
                    continue
                    
                # Calculate remaining dependents
                rem = remaining_children.get(candidate, 0)
                
                # If no remaining children, cost is 0 (can be evicted)
                if rem == 0:
                    cost = 0
                else:
                    # Estimate distance to sink: average distance from this vertex to any sink
                    # For simplicity, use 1 as a proxy (will be refined)
                    cost = rem * 1  # Simple: just use remaining count
                
                if cost < best_cost:
                    best_cost = cost
                    best_evict = candidate
            
            if best_evict is None:
                # All are outputs, evict the one with lowest remaining children
                best_evict = min(fast_mem, key=lambda x: remaining_children.get(x, 0))
            
            operations.append(['store', best_evict])
            fast_mem.discard(best_evict)
            slow_mem.add(best_evict)
        
        # Update remaining children count for this vertex's parents
        for p in parents_map.get(v, []):
            if p in remaining_children:
                remaining_children[p] -= 1
    
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
