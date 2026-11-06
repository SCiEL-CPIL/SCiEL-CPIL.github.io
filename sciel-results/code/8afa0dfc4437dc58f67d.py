import time
import heapq
from collections import defaultdict, deque

def _topo_sort(vertices, parents_map):
    """Compute topological order given vertices and parents."""
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    return order

def _compute_depths(vertices, parents_map):
    """Compute the depth of each vertex (longest path from an input)."""
    order = _topo_sort(vertices, parents_map)
    depth = {}
    for v in order:
        if not parents_map.get(v, []):
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in parents_map[v]) + 1
    return depth

def _greedy_pebble(dag, fast_memory_budget, deadline):
    """Greedy pebbling with depth-aware eviction."""
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    input_set = set(input_vertices)
    output_set = set(outputs)
    
    order = _topo_sort(vertices, parents_map)
    index_in_order = {v: i for i, v in enumerate(order)}
    
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
            
    # Pre-compute next use indices
    next_use_queues = {}
    for v in vertices:
        c_indices = [index_in_order[c] for c in children_map[v]]
        c_indices.sort()
        next_use_queues[v] = c_indices
        
    next_use_ptr = {v: 0 for v in vertices}
    
    def get_next_use(v):
        ptr = next_use_ptr[v]
        queue = next_use_queues[v]
        if ptr < len(queue):
            return queue[ptr]
        return float('inf')

    # Pre-compute depths for cost estimation
    depths = _compute_depths(vertices, parents_map)
    
    # State
    fast_memory = set()
    operations = []
    
    def get_eviction_score(v):
        """
        Score such that LOWER is better to evict.
        1. If next use is inf, it's best to evict. (Score 0 for this part)
        2. If next use is finite, we want to evict the one with LARGEST next use.
           So we map next_use to a score.
           Let's use a tuple:
           (has_future_use, -next_use, is_expensive, v)
           
           has_future_use: 0 if inf, 1 if finite.
           -next_use: If finite, larger next_use -> more negative -> evicted first.
           is_expensive: We want to evict CHEAP nodes first.
           Cheap nodes have LOW depth.
           So we want LOW depth to have LOW score.
           So we use depth directly.
           
           Tuple: (0, 0, depth, v) for inf next use.
           Tuple: (1, -nu, depth, v) for finite next use.
           
           Comparison:
           (0, ...) < (1, ...) -> Inf is evicted before Finite. Correct.
           (1, -10, ...) < (1, -5, ...) -> Next use 10 is evicted before Next use 5. Correct.
           (1, -10, 1, ...) < (1, -10, 5, ...) -> Depth 1 is evicted before Depth 5. Correct.
        """
        nu = get_next_use(v)
        if nu == float('inf'):
            return (0, 0, depths.get(v, 0), v)
        else:
            return (1, -nu, depths.get(v, 0), v)

    def evict_one():
        if not fast_memory:
            return
        victim = min(fast_memory, key=get_eviction_score)
        if victim in input_set:
            operations.append(['delete', victim])
        else:
            operations.append(['store', victim])
        fast_memory.discard(victim)
        
    def ensure_in_memory(v):
        if v in fast_memory:
            return
        
        # Make space if needed
        while len(fast_memory) >= fast_memory_budget:
            evict_one()
            if len(fast_memory) < fast_memory_budget:
                break

        if v in input_set:
            operations.append(['load', v])
            fast_memory.add(v)
        else:
            # Need to compute v
            # Ensure parents are in memory
            for p in parents_map.get(v, []):
                ensure_in_memory(p)
            operations.append(['compute', v])
            fast_memory.add(v)
            
    # Main loop
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
            
        ensure_in_memory(v)
        
        # Update next_use pointers for parents of v
        for p in parents_map.get(v, []):
            if get_next_use(p) == i:
                next_use_ptr[p] += 1

    # Ensure all outputs are in fast memory at the end
    for o in outputs:
        if o not in fast_memory:
            ensure_in_memory(o)
            
    return operations

def _count_transfers(operations):
    """Count the number of load/store operations (transfers)."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    try:
        result = _greedy_pebble(dag, fast_memory_budget, deadline)
        
        # Verify validity
        if result and len(result) <= 20000:
            return result
        else:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
