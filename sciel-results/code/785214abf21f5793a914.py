"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Parse DAG structure
    # dag is expected to have:
    # - vertices (list of vertex IDs)
    # - parents (dict or list mapping vertex -> list of parent vertices)
    # - outputs (list of output vertex IDs)
    # - inputs (list of input vertex IDs, start blue)
    
    # Try to extract structure from dag
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', dag.get('sinks', []))
        inputs = dag.get('inputs', dag.get('sources', []))
    elif isinstance(dag, (list, tuple)):
        # Assume dag is a list of vertices with some structure
        vertices = list(range(len(dag)))
        parents = {}
        outputs = []
        inputs = []
        # Try to infer from the structure
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                p = v.get('parents', v.get('deps', []))
                parents[i] = list(p) if p else []
            else:
                parents[i] = []
    else:
        # Fallback to baseline
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # If we couldn't parse properly, fallback
    if not vertices:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Build adjacency and in-degree info
    n = len(vertices)
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Count how many times each vertex is used (fan-out for eviction priority)
    use_count = defaultdict(int)
    for v in vertices:
        for c in children[v]:
            use_count[v] += 1
    
    # Topological order - assume vertices are already in topological order
    # or compute it
    in_degree = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] += 1
    
    topo_order = []
    queue = [v for v in vertices if in_degree[v] == 0]
    from collections import deque
    queue = deque(queue)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle or issue, fallback
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Identify input and output vertices
    input_set = set(inputs) if inputs else set(v for v in vertices if in_degree[v] == 0)
    output_set = set(outputs) if outputs else set(v for v in vertices if len(children[v]) == 0)
    
    # State: for each vertex, track if it's blue (in fast memory) or red (in slow memory)
    # Initially, inputs are blue, others are red (need to be computed)
    state = {}  # vertex -> 'blue' or 'red'
    for v in vertices:
        if v in input_set:
            state[v] = 'blue'
        else:
            state[v] = 'red'
    
    # Fast memory: set of vertices currently blue
    fast_mem = set(v for v in vertices if state[v] == 'blue')
    
    # Operations list
    operations = []
    
    # We process in topological order
    # For each vertex, we need its parents to be available
    # Strategy: keep high-use-count values in fast memory
    
    # Priority for keeping in fast memory: use_count (higher = keep longer)
    # When we need to evict, evict the one with lowest use_count among non-output vertices
    
    for v in topo_order:
        if time.time() > deadline:
            break
        
        # Check if all parents are blue
        parents_list = parents.get(v, [])
        all_parents_blue = all(state[p] == 'blue' for p in parents_list)
        
        if not all_parents_blue:
            # Need to load some parents
            for p in parents_list:
                if state[p] == 'red':
                    # Load parent from slow memory to fast
                    operations.append(['load', p])
                    state[p] = 'blue'
                    fast_mem.add(p)
            
            # After loading, check fast memory budget
            while len(fast_mem) > fast_memory_budget:
                # Need to evict something
                # Evict the vertex with lowest priority (lowest use_count, not output, not needed soon)
                candidates = [u for u in fast_mem if u not in output_set and u != v]
                if not candidates:
                    break
                # Pick the one with lowest use_count
                victim = min(candidates, key=lambda u: use_count[u])
                operations.append(['store', victim])
                state[victim] = 'red'
                fast_mem.discard(victim)
        
        # Compute v
        if state[v] == 'red':
            operations.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)
        
        # Check fast memory budget after compute
        while len(fast_mem) > fast_memory_budget:
            candidates = [u for u in fast_mem if u not in output_set]
            if not candidates:
                break
            victim = min(candidates, key=lambda u: use_count[u])
            operations.append(['store', victim])
            state[victim] = 'red'
            fast_mem.discard(victim)
    
    # Ensure all outputs are blue
    for v in output_set:
        if state[v] == 'red':
            operations.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
        
        # Check budget
        while len(fast_mem) > fast_memory_budget:
            candidates = [u for u in fast_mem if u not in output_set]
            if not candidates:
                break
            victim = min(candidates, key=lambda u: use_count[u])
            operations.append(['store', victim])
            state[victim] = 'red'
            fast_mem.discard(victim)
    
    # Limit operations to 20000
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
