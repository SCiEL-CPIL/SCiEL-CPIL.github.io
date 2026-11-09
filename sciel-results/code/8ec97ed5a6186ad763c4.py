"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    parents = dag.get('parents', {})
    inputs = dag.get('inputs', [])
    outputs = dag.get('outputs', [])
    
    all_vertices = set(inputs) | set(outputs)
    for v, ps in parents.items():
        all_vertices.add(v)
        for p in ps:
            all_vertices.add(p)
    
    all_vertices = list(all_vertices)
    
    # Topological sort
    in_degree = defaultdict(int)
    children = defaultdict(list)
    for v in all_vertices:
        if v not in in_degree:
            in_degree[v] = 0
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
            in_degree[v] += 1
    
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for c in children[u]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(all_vertices):
        topo_order = list(all_vertices)
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Fan-out: how many times each vertex is used as a parent
    fan_out = defaultdict(int)
    for v, ps in parents.items():
        for p in ps:
            fan_out[p] += 1
    
    # For each vertex, determine if its value needs to persist beyond its immediate use
    # A vertex needs to be stored to slow memory if:
    # 1. It's an output (must finish blue)
    # 2. It has multiple children and we can't keep it in fast memory until all children are processed
    
    # Strategy: Process in topological order. Keep vertices in fast memory as long as possible.
    # When over budget, evict vertices that are no longer needed (all children processed).
    # For vertices that are still needed, store to slow memory and reload when needed.
    
    in_fast = set(inputs)
    computed = set(inputs)  # Inputs are considered available
    stored = set()  # Vertices stored to slow memory
    
    operations = []
    
    # Track which children of each vertex have been processed
    remaining_children = defaultdict(int)
    for v, ps in parents.items():
        for p in ps:
            remaining_children[p] += 1
    
    for i, v in enumerate(topo_order):
        if time.time() - start_time > DEADLINE:
            break
        
        if v in input_set:
            computed.add(v)
            continue
        
        # Ensure parents are in fast memory
        ps = parents.get(v, [])
        for p in ps:
            if p not in in_fast:
                operations.append(['load', p])
                in_fast.add(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        in_fast.add(v)
        
        # Decrement remaining children count for parents
        for p in ps:
            remaining_children[p] -= 1
        
        # Evict if over budget
        while len(in_fast) > fast_memory_budget:
            # Find vertex to evict
            best_evict = None
            best_score = float('inf')
            
            for u in in_fast:
                if u in output_set:
                    continue  # Don't evict outputs
                
                # Score: lower is better to evict
                # If no remaining children, evict first
                if remaining_children.get(u, 0) == 0:
                    score = -10000 + fan_out.get(u, 0)
                else:
                    score = 10000 - fan_out.get(u, 0)
                
                if score < best_score:
                    best_score = score
                    best_evict = u
            
            if best_evict is None:
                break
            
            if best_evict in computed and best_evict not in input_set:
                operations.append(['store', best_evict])
                stored.add(best_evict)
            
            in_fast.remove(best_evict)
    
    # Ensure outputs are in fast memory
    for o in outputs:
        if o not in in_fast:
            operations.append(['load', o])
            in_fast.add(o)
    
    return operations

# EVOLVE-BLOCK-END
