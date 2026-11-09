"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (transfers) for DAG pebbling.
    """
    start_time = time.time()
    DEADLINE = start_time + 80
    
    # Parse DAG structure
    if hasattr(dag, 'parents'):
        parents = dict(dag.parents)
    elif isinstance(dag, dict) and 'parents' in dag:
        parents = dict(dag['parents'])
    else:
        parents = {}
    
    if hasattr(dag, 'outputs'):
        outputs = list(dag.outputs)
    elif isinstance(dag, dict) and 'outputs' in dag:
        outputs = list(dag['outputs'])
    else:
        outputs = []
    
    if hasattr(dag, 'inputs'):
        inputs = list(dag.inputs)
    elif isinstance(dag, dict) and 'inputs' in dag:
        inputs = list(dag['inputs'])
    else:
        all_v = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_v.add(p)
        inputs = [v for v in all_v if v not in parents or not parents.get(v)]
    
    all_vertices = set()
    for v in parents:
        all_vertices.add(v)
        for p in parents[v]:
            all_vertices.add(p)
    for v in outputs:
        all_vertices.add(v)
    for v in inputs:
        all_vertices.add(v)
    
    if not all_vertices:
        return []
    
    # Build children map
    children = {v: [] for v in all_vertices}
    for v in parents:
        for p in parents[v]:
            if p in children:
                children[p].append(v)
    
    # Topological order
    in_degree = {}
    for v in all_vertices:
        in_degree[v] = len(parents.get(v, []))
    
    topo_order = []
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(all_vertices):
        remaining = [v for v in all_vertices if v not in set(topo_order)]
        topo_order.extend(remaining)
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    budget = fast_memory_budget if fast_memory_budget is not None else len(all_vertices)
    if budget < 1:
        budget = 1
    
    # Count remaining usage for each vertex
    remaining_usage = {}
    for v in all_vertices:
        remaining_usage[v] = len(children.get(v, []))
    
    # Fast memory state
    fast_mem = {}
    operations = []
    
    # Inputs start blue
    for v in inputs:
        if v in all_vertices:
            fast_mem[v] = 'blue'
    
    def evict_one():
        """Evict a vertex from fast memory, preferring non-outputs with low remaining usage."""
        if not fast_mem:
            return
        # Prefer non-outputs, then lowest remaining usage
        candidates = [ev for ev in fast_mem if ev not in output_set]
        if not candidates:
            candidates = list(fast_mem.keys())
        
        best = min(candidates, key=lambda ev: (remaining_usage.get(ev, 0), ev))
        operations.append(['store', best])
        del fast_mem[best]
    
    # Process in topological order
    for v in topo_order:
        if time.time() > DEADLINE:
            break
        
        if v in input_set:
            continue
        
        # Ensure parents are in fast memory
        v_parents = parents.get(v, [])
        for p in v_parents:
            if p not in fast_mem:
                if len(fast_mem) >= budget:
                    evict_one()
                operations.append(['load', p])
                fast_mem[p] = 'red'
        
        # Compute
        operations.append(['compute', v])
        
        # Add to fast memory
        if len(fast_mem) >= budget:
            evict_one()
        fast_mem[v] = 'red'
        
        # Update remaining usage for parents
        for p in v_parents:
            if p in remaining_usage:
                remaining_usage[p] -= 1
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in fast_mem:
            if len(fast_mem) >= budget:
                evict_one()
            operations.append(['load', v])
            fast_mem[v] = 'blue'
        else:
            fast_mem[v] = 'blue'
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
