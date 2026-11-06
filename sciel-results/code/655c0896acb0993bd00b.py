"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 80  # seconds, leave safety margin before 90s kill
    
    # Parse the DAG structure - handle various possible formats
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        parents_map = {k: list(v) for k, v in dag.get('parents', {}).items()}
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        # Fallback: assume dag is a list of (vertex, parents) tuples
        vertices = []
        parents_map = {}
        inputs = set()
        outputs = set()
        for item in dag:
            if isinstance(item, tuple) and len(item) == 2:
                v, parents = item
                vertices.append(v)
                parents_map[v] = list(parents)
            elif isinstance(item, dict):
                v = item.get('vertex', item.get('id'))
                parents = item.get('parents', [])
                vertices.append(v)
                parents_map[v] = list(parents)
                if item.get('is_input'):
                    inputs.add(v)
                if item.get('is_output'):
                    outputs.add(v)
    
    # If no inputs/outputs specified, assume first/last in topo order
    if not inputs and vertices:
        inputs = {vertices[0]}
    if not outputs and vertices:
        outputs = {vertices[-1]}
    
    # Build children map
    children_map = defaultdict(list)
    for v, parents in parents_map.items():
        for p in parents:
            children_map[p].append(v)
    
    # Topological order (assume vertices are already in topological order)
    topo_order = list(vertices)
    
    # Count future uses for each vertex
    future_uses = defaultdict(int)
    for v in topo_order:
        for child in children_map[v]:
            future_uses[v] += 1
    for v in outputs:
        future_uses[v] += 10  # Boost for outputs
    
    ops = []
    fast_mem = set()
    
    def get_priority(v):
        return future_uses[v]
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in inputs:
            if v not in fast_mem:
                ops.append(['load', v])
                fast_mem.add(v)
            continue
        
        # Ensure parents are in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                ops.append(['load', p])
                fast_mem.add(p)
        
        # Compute
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # Evict if over budget
        while len(fast_mem) > fast_memory_budget:
            candidates = [(get_priority(u), u) for u in fast_mem]
            candidates.sort()
            if not candidates:
                break
            _, evict = candidates[0]
            if evict in outputs:
                ops.append(['store', evict])
            fast_mem.discard(evict)
    
    # Store remaining outputs
    for v in outputs:
        if v in fast_mem:
            ops.append(['store', v])
            fast_mem.discard(v)
    
    return ops[:20000]
# EVOLVE-BLOCK-END
