"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    if not vertices:
        return []
    
    n = len(vertices)
    vertex_index = {v: i for i, v in enumerate(vertices)}
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    child_indices = {}
    for v in vertices:
        child_indices[v] = sorted([vertex_index[c] for c in children[v]])
    
    fast_memory = set()
    operations = []
    computed = set()
    next_ptr = defaultdict(int)
    
    for v in vertices:
        if time.time() - start_time > DEADLINE:
            break
        
        if v not in inputs:
            for p in parents[v]:
                if p not in fast_memory:
                    operations.append(['load', p])
                    fast_memory.add(p)
            
            operations.append(['compute', v])
            computed.add(v)
            fast_memory.add(v)
        
        while len(fast_memory) > fast_memory_budget:
            best_evict = None
            best_next_use = -1
            
            for u in fast_memory:
                ptr = next_ptr[u]
                ci = child_indices[u]
                cur_idx = vertex_index[v]
                
                while ptr < len(ci) and ci[ptr] < cur_idx:
                    ptr += 1
                next_ptr[u] = ptr
                
                if ptr < len(ci):
                    next_use = ci[ptr]
                else:
                    next_use = -1
                
                if u in outputs:
                    next_use = max(next_use, n)
                
                if next_use > best_next_use:
                    best_next_use = next_use
                    best_evict = u
            
            if best_evict is not None:
                if best_evict in outputs:
                    operations.append(['store', best_evict])
                else:
                    operations.append(['delete', best_evict])
                fast_memory.discard(best_evict)
    
    for v in outputs:
        if v in fast_memory:
            operations.append(['store', v])
            fast_memory.discard(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
