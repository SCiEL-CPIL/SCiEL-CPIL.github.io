"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling to minimize transfers (load + store operations).
    
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    start_time = time.time()
    
    # Parse DAG structure
    vertices = list(dag.get('vertices', range(dag.get('num_vertices', 0))))
    parents_dict = dag.get('parents', {})
    if isinstance(parents_dict, dict):
        parents_dict = {int(k): list(v) for k, v in parents_dict.items()}
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents_dict.get(v, []):
            children[p].append(v)
    
    # Topological order (assume vertices are already in topo order)
    topo_order = list(vertices)
    
    # Compute naive schedule for baseline
    naive_ops = []
    for v in topo_order:
        if v in inputs:
            naive_ops.append(['load', v])
        else:
            naive_ops.append(['compute', v])
        if v in outputs:
            naive_ops.append(['store', v])
    
    # Greedy pebbling strategy:
    # Keep values in fast memory if they will be reused
    # Recompute when memory is tight
    
    fast_mem = set()
    ops = []
    computed = set()
    
    # Helper: evict a vertex from fast memory
    def evict():
        nonlocal fast_mem
        if not fast_mem:
            return
        # Find vertex with fewest future uses
        best_victim = None
        min_uses = float('inf')
        for v in fast_mem:
            future_uses = len([c for c in children.get(v, []) if c not in computed])
            if future_uses < min_uses:
                min_uses = future_uses
                best_victim = v
        if best_victim is not None:
            ops.append(['delete', best_victim])
            fast_mem.discard(best_victim)
    
    # Helper: ensure a vertex is in fast memory
    def ensure_in_fast(v):
        nonlocal fast_mem, computed
        if v in fast_mem:
            return
        if len(fast_mem) >= fast_memory_budget:
            evict()
        if v in inputs:
            ops.append(['load', v])
        else:
            # Recompute
            for p in parents_dict.get(v, []):
                ensure_in_fast(p)
            ops.append(['compute', v])
            computed.add(v)
        fast_mem.add(v)
    
    # Process vertices in topological order
    for v in topo_order:
        if v in inputs:
            # Load input
            ensure_in_fast(v)
            computed.add(v)
        else:
            # Ensure parents are in fast memory
            for p in parents_dict.get(v, []):
                ensure_in_fast(p)
            # Compute v
            ops.append(['compute', v])
            fast_mem.add(v)
            computed.add(v)
        
        # Store output
        if v in outputs:
            ops.append(['store', v])
        
        # Optionally delete v if no future uses
        future_uses = len([c for c in children.get(v, []) if c not in computed])
        if future_uses == 0 and v not in outputs and len(fast_mem) > fast_memory_budget // 2:
            ops.append(['delete', v])
            fast_mem.discard(v)
    
    # Ensure all outputs are stored
    stored = set(op[1] for op in ops if op[0] == 'store')
    for out in outputs:
        if out not in stored:
            ensure_in_fast(out)
            ops.append(['store', out])
    
    # Check operation limit
    if len(ops) > 20000:
        return naive_ops
    
    return ops

# EVOLVE-BLOCK-END
