"""Communication-minimizing DAG pebbling with caching and eviction."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85
    
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    n = len(vertices)
    B = fast_memory_budget
    vindex = {v: i for i, v in enumerate(vertices)}
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # For each vertex, compute the next position where it's needed as a parent
    # (for eviction decisions)
    next_parent_use = {}
    for i, v in enumerate(vertices):
        min_child_idx = float('inf')
        for c in children.get(v, []):
            if c in vindex and vindex[c] > i:
                min_child_idx = min(min_child_idx, vindex[c])
        next_parent_use[v] = min_child_idx
    
    fast_mem = set()
    ops = []
    
    def load(v):
        if v not in fast_mem:
            fast_mem.add(v)
            ops.append(['load', v])
    
    def store(v):
        ops.append(['store', v])
        if v in fast_mem:
            fast_mem.remove(v)
    
    def compute(v):
        ops.append(['compute', v])
        if v not in fast_mem:
            fast_mem.add(v)
    
    def delete(v):
        if v in fast_mem:
            fast_mem.remove(v)
            ops.append(['delete', v])
    
    def evict_one():
        if not fast_mem:
            return
        best = None
        best_use = -1
        for fm in fast_mem:
            nu = next_parent_use.get(fm, float('inf'))
            if nu > best_use:
                best_use = nu
                best = fm
        if best is not None:
            delete(best)
    
    for i, v in enumerate(vertices):
        if time.time() - start_time > DEADLINE:
            break
        
        if v not in inputs:
            # Need to compute v
            for p in parents.get(v, []):
                if p not in fast_mem:
                    if len(fast_mem) >= B:
                        evict_one()
                    load(p)
            compute(v)
        
        if v in outputs:
            store(v)
    
    # Ensure all outputs are stored
    stored = set(op[1] for op in ops if op[0] == 'store')
    for v in outputs:
        if v not in stored:
            if v not in fast_mem:
                if len(fast_mem) >= B:
                    evict_one()
                load(v)
            store(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops

# EVOLVE-BLOCK-END
