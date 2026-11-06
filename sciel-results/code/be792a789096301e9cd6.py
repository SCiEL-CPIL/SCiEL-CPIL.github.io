"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    parents = dag.get('parents', [])
    outputs = set(dag.get('outputs', []))
    n = len(parents)
    
    if n == 0:
        return []
    
    # Compute layers
    layer = [0] * n
    for v in range(n):
        if parents[v]:
            layer[v] = max(layer[p] for p in parents[v]) + 1
    
    max_layer = max(layer)
    layers = [[] for _ in range(max_layer + 1)]
    for v in range(n):
        layers[layer[v]].append(v)
    
    # Children and remaining counts
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    remaining_children = [len(children[v]) for v in range(n)]
    
    fast_mem = set()
    ops = []
    
    for L in range(max_layer + 1):
        if time.time() - start_time > DEADLINE:
            break
        if not layers[L]:
            continue
        
        # Load needed parents
        for v in layers[L]:
            for p in parents[v]:
                if p not in fast_mem:
                    ops.append(["load", p])
                    fast_mem.add(p)
        
        # Compute
        for v in layers[L]:
            ops.append(["compute", v])
            fast_mem.add(v)
        
        # Evict if over budget: prioritize keeping outputs and high-remaining-children
        while len(fast_mem) > fast_memory_budget:
            # Score each vertex: lower score = evict first
            scored = []
            for v in fast_mem:
                if v in outputs:
                    score = 1000000 + remaining_children[v]  # Keep outputs
                else:
                    score = remaining_children[v]
                scored.append((score, v))
            scored.sort(key=lambda x: x[0])
            _, v_evict = scored[0]
            ops.append(["store", v_evict])
            fast_mem.remove(v_evict)
        
        # Update remaining_children for parents
        for v in layers[L]:
            for p in parents[v]:
                remaining_children[p] -= 1
    
    # Outputs must finish blue (in slow memory)
    for v in outputs:
        if v in fast_mem:
            ops.append(["store", v])
            fast_mem.remove(v)
    
    return ops

# EVOLVE-BLOCK-END
