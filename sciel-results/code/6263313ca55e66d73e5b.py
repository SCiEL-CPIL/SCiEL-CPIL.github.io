"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    num_vertices = dag['num_vertices']
    parents = dag['parents']
    outputs = dag['outputs']
    
    children = [[] for _ in range(num_vertices)]
    for v in range(num_vertices):
        for p in parents[v]:
            children[p].append(v)
    
    out_degree = [len(children[v]) for v in range(num_vertices)]
    is_input = [len(parents[v]) == 0 for v in range(num_vertices)]
    is_output = [False] * num_vertices
    for o in outputs:
        is_output[o] = True
    
    # Compute persistence cost: out_degree * (1 + dist_to_sink)
    dist_to_sink = [0] * num_vertices
    for v in range(num_vertices - 1, -1, -1):
        if is_output[v]:
            dist_to_sink[v] = 0
        elif children[v]:
            dist_to_sink[v] = 1 + max(dist_to_sink[c] for c in children[v])
        else:
            dist_to_sink[v] = 0
    
    def persistence_cost(v):
        return out_degree[v] * (dist_to_sink[v] + 1)
    
    ops = []
    fast_mem = set()
    slow_blue = set()
    remaining_use = list(out_degree)
    processed = [False] * num_vertices
    
    for v in range(num_vertices):
        if time.time() - start_time > DEADLINE:
            break
        
        if is_input[v]:
            slow_blue.add(v)
            processed[v] = True
            continue
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in fast_mem:
                if p in slow_blue:
                    ops.append(['load', p])
                    fast_mem.add(p)
                else:
                    # Should not happen if we never delete without storing
                    ops.append(['load', p])
                    fast_mem.add(p)
        
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        processed[v] = True
        
        # Update remaining_use for parents
        for p in parents[v]:
            remaining_use[p] -= 1
        
        # If v is output, store it
        if is_output[v]:
            ops.append(['store', v])
            slow_blue.add(v)
            fast_mem.discard(v)
            remaining_use[v] = 0
            continue
        
        # If no longer needed, we can delete (free)
        if remaining_use[v] == 0:
            ops.append(['delete', v])
            fast_mem.discard(v)
            continue
        
        # Check memory budget
        while len(fast_mem) > fast_memory_budget:
            # Find vertex to evict: lowest persistence cost, not v, not an output
            candidates = []
            for u in fast_mem:
                if u == v:
                    continue
                if is_output[u]:
                    continue
                candidates.append((persistence_cost(u), u))
            if candidates:
                candidates.sort()
                evict_u = candidates[0][1]
                # Store it so we can reload later
                ops.append(['store', evict_u])
                slow_blue.add(evict_u)
                fast_mem.discard(evict_u)
            else:
                break
    
    # Ensure all outputs are in slow_blue
    for o in outputs:
        if o not in slow_blue:
            if o in fast_mem:
                ops.append(['store', o])
                slow_blue.add(o)
                fast_mem.discard(o)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops

# EVOLVE-BLOCK-END
