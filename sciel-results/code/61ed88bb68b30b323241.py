"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG
    if isinstance(dag, dict):
        parents = dag.get('parents', dag.get('parent', []))
        outputs = dag.get('outputs', dag.get('output', []))
    else:
        parents = dag.parents
        outputs = dag.outputs
    
    n = len(parents)
    if n == 0:
        return []
    
    inputs = [i for i in range(n) if len(parents[i]) == 0]
    output_set = set(outputs)
    
    # Children map
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # last_needed[v] = the last vertex index that depends on v (as a parent)
    # If v is an output, last_needed[v] = n (needed until end)
    last_needed = [-1] * n
    for v in range(n):
        for c in children[v]:
            last_needed[v] = max(last_needed[v], c)
    for v in outputs:
        last_needed[v] = n
    
    # Simulate pebbling
    fast_mem = set(inputs)  # inputs start blue (in fast memory)
    computed = set(inputs)
    operations = []
    
    # Process vertices in topological order (0 to n-1)
    for v in range(n):
        if time.time() - start_time > DEADLINE:
            break
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in fast_mem:
                # Need to load p; evict if necessary
                while len(fast_mem) >= fast_memory_budget and p not in fast_mem:
                    # Find eviction candidate: not output, not v, not a parent of v, smallest last_needed
                    candidates = []
                    for u in fast_mem:
                        if u == v or u in output_set or u in parents[v]:
                            continue
                        candidates.append((last_needed[u], u))
                    if not candidates:
                        # Fallback: evict non-output, non-v vertex with smallest last_needed
                        candidates = [(last_needed[u], u) for u in fast_mem if u != v and u not in output_set]
                    if not candidates:
                        break
                    candidates.sort()
                    _, evict_v = candidates[0]
                    operations.append(['store', evict_v])
                    fast_mem.discard(evict_v)
                
                if p not in fast_mem:
                    operations.append(['load', p])
                    fast_mem.add(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        if v not in fast_mem:
            fast_mem.add(v)
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if v not in fast_mem:
            while len(fast_mem) >= fast_memory_budget and v not in fast_mem:
                candidates = [(last_needed[u], u) for u in fast_mem if u != v and u not in output_set]
                if not candidates:
                    candidates = [(last_needed[u], u) for u in fast_mem if u != v]
                if not candidates:
                    break
                candidates.sort()
                _, evict_v = candidates[0]
                operations.append(['store', evict_v])
                fast_mem.discard(evict_v)
            if v not in fast_mem:
                operations.append(['load', v])
                fast_mem.add(v)
    
    # Enforce operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
