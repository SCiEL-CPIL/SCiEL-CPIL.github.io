"""DAG pebbling with communication minimization."""
import time

def pebble(dag, fast_memory_budget):
    """
    Generate a pebbling schedule for a DAG to minimize communication (transfers).
    
    Args:
        dag: A dict with:
            - 'parents': list of lists, parents[v] = list of parent vertex IDs
            - 'inputs': list of input vertex IDs (start in fast memory)
            - 'outputs': list of output vertex IDs (must finish in fast memory)
        fast_memory_budget: int, max number of vertices in fast memory
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0

    # Parse DAG
    parents = dag['parents']
    n = len(parents)
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Build children map
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Topological order (assume vertices are already in topo order)
    topo = list(range(n))
    pos = [i for i in range(n)]
    
    # Compute last use position for each vertex
    last_use = [0] * n
    for v in range(n - 1, -1, -1):
        if children[v]:
            last_use[v] = max(pos[c] for c in children[v])
        else:
            last_use[v] = pos[v]
    
    # Determine needed vertices (ancestors of outputs)
    needed = set()
    def mark_needed(v):
        if v in needed:
            return
        needed.add(v)
        for p in parents[v]:
            mark_needed(p)
    for o in outputs:
        mark_needed(o)
    
    # State
    fast = set(inputs)  # Inputs start in fast memory
    slow = set()
    computed = set(inputs)
    ops = []
    
    # Process in topological order
    for v in topo:
        if v not in needed:
            continue
        if v in inputs:
            continue
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in fast:
                # Need to load p
                if len(fast) >= fast_memory_budget:
                    # Evict something
                    # Prefer evicting vertices that are not parents of v, not outputs, and have furthest last_use
                    candidates = [x for x in fast if x != p and x not in parents[v] and x not in outputs]
                    if candidates:
                        victim = max(candidates, key=lambda x: last_use[x])
                        fast.remove(victim)
                        slow.add(victim)
                        ops.append(['store', victim])
                    else:
                        # Evict any non-output, non-parent
                        candidates = [x for x in fast if x not in outputs]
                        if candidates:
                            victim = max(candidates, key=lambda x: last_use[x])
                            fast.remove(victim)
                            slow.add(victim)
                            ops.append(['store', victim])
                        else:
                            # All are outputs - can't evict, this is a problem
                            # Just proceed and hope it works (shouldn't happen in valid instances)
                            pass
                
                if p in slow:
                    slow.remove(p)
                fast.add(p)
                ops.append(['load', p])
        
        # Compute v
        ops.append(['compute', v])
        fast.add(v)
        computed.add(v)
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast:
            if o in slow:
                slow.remove(o)
            fast.add(o)
            ops.append(['load', o])
    
    if time.time() - start_time > DEADLINE:
        return ops
    
    return ops
