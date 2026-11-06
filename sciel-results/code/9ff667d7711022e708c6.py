"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Generate a pebbling schedule for a DAG to minimize communication.
    
    dag structure (assumed):
      - dag['n']: number of vertices (0..n-1)
      - dag['parents']: list where parents[i] is list of parent vertices of i
      - dag['inputs']: list of input vertex indices (start blue/slow)
      - dag['outputs']: list of output vertex indices (must end blue/slow)
    
    Returns list of [operation, vertex] where operation is 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    n = dag['n']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    # Build children adjacency
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    # Topological order (Kahn's algorithm)
    in_deg = [0] * n
    for i in range(n):
        for p in parents[i]:
            in_deg[i] += 1
    
    topo_order = []
    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    
    if len(topo_order) != n:
        topo_order = list(range(n))
    
    operations = []
    in_fast = set()
    computed = set()
    
    def make_room():
        """Make room in fast memory by deleting a vertex."""
        nonlocal in_fast, operations
        if fast_memory_budget == 0:
            return
        if len(in_fast) < fast_memory_budget:
            return
        # Find a vertex to delete: prefer non-output whose children are all computed
        best = None
        best_priority = 2
        for v in in_fast:
            if v in outputs:
                continue
            all_children_computed = all(c in computed for c in children[v])
            if all_children_computed:
                priority = 0
            else:
                priority = 1
            if priority < best_priority:
                best_priority = priority
                best = v
        if best is not None:
            operations.append(['delete', best])
            in_fast.remove(best)
    
    # Load all inputs
    for v in sorted(inputs):
        if len(in_fast) >= fast_memory_budget and fast_memory_budget > 0:
            make_room()
        if v not in in_fast:
            operations.append(['load', v])
            in_fast.add(v)
            computed.add(v)
    
    # Process in topological order
    for v in topo_order:
        if v in inputs:
            continue
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                if len(in_fast) >= fast_memory_budget and fast_memory_budget > 0:
                    make_room()
                if p not in in_fast:
                    operations.append(['load', p])
                    in_fast.add(p)
        
        # Compute v
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget and fast_memory_budget > 0:
                make_room()
        operations.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
    
    # Store all outputs
    for v in sorted(outputs):
        if v in in_fast:
            operations.append(['store', v])
            in_fast.remove(v)
        else:
            # Should not happen, but handle gracefully
            if len(in_fast) >= fast_memory_budget and fast_memory_budget > 0:
                make_room()
            if v not in in_fast:
                operations.append(['load', v])
                in_fast.add(v)
            operations.append(['store', v])
            in_fast.remove(v)
    
    return operations
# EVOLVE-BLOCK-END
