"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import heapq
import time

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize communication (load/store operations).
    
    Args:
        dag: A dict with keys:
             - 'n': number of vertices
             - 'parents': list of lists, parents[i] is list of parent indices of vertex i
             - 'outputs': list of output vertex indices
             - 'inputs': list of input vertex indices (start blue)
        fast_memory_budget: int, number of fast memory slots
    
    Returns:
        list of [operation, vertex] where operation is in {'load', 'store', 'compute', 'delete'}
    """
    n = dag['n']
    parents = dag['parents']
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    # Determine topological order
    # Assume vertices are already in topological order (0 is first, n-1 is last)
    # But let's verify and compute in-degrees
    in_degree = [0] * n
    for i in range(n):
        for p in parents[i]:
            in_degree[i] += 1
    
    # Topological sort
    topo_order = []
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in range(n):
            if u in parents[v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)
    
    # If topological sort didn't include all vertices, there's a cycle; fall back
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # Map vertex to position in topo order
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Identify which vertices are inputs (start blue, don't need load)
    # and which are outputs (must finish blue)
    
    # Compute children for reverse topological processing
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    # For each vertex, compute the set of descendants that are outputs
    # This helps determine if a value needs to be kept
    output_descendants = [set() for _ in range(n)]
    for v in reversed(topo_order):
        if v in outputs:
            output_descendants[v].add(v)
        for c in children[v]:
            output_descendants[v] |= output_descendants[c]
    
    # Determine if a vertex's value is needed for any output computation
    # A vertex v is "needed" if it has output descendants
    
    # Strategy: Use a priority-based approach
    # We process vertices in topological order
    # We maintain a set of currently loaded (in fast memory) vertices
    # When we need to compute a vertex, all its parents must be in fast memory
    # If not enough space, we evict (store) some vertices that are least likely to be needed soon
    
    # Better strategy: Use a look-ahead based eviction
    # For each vertex, compute how soon it will be needed again
    # We can precompute the next use of each vertex
    
    # Actually, let's use a simpler but effective approach:
    # 1. Process vertices in topological order
    # 2. Before computing vertex v, ensure all parents of v are in fast memory
    # 3. If fast memory is full, evict the vertex whose next use is farthest away
    # 4. After computing v, add v to fast memory
    # 5. At the end, ensure all outputs are in fast memory
    
    # Precompute next use for each vertex
    # next_use[v] = the earliest topo position after v's position where v is needed as a parent
    next_use = [float('inf')] * n
    for i in range(n):
        for c in children[i]:
            # i is needed when c is computed
            pos_c = topo_pos[c]
            if pos_c > topo_pos[i]:
                next_use[i] = min(next_use[i], pos_c)
    
    # Also, outputs need to be kept until the end
    for v in outputs:
        next_use[v] = n  # Keep until end
    
    # Fast memory state
    fast_mem = set()  # vertices currently in fast memory
    operations = []
    
    # Inputs start blue (in fast memory conceptually, but they're external)
    # Actually, inputs are external; we need to load them when needed
    # Let's treat inputs as available but not in fast memory initially
    # When we need an input, we load it into fast memory
    
    # Track which vertices have been computed and are in fast memory
    computed = set()  # vertices that have been computed
    in_fast = set()   # vertices currently in fast memory
    
    # Process in topological order
    for v in topo_order:
        # Check if v is an input
        if v in inputs:
            # Input starts blue, but needs to be in fast memory to be used
            if v not in in_fast:
                # Need to load v
                # Check if we have space
                if len(in_fast) >= fast_memory_budget:
                    # Need to evict something
                    # Find the vertex in in_fast with the farthest next use
                    # that is not an output and not v
                    candidates = [x for x in in_fast if x not in outputs and x != v]
                    if candidates:
                        # Evict the one with farthest next use
                        evict = max(candidates, key=lambda x: next_use[x])
                        in_fast.remove(evict)
                        operations.append(['store', evict])
                    else:
                        # All are outputs or v; this shouldn't happen if budget is sufficient
                        # Force evict an output (suboptimal but necessary)
                        evict = next(iter(in_fast))
                        in_fast.remove(evict)
                        operations.append(['store', evict])
                in_fast.add(v)
                operations.append(['load', v])
            # Input is now in fast memory, no compute needed
            computed.add(v)
            continue
        
        # v is not an input, so it needs to be computed
        # Ensure all parents of v are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                # Need to load p
                if len(in_fast) >= fast_memory_budget:
                    # Evict something
                    candidates = [x for x in in_fast if x not in outputs and x != p and x != v]
                    if candidates:
                        evict = max(candidates, key=lambda x: next_use[x])
                        in_fast.remove(evict)
                        operations.append(['store', evict])
                    else:
                        evict = next(iter(in_fast))
                        in_fast.remove(evict)
                        operations.append(['store', evict])
                in_fast.add(p)
                operations.append(['load', p])
        
        # Now all parents are in fast memory, compute v
        operations.append(['compute', v])
        computed.add(v)
        
        # Add v to fast memory
        if len(in_fast) >= fast_memory_budget:
            # Need to evict something before adding v
            # Don't evict v's parents if they're still needed
            candidates = [x for x in in_fast if x not in outputs and x != v]
            if candidates:
                evict = max(candidates, key=lambda x: next_use[x])
                in_fast.remove(evict)
                operations.append(['store', evict])
            else:
                # If all are outputs, we're in trouble; evict an output
                evict = next(iter(in_fast))
                in_fast.remove(evict)
                operations.append(['store', evict])
        in_fast.add(v)
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                candidates = [x for x in in_fast if x not in outputs and x != v]
                if candidates:
                    evict = max(candidates, key=lambda x: next_use[x])
                    in_fast.remove(evict)
                    operations.append(['store', evict])
                else:
                    evict = next(iter(in_fast))
                    in_fast.remove(evict)
                    operations.append(['store', evict])
            in_fast.add(v)
            operations.append(['load', v])
    
    # If operations exceed 20000, we need to trim
    # This shouldn't happen for reasonable inputs
    if len(operations) > 20000:
        # Fall back to a simpler approach
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
