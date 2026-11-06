"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict, deque

def _topological_order(n, parents):
    """Compute topological order from parents list (0-indexed)."""
    indegree = [0] * n
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
            indegree[v] += 1
    q = deque([i for i in range(n) if indegree[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for w in children[u]:
            indegree[w] -= 1
            if indegree[w] == 0:
                q.append(w)
    return order

def _compute_depths(n, parents, topo):
    """Compute depth (longest path from source) for each vertex."""
    depth = [0] * n
    for v in topo:
        for p in parents[v]:
            if depth[p] + 1 > depth[v]:
                depth[v] = depth[p] + 1
    return depth

def _compute_remaining_depths(n, parents, children, topo):
    """Compute remaining depth (longest path to sink) for each vertex."""
    rem_depth = [0] * n
    # Process in reverse topological order
    for v in reversed(topo):
        for w in children[v]:
            if rem_depth[w] + 1 > rem_depth[v]:
                rem_depth[v] = rem_depth[w] + 1
    return rem_depth

def _is_output(dag, v):
    """Check if vertex v is an output vertex."""
    if isinstance(dag, dict):
        if 'outputs' in dag:
            return v in dag['outputs']
        if 'output_vertices' in dag:
            return v in dag['output_vertices']
    return False

def _get_parents(dag, v):
    """Get parents of vertex v from dag."""
    if isinstance(dag, dict):
        if 'parents' in dag:
            return dag['parents'][v]
        if 'edges' in dag:
            return dag['edges'][v]
    return []

def _get_vertices(dag):
    """Get list of vertices from dag."""
    if isinstance(dag, dict):
        if 'n' in dag:
            return list(range(dag['n']))
        if 'vertices' in dag:
            return dag['vertices']
        if 'parents' in dag:
            return list(range(len(dag['parents'])))
    return []

def pebble(dag, fast_memory_budget):
    """
    Pebble the DAG with minimal transfers.
    
    dag: dict with 'parents' (list of lists, 0-indexed) and 'outputs' (set/list of output vertices)
         or similar structure
    fast_memory_budget: int, number of fast memory slots
    returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill
    
    # Extract DAG structure
    if isinstance(dag, dict):
        if 'parents' in dag:
            parents = dag['parents']
        elif 'edges' in dag:
            parents = dag['edges']
        else:
            parents = []
        
        if 'outputs' in dag:
            outputs = set(dag['outputs'])
        elif 'output_vertices' in dag:
            outputs = set(dag['output_vertices'])
        else:
            outputs = set()
        
        if 'n' in dag:
            n = dag['n']
        elif 'vertices' in dag:
            n = len(dag['vertices'])
        elif parents:
            n = len(parents)
        else:
            n = 0
    else:
        # Fallback: assume list of parents
        parents = dag if isinstance(dag, list) else []
        n = len(parents)
        outputs = set()
    
    if n == 0:
        return []
    
    # Build children adjacency
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Topological order
    topo = _topological_order(n, parents)
    if len(topo) != n:
        # Cycle detected, return empty
        return []
    
    # Compute depths
    depth = _compute_depths(n, parents, topo)
    rem_depth = _compute_remaining_depths(n, parents, children, topo)
    
    # Identify input vertices (no parents)
    inputs = [v for v in range(n) if len(parents[v]) == 0]
    
    # Strategy: We need to keep all outputs blue at the end.
    # We use a priority-based scheduling approach.
    # Priority: higher remaining depth = more critical (on critical path)
    # We want to keep high-priority vertices in fast memory.
    
    # State:
    # - fast_mem: set of vertices currently in fast memory (blue)
    # - slow_mem: set of vertices stored in slow memory (blue but not in fast)
    # - computed: set of vertices that have been computed (either in fast or slow)
    
    # Operations:
    # load(v): move v from slow to fast (transfer)
    # store(v): move v from fast to slow (transfer)
    # compute(v): compute v in fast memory (requires all parents in fast)
    # delete(v): remove v from fast memory (no transfer, just free slot)
    
    # Initial state: all inputs are blue (in slow memory? or already in fast?)
    # "Inputs start blue" - let's assume they start in slow memory.
    # Actually, re-reading: "Inputs start blue, all outputs must finish blue"
    # "every red value consumes one fast-memory slot, including newly computed results"
    # So blue = in slow memory or already available, red = in fast memory.
    # Wait, "red value" might mean "not yet computed" or "in fast memory"?
    # Let me re-read: "every red value consumes one fast-memory slot"
    # This suggests red = in fast memory. Blue = in slow memory or computed and stored.
    # Actually, in pebbling, typically:
    # - A pebble on a vertex means it's in memory
    # - Here, fast memory has limited slots
    # - Slow memory is unlimited
    # - "blue" likely means in slow memory (available for load)
    # - "red" likely means in fast memory (consumes a slot)
    # - Inputs start blue = in slow memory
    # - Outputs must finish blue = in slow memory at end
    
    # Let's implement a simple but effective strategy:
    # 1. Process vertices in topological order
    # 2. For each vertex, ensure parents are in fast memory
    # 3. Compute the vertex
    # 4. If fast memory is full, evict a low-priority vertex
    # 5. At the end, ensure all outputs are in slow memory (blue)
    
    operations = []
    fast_mem = set()  # vertices in fast memory (red)
    slow_mem = set(inputs)  # vertices in slow memory (blue)
    
    # Priority function: vertices with higher remaining depth are more critical
    # Also, vertices that are ancestors of many outputs are more critical
    def priority(v):
        return rem_depth[v]
    
    # Process in topological order
    for v in topo:
        if time.time() > deadline:
            break
        
        # If v is an input, it's already in slow memory
        if v in inputs:
            # Load into fast memory if needed for computation
            continue
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in fast_mem:
                # Need to load p from slow memory
                # First, check if we have space
                if len(fast_mem) >= fast_memory_budget:
                    # Evict a low-priority vertex
                    # Find the vertex with lowest priority in fast_mem
                    min_prio = float('inf')
                    evict_v = None
                    for u in fast_mem:
                        if u not in outputs:  # Don't evict outputs if possible
                            if priority(u) < min_prio:
                                min_prio = priority(u)
                                evict_v = u
                    if evict_v is None:
                        # All are outputs, evict the one with lowest priority
                        for u in fast_mem:
                            if priority(u) < min_prio:
                                min_prio = priority(u)
                                evict_v = u
                    if evict_v is not None:
                        operations.append(['store', evict_v])
                        fast_mem.remove(evict_v)
                        slow_mem.add(evict_v)
                
                operations.append(['load', p])
                fast_mem.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_mem.add(v)
        slow_mem.discard(v)
        
        # Check if we can free some space
        # If fast_mem is over budget, evict
        while len(fast_mem) > fast_memory_budget:
            # Find the vertex with lowest priority that is not needed for upcoming computations
            min_prio = float('inf')
            evict_v = None
            for u in fast_mem:
                # Don't evict if u is an ancestor of any output not yet computed
                # For simplicity, just use priority
                if priority(u) < min_prio:
                    min_prio = priority(u)
                    evict_v = u
            if evict_v is not None:
                operations.append(['store', evict_v])
                fast_mem.remove(evict_v)
                slow_mem.add(evict_v)
            else:
                break
    
    # Ensure all outputs are in slow memory (blue)
    for v in outputs:
        if v in fast_mem:
            operations.append(['store', v])
            fast_mem.remove(v)
            slow_mem.add(v)
    
    return operations
# EVOLVE-BLOCK-END
