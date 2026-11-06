"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    The strategy is to use a recursive divide-and-conquer approach on the DAG structure.
    For each sub-DAG, we compute its outputs while keeping intermediate results in fast memory
    as much as possible. We use a greedy topological approach with careful memory management.
    
    The key insight is to minimize loads/stores by:
    1. Computing nodes in topological order
    2. Keeping results in fast memory until they're no longer needed
    3. Only storing to slow memory when fast memory is full
    4. Loading from slow memory only when a value is needed again
    
    For layered DAGs, we can exploit the structure to minimize communication.
    """
    
    # Parse the DAG structure
    # dag format: typically has 'vertices', 'edges' or similar structure
    # Let's handle common formats
    
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        # Try to get edges
        if 'edges' in dag:
            edges = dag['edges']
            # Build adjacency and in-degree
            n = len(vertices)
            # Map vertex names to indices
            v_to_idx = {v: i for i, v in enumerate(vertices)}
            adj = [[] for _ in range(n)]
            in_degree = [0] * n
            for u, v in edges:
                if u in v_to_idx and v in v_to_idx:
                    adj[v_to_idx[u]].append(v_to_idx[v])
                    in_degree[v_to_idx[v]] += 1
        elif 'parents' in dag:
            # parents[v] = list of parent vertices
            n = len(vertices)
            v_to_idx = {v: i for i, v in enumerate(vertices)}
            adj = [[] for _ in range(n)]
            in_degree = [0] * n
            for v in vertices:
                if v in v_to_idx:
                    vi = v_to_idx[v]
                    for p in dag['parents'].get(v, []):
                        if p in v_to_idx:
                            adj[v_to_idx[p]].append(vi)
                            in_degree[vi] += 1
        else:
            # Fallback: assume vertices are 0..n-1 and edges are given differently
            n = len(vertices)
            adj = [[] for _ in range(n)]
            in_degree = [0] * n
            # Try to infer from structure
            if 'children' in dag:
                for v in vertices:
                    if v in v_to_idx:
                        vi = v_to_idx[v]
                        for c in dag['children'].get(v, []):
                            if c in v_to_idx:
                                adj[vi].append(v_to_idx[c])
                                in_degree[v_to_idx[c]] += 1
    else:
        # Assume dag is a list of vertices
        vertices = list(dag)
        n = len(vertices)
        v_to_idx = {v: i for i, v in enumerate(vertices)}
        adj = [[] for _ in range(n)]
        in_degree = [0] * n
        # No edges, just compute in order
        # Assume sequential
        for i in range(n - 1):
            adj[i].append(i + 1)
            in_degree[i + 1] += 1
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Identify inputs (in_degree == 0) and outputs
    inputs = [i for i in range(n) if in_degree[i] == 0]
    
    # Find output vertices - typically the last ones or those with no children
    out_degree = [0] * n
    for i in range(n):
        for j in adj[i]:
            out_degree[i] += 1
    outputs = [i for i in range(n) if out_degree[i] == 0]
    
    # If no clear outputs, assume all vertices are outputs
    if not outputs:
        outputs = list(range(n))
    
    # Fast memory budget
    budget = fast_memory_budget if fast_memory_budget > 0 else 1
    
    # State: fast_memory is a dict mapping vertex_idx -> value (we'll track which are in fast mem)
    # We need to produce a list of operations
    
    operations = []
    
    # Track which vertices are currently in fast memory
    in_fast = set()
    # Track which vertices have been computed (their value is available somewhere)
    computed = set()
    # Track which vertices are stored in slow memory
    in_slow = set()
    
    def load(v):
        """Load vertex v from slow memory to fast memory."""
        operations.append(['load', v])
        in_fast.add(v)
        in_slow.discard(v)
    
    def store(v):
        """Store vertex v from fast memory to slow memory."""
        operations.append(['store', v])
        in_fast.discard(v)
        in_slow.add(v)
    
    def compute(v):
        """Compute vertex v. Parents must be in fast memory."""
        operations.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
    
    def delete(v):
        """Delete vertex v from fast memory."""
        operations.append(['delete', v])
        in_fast.discard(v)
    
    # First, load all inputs
    for i in inputs:
        if i not in in_fast:
            load(i)
            computed.add(i)
    
    # Now process in topological order
    # Use Kahn's algorithm
    from collections import deque
    
    in_deg = in_degree[:]
    queue = deque()
    for i in range(n):
        if in_deg[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't include all vertices, add remaining
    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)
    
    # Process each vertex in topological order
    for v in topo_order:
        if v in inputs:
            continue  # Already loaded
        
        # Check if all parents are in fast memory
        parents = [u for u in range(n) if v in adj[u]]
        
        # Load any parents that are not in fast memory but are computed
        for p in parents:
            if p not in in_fast:
                if p in in_slow:
                    # Need to load from slow memory
                    if len(in_fast) >= budget:
                        # Need to make room
                        # Find a vertex in fast memory that is not a parent of v and can be evicted
                        evicted = False
                        for f in list(in_fast):
                            # Can we evict f?
                            # f should not be needed for current computation
                            if f not in parents and f != v:
                                store(f)
                                evicted = True
                                break
                        if not evicted and len(in_fast) >= budget:
                            # Evict any vertex
                            for f in list(in_fast):
                                if f != v:
                                    store(f)
                                    break
                    load(p)
                elif p in computed:
                    # Shouldn't happen if we tracked correctly
                    load(p)
        
        # Compute v
        compute(v)
        
        # Check if we need to evict something
        if len(in_fast) > budget:
            # Find a vertex to evict
            # Prefer to keep vertices that will be needed soon
            # Simple heuristic: evict the one with the fewest remaining uses
            remaining_uses = {}
            for f in in_fast:
                count = 0
                # Count how many future vertices need f
                for future_v in topo_order:
                    if future_v == v:
                        continue
                    # Check if f is a parent of future_v
                    if future_v in adj[f]:
                        count += 1
                remaining_uses[f] = count
            
            # Evict the one with the lowest remaining uses
            candidates = [f for f in in_fast if f != v]
            if candidates:
                to_evict = min(candidates, key=lambda f: remaining_uses.get(f, 0))
                store(to_evict)
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v not in in_fast:
            if v in in_slow or v in computed:
                if len(in_fast) >= budget:
                    # Evict something
                    for f in list(in_fast):
                        if f not in outputs:
                            store(f)
                            break
                load(v)
    
    return operations
# EVOLVE-BLOCK-END
