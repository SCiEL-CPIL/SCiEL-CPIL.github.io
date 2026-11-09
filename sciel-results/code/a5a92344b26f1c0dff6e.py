"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve the communication-minimizing DAG pebbling problem.
    
    dag: dict with keys like 'parents' (list of lists, topologically ordered),
         'outputs' (list of vertex indices), and possibly 'n' (number of vertices)
    fast_memory_budget: int, max number of red values in fast memory
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 
             'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse DAG structure
    if isinstance(dag, dict):
        parents = dag.get('parents', [])
        outputs = dag.get('outputs', [])
        n = dag.get('n', len(parents))
    elif isinstance(dag, (list, tuple)):
        # Maybe it's a list of parent lists
        parents = list(dag)
        outputs = []
        n = len(parents)
    else:
        # Fallback: try to extract
        parents = getattr(dag, 'parents', [])
        outputs = getattr(dag, 'outputs', [])
        n = getattr(dag, 'n', len(parents))
    
    if not parents:
        return []
    
    n = len(parents)
    
    # Determine input vertices (vertices with no parents)
    input_vertices = [i for i in range(n) if len(parents[i]) == 0]
    
    # Determine output vertices
    if not outputs:
        # Outputs are typically the last layer or vertices with no children
        # Let's assume outputs are specified or we need to compute all
        outputs = list(range(n))
    
    # Build children mapping
    children = defaultdict(list)
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Topological order is already given (parents before children)
    
    # Strategy: Process in topological order, keeping values in fast memory
    # when beneficial. Use a greedy approach with lookahead.
    
    # First, let's try a simple but effective strategy:
    # - Load inputs as needed
    # - Compute vertices in topological order
    # - Keep computed values in fast memory if budget allows
    # - Store to slow memory when fast memory is full and value is no longer needed soon
    # - Load back when needed again
    
    # Track state
    in_fast = set()  # vertices currently in fast memory (red)
    in_slow = set()  # vertices currently in slow memory
    operations = []
    
    # Count how many times each vertex is needed (as a parent)
    use_count = defaultdict(int)
    for v in range(n):
        for p in parents[v]:
            use_count[p] += 1
    
    # Process in topological order
    # We'll use a priority: process vertices whose parents are all available
    
    # Initialize: inputs are available in slow memory
    for v in input_vertices:
        in_slow.add(v)
    
    # We need to compute all output vertices
    # Use a worklist approach
    
    # For efficiency, let's use a simple strategy:
    # 1. For each vertex in topo order, ensure its parents are in fast memory
    # 2. Compute the vertex
    # 3. Decide whether to keep it in fast or store it
    
    # Better: use a scheduling that minimizes transfers
    # For layered DAGs, process layer by layer
    
    # Let's implement a greedy scheduler:
    
    def get_layer(v):
        """Get the layer of vertex v (longest path from input)."""
        if v in input_vertices:
            return 0
        return max(get_layer(p) for p in parents[v]) + 1
    
    # Memoize layer computation
    layer_cache = {}
    def get_layer_memo(v):
        if v in layer_cache:
            return layer_cache[v]
        if v in input_vertices:
            layer_cache[v] = 0
            return 0
        layer_cache[v] = max(get_layer_memo(p) for p in parents[v]) + 1
        return layer_cache[v]
    
    # Group vertices by layer
    from collections import defaultdict
    layers = defaultdict(list)
    for v in range(n):
        layers[get_layer_memo(v)].append(v)
    
    max_layer = max(layers.keys()) if layers else 0
    
    # Process layer by layer
    # For each layer, we need to have all parents in fast memory to compute
    # Strategy: keep a window of layers in fast memory
    
    # Simple effective strategy:
    # - Maintain fast memory with values needed for next computations
    # - When fast memory is full, evict the value that will be needed least soon
    
    # Let's implement a simpler but effective approach:
    # Process vertices in topological order, maintaining fast memory optimally
    
    # Reset state
    in_fast = set()
    in_slow = set(input_vertices)  # inputs start in slow memory
    computed = set()  # vertices that have been computed at least once
    operations = []
    
    # We'll process in topological order
    # For each vertex, ensure parents are in fast, compute, then manage memory
    
    # To be efficient, let's precompute the order of first use for each vertex
    first_use = {}
    for v in range(n):
        for p in parents[v]:
            if p not in first_use:
                first_use[p] = v
    
    # Process in topological order (which is already the order of vertices 0..n-1)
    for v in range(n):
        if time.time() - start_time > DEADLINE:
            # Return what we have so far (should be valid if we've been careful)
            break
        
        # Check if this vertex needs to be computed (it's an output or has children)
        has_children = len(children[v]) > 0
        is_output = v in outputs
        
        # We need to compute v if it's an output or has children that need it
        # Actually, we need to compute all vertices that are ancestors of outputs
        
        # For simplicity, compute all vertices in topo order
        # Ensure parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                # Need to load p into fast memory
                if p in in_slow:
                    # Load from slow to fast
                    if len(in_fast) >= fast_memory_budget:
                        # Need to evict something
                        # Find a vertex in fast that won't be needed soon
                        # Simple heuristic: evict the one with highest index (least likely needed)
                        # Better: evict the one that is not a parent of any uncomputed vertex
                        to_evict = None
                        for ev in list(in_fast):
                            # Check if ev is needed by any uncomputed vertex
                            needed = False
                            for c in children[ev]:
                                if c >= v and c not in computed:
                                    needed = True
                                    break
                            if not needed:
                                to_evict = ev
                                break
                        if to_evict is None:
                            # Evict the one with highest index
                            to_evict = max(in_fast)
                        # Store to_evict to slow memory
                        operations.append(['store', to_evict])
                        in_fast.remove(to_evict)
                        in_slow.add(to_evict)
                    
                    operations.append(['load', p])
                    in_fast.add(p)
                    in_slow.discard(p)
                else:
                    # p was never computed or was deleted
                    # This shouldn't happen if we process in topo order
                    pass
        
        # Compute v
        operations.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
        in_slow.discard(v)  # v is now in fast, not in slow
        
        # Decide whether to keep v in fast or store it
        # Keep it if it will be needed soon and we have budget
        # For now, keep all in fast until budget is exceeded
        # We'll handle eviction when we need to load something
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v not in in_fast:
            if v in in_slow:
                if len(in_fast) >= fast_memory_budget:
                    to_evict = max(in_fast)
                    operations.append(['store', to_evict])
                    in_fast.remove(to_evict)
                    in_slow.add(to_evict)
                operations.append(['load', v])
                in_fast.add(v)
                in_slow.discard(v)
    
    return operations

# EVOLVE-BLOCK-END
