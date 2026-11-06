"""DAG Pebbling with communication minimization."""
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize communication (transfers).
    
    Args:
        dag: A dict with keys:
            - 'vertices': list of vertex indices
            - 'parents': dict mapping vertex -> list of parent vertices
            - 'inputs': list of input vertex indices
            - 'outputs': list of output vertex indices
        fast_memory_budget: int, number of fast memory slots
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    # Parse DAG
    vertices = dag.get('vertices', list(range(dag.get('num_vertices', 0))))
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build children map and in-degrees
    children = defaultdict(list)
    in_degree = [0] * n
    for v in range(n):
        v_parents = parents.get(v, [])
        in_degree[v] = len(v_parents)
        for p in v_parents:
            children[p].append(v)
    
    # Topological order
    in_deg = in_degree[:]
    queue = deque([v for v in range(n) if in_deg[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    
    if len(topo) != n:
        # Cycle - shouldn't happen for DAG, but handle gracefully
        topo = list(range(n))
    
    # Count usage: how many times each vertex's value is needed
    # A vertex is "used" by its children and if it's an output
    usage_count = [0] * n
    for v in range(n):
        usage_count[v] = len(children[v])
        if v in outputs:
            usage_count[v] += 1  # output needs to be stored
    
    # Fast memory budget
    B = max(1, fast_memory_budget)
    
    # Strategy: Process in topological order.
    # Maintain a set of vertices currently in fast memory (red pebbles).
    # When computing a vertex, its parents must be available (in fast memory or loaded).
    # After computing, we may keep the result in fast memory or store it.
    # We can also recompute vertices later if needed.
    
    # Key insight: For each vertex, decide whether to:
    # 1. Compute once and keep in fast memory (if budget allows)
    # 2. Compute once and store (if it's an output or used many times)
    # 3. Recompute each time it's needed
    
    # Greedy approach: 
    # - Process in topological order
    # - Keep a set of "live" vertices in fast memory
    # - When a vertex is no longer needed (all its children computed and it's not an output), remove it
    # - If fast memory is full, evict the vertex with lowest future usage
    
    # Let's use a priority-based eviction strategy
    # For each vertex, compute "future_use": number of times it will be needed after current point
    
    # Simpler approach for layered DAGs:
    # Process layer by layer. Within each layer, compute all nodes.
    # Keep only the most useful nodes from the previous layer in fast memory.
    
    # Let's try a dynamic approach:
    # 1. First, determine a "keep" set: vertices worth keeping in fast memory
    # 2. Compute in topological order, loading/storing as needed
    
    # Value of keeping a vertex v in fast memory:
    # - Saves a load operation each time v is used as a parent
    # - Cost: occupies one fast memory slot
    
    # Let's compute the "transfer cost" if we don't keep v:
    # Each use of v as a parent requires a load (if v was stored) or recompute (if v was not stored)
    
    # For simplicity, let's use a heuristic:
    # - Always keep outputs in fast memory until the end (or store them immediately)
    # - For intermediate vertices, keep them if usage_count[v] > 1 and budget allows
    # - Process in topological order, evicting low-usage vertices when budget is exceeded
    
    operations = []
    in_fast_memory = set()  # vertices currently in fast memory (red)
    computed = set()  # vertices that have been computed at least once
    stored = set()  # vertices that have been stored to slow memory
    
    def is_deadline():
        return time.time() - start_time > DEADLINE
    
    # Process in topological order
    for v in topo:
        if is_deadline():
            break
        
        # Ensure all parents are available
        v_parents = parents.get(v, [])
        for p in v_parents:
            if p not in in_fast_memory and p not in stored:
                # Need to compute p first - but p should be before v in topo order
                # This shouldn't happen if topo order is correct
                pass
            elif p not in in_fast_memory and p in stored:
                # Load p from slow memory
                operations.append(['load', p])
                in_fast_memory.add(p)
            # If p is in fast memory, no action needed
        
        # Compute v
        if v not in computed:
            operations.append(['compute', v])
            computed.add(v)
            in_fast_memory.add(v)
        
        # Check if we need to evict to make room
        while len(in_fast_memory) > B:
            # Find vertex to evict: lowest future usage, not an output
            best_evict = None
            best_score = float('inf')
            for u in in_fast_memory:
                if u in outputs:
                    continue  # Don't evict outputs
                # Score: lower usage = better to evict
                score = usage_count[u]
                if score < best_score:
                    best_score = score
                    best_evict = u
            if best_evict is None:
                # All are outputs, just evict one
                best_evict = next(iter(in_fast_memory))
            in_fast_memory.discard(best_evict)
            operations.append(['delete', best_evict])
        
        # If v is an output, store it
        if v in outputs:
            operations.append(['store', v])
            stored.add(v)
            # Remove from fast memory after storing
            in_fast_memory.discard(v)
    
    # Verify all outputs are stored
    for v in outputs:
        if v not in stored:
            # Need to ensure output is stored
            if v in in_fast_memory:
                operations.append(['store', v])
                in_fast_memory.discard(v)
            else:
                # Recompute and store
                # This is a fallback - shouldn't happen with correct topo processing
                pass
    
    # Truncate if too many operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
