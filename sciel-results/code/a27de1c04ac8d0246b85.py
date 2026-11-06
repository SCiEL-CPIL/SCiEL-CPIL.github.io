"""
Communication-minimizing DAG pebbling solver.

Strategy:
- Parse the DAG structure (topologically ordered parents, output vertices).
- Use a dynamic-programming / greedy heuristic to minimize total memory transfers.
- Key idea: process vertices in topological order, but carefully manage the fast memory
  (pebbles) to keep as many intermediate values as possible in fast memory, minimizing
  reloads from slow memory.
- Since inputs start blue (in fast memory) and outputs must finish blue, we need to ensure
  that at the end, all output vertices are pebbled.
- We use a greedy strategy: for each vertex in topological order, if it's not already
  pebbled, compute it (loading parents if needed), and then decide whether to keep it
  pebbled based on future usage and memory budget.

The baseline likely does a naive topological pass with minimal memory management.
We aim to do better by:
1. Computing a "future demand" for each vertex (how many times it will be needed later).
2. Using a priority-based eviction policy when memory is full.
3. Minimizing redundant computations by keeping high-demand vertices pebbled.
"""

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize communication (memory transfers).
    
    Parameters:
    - dag: A dictionary or object representing the DAG. Expected structure:
           - dag['vertices'] or dag['nodes']: list of vertex identifiers
           - dag['parents']: dict mapping vertex -> list of parent vertices
           - dag['outputs']: list of output vertex identifiers
           - dag['inputs']: list of input vertex identifiers (start blue)
    - fast_memory_budget: int, maximum number of vertices that can be pebbled at once.
    
    Returns:
    - list of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'
    """
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        parents = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        inputs = set(dag.get('inputs', []))
    else:
        # Try attribute-based access
        vertices = getattr(dag, 'vertices', getattr(dag, 'nodes', []))
        parents = getattr(dag, 'parents', {})
        outputs = set(getattr(dag, 'outputs', []))
        inputs = set(getattr(dag, 'inputs', []))
    
    # Build topological order
    if not vertices:
        return []
    
    # Compute topological order using Kahn's algorithm
    from collections import deque, defaultdict
    
    # Build adjacency list (child -> parents, so we need reverse: parent -> children)
    children = defaultdict(list)
    in_degree = defaultdict(int)
    
    for v in vertices:
        if v not in in_degree:
            in_degree[v] = 0
        for p in parents.get(v, []):
            children[p].append(v)
            in_degree[v] += 1
    
    # Topological sort
    queue = deque()
    for v in vertices:
        if in_degree[v] == 0:
            queue.append(v)
    
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topo_order doesn't include all vertices, there's a cycle; add remaining
    if len(topo_order) < len(vertices):
        remaining = [v for v in vertices if v not in set(topo_order)]
        topo_order.extend(remaining)
    
    # Compute future demand: for each vertex, how many times it's needed by descendants
    # that are outputs or have other uses
    demand = {}
    for v in vertices:
        demand[v] = 0
    
    # Count how many times each vertex is a parent of some vertex
    for v in vertices:
        for p in parents.get(v, []):
            demand[p] += 1
    
    # Also count outputs as having demand
    for v in outputs:
        demand[v] += 1
    
    # State: set of currently pebbled (red) vertices
    pebbled = set(inputs)  # Inputs start blue (pebbled)
    operations = []
    
    # Track which vertices have been computed (to avoid recomputation if possible)
    computed = set(inputs)
    
    # For each vertex in topological order, ensure it's pebbled when needed
    for v in topo_order:
        if v in pebbled:
            continue
        
        # Need to compute v: ensure all parents are pebbled
        parent_list = parents.get(v, [])
        
        # Load any parents that are not pebbled
        for p in parent_list:
            if p not in pebbled:
                # Need to make space if necessary
                if len(pebbled) >= fast_memory_budget:
                    # Evict a vertex with lowest future demand
                    # Don't evict inputs or outputs if possible
                    candidates = [x for x in pebbled if x not in outputs and x not in inputs]
                    if not candidates:
                        candidates = [x for x in pebbled if x not in outputs]
                    if not candidates:
                        candidates = list(pebbled)
                    
                    # Pick the one with lowest demand
                    min_demand = min(candidates, key=lambda x: demand[x])
                    operations.append(['delete', min_demand])
                    pebbled.discard(min_demand)
                
                # Load parent from slow memory
                operations.append(['load', p])
                pebbled.add(p)
        
        # Compute v
        operations.append(['compute', v])
        pebbled.add(v)
        computed.add(v)
        
        # If memory is full and v has low demand, we might evict it later
        # But for now, keep it pebbled
    
    # Ensure all outputs are pebbled at the end
    for v in outputs:
        if v not in pebbled:
            # Load it back
            if len(pebbled) >= fast_memory_budget:
                # Evict something not in outputs
                candidates = [x for x in pebbled if x not in outputs]
                if candidates:
                    min_demand = min(candidates, key=lambda x: demand[x])
                    operations.append(['delete', min_demand])
                    pebbled.discard(min_demand)
            
            operations.append(['load', v])
            pebbled.add(v)
    
    # Check operation limit
    if len(operations) > 20000:
        # Truncate if necessary (shouldn't happen for reasonable DAGs)
        operations = operations[:20000]
    
    return operations
