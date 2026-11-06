<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag: dict with keys:
        - 'vertices': list of vertex indices
        - 'edges': list of (parent, child) tuples (topologically ordered)
        - 'inputs': set/list of input vertex indices
        - 'outputs': set/list of output vertex indices
        - 'num_vertices': int
    fast_memory_budget: int (number of red pebbles allowed)
    
    Returns: list of [operation, vertex] where operation is one of
             'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # Extract DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        num_vertices = dag.get('num_vertices', len(vertices) if vertices else 0)
    else:
        # Try to handle other formats
        vertices = list(dag)
        edges = []
        inputs = set()
        outputs = set()
        num_vertices = len(vertices)
    
    if not vertices and num_vertices == 0:
        return []
    
    # Build adjacency and reverse adjacency
    children = {v: [] for v in range(num_vertices)}
    parents = {v: [] for v in range(num_vertices)}
    for (p, c) in edges:
        if p in children and c in parents:
            children[p].append(c)
            parents[c].append(p)
    
    # Count in-degrees for topological understanding
    in_degree = {v: len(parents[v]) for v in range(num_vertices)}
    
    # Identify leaves (vertices with no children) and roots (vertices with no parents)
    roots = [v for v in range(num_vertices) if in_degree[v] == 0]
    leaves = [v for v in range(num_vertices) if len(children[v]) == 0]
    
    # Check if it's a layered DAG (common in these problems)
    # A layered DAG has clear levels where edges only go from level i to level i+1
    # Let's try to detect layers via longest path from roots
    layer = {}
    for v in roots:
        layer[v] = 0
    # Topological order traversal
    topo_order = []
    queue = deque(roots)
    in_deg_copy = dict(in_degree)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_deg_copy[c] -= 1
            if in_deg_copy[c] == 0:
                queue.append(c)
    
    if len(topo_order) != num_vertices:
        # Cycle or disconnected; handle gracefully
        topo_order = list(range(num_vertices))
    
    # Assign layers based on longest path
    for v in topo_order:
        if v not in layer:
            layer[v] = 0
        for c in children[v]:
            if c not in layer or layer[c] < layer[v] + 1:
                layer[c] = layer[v] + 1
    
    max_layer = max(layer.values()) if layer else 0
    
    # Strategy: For layered DAGs with limited fast memory, use a layer-by-layer approach
    # with recomputation when necessary.
    # 
    # The key insight: if fast_memory_budget >= max_layer + 1, we can process layer by layer
    # without recomputation. Otherwise, we need to recompute.
    #
    # For communication minimization, we want to minimize the number of load/store operations.
    # 'compute' operations are free (they don't transfer data).
    #
    # Let's implement a recursive pebbling strategy with memoization of which vertices
    # are currently in fast memory.
    
    # For now, implement a baseline that's better than the naive approach
    # by being smarter about when to store/delete.
    
    # Simple strategy: 
    # 1. Load all inputs
    # 2. Process in topological order, computing when possible
    # 3. Store outputs when computed
    # 4. Delete intermediate values when no longer needed
    
    operations = []
    in_fast_memory = set()  # vertices currently in fast memory (red pebbles)
    computed = set()  # vertices that have been computed
    
    # Load all inputs
    for v in sorted(inputs):
        operations.append(['load', v])
        in_fast_memory.add(v)
        computed.add(v)
    
    # Process in topological order
    for v in topo_order:
        if v in inputs:
            continue  # already loaded
        
        # Check if all parents are in fast memory
        all_parents_available = all(p in in_fast_memory for p in parents[v])
        
        if all_parents_available:
            # Compute v
            operations.append(['compute', v])
            in_fast_memory.add(v)
            computed.add(v)
            
            # If fast memory is full and v is not an output, consider deleting
            # some other non-output vertex
            if len(in_fast_memory) > fast_memory_budget:
                # Find a vertex to delete: prefer non-output, non-input vertices
                # that are no longer needed by any future vertex
                candidates = []
                for u in in_fast_memory:
                    if u not in outputs and u not in inputs:
                        # Check if u is still needed by any vertex not yet computed
                        still_needed = False
                        for c in children[u]:
                            if c not in computed:
                                still_needed = True
                                break
                        if not still_needed:
                            candidates.append(u)
                
                if candidates:
                    # Delete the one with the highest topological order (furthest from us)
                    to_delete = max(candidates, key=lambda x: topo_order.index(x))
                    operations.append(['delete', to_delete])
                    in_fast_memory.discard(to_delete)
                else:
                    # Must delete something; delete a non-output vertex
                    for u in in_fast_memory:
                        if u not in outputs:
                            operations.append(['delete', u])
                            in_fast_memory.discard(u)
                            break
        
        # If v is an output, store it
        if v in outputs:
            operations.append(['store', v])
            in_fast_memory.discard(v)
    
    # Check if all outputs are stored
    # If any output wasn't computed, we need to recompute
    # This is a simplification; a full solution would handle recomputation
    
    return operations
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Let me reconsider. The baseline is being called from a module, and I don't know the exact structure of `dag`. Let me look at what information I can extract and write a more robust solution.

