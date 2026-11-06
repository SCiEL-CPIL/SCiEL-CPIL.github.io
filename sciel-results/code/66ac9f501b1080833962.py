"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag structure assumed:
    - dag['vertices'] or dag['n']: number of vertices
    - dag['parents']: list of lists, parents[i] = list of parent vertex indices
    - dag['children']: list of lists, children[i] = list of child vertex indices  
    - dag['inputs']: list of input vertex indices
    - dag['outputs']: list of output vertex indices
    
    Returns list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Extract DAG structure
    if isinstance(dag, dict):
        n = dag.get('n', dag.get('num_vertices', 0))
        if n == 0:
            n = len(dag.get('parents', []))
        parents = dag.get('parents', [])
        children = dag.get('children', [])
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
    else:
        # Try to handle other formats
        n = len(dag)
        parents = []
        children = [[] for _ in range(n)]
        for i, p in enumerate(dag):
            parents.append(p)
            for parent in p:
                children[parent].append(i)
        inputs = []
        outputs = []
    
    if n == 0:
        return []
    
    # Build children if not provided
    if not children:
        children = [[] for _ in range(n)]
        for i, p in enumerate(parents):
            for parent in p:
                children[parent].append(i)
    
    # Identify inputs and outputs if not provided
    if not inputs:
        inputs = [i for i in range(n) if not parents[i]]
    if not outputs:
        outputs = [i for i in range(n) if not children[i]]
    
    # Count how many times each vertex is needed (fan-out)
    fan_out = [len(children[i]) for i in range(n)]
    
    # Compute "importance" of each vertex: how many descendants need it
    # Use memoized DFS to count unique descendants
    descendant_count = [0] * n
    visited = [False] * n
    stack = []
    
    # For each vertex, count how many distinct descendants it has
    # This helps determine how valuable it is to keep in fast memory
    def count_descendants(v):
        if visited[v]:
            return descendant_count[v]
        visited[v] = True
        total = 0
        seen = set()
        for child in children[v]:
            if child not in seen:
                seen.add(child)
                total += 1
                total += count_descendants(child)
        descendant_count[v] = total
        return total
    
    # Simpler: just use direct fan-out as proxy
    importance = fan_out[:]
    
    # Topological order (vertices should already be in topological order)
    # Verify and get order
    topo_order = list(range(n))
    
    # Track state: blue (in fast memory), red (in slow memory), or absent
    # blue_set: vertices currently in fast memory (blue)
    # We need to track which vertices have been computed and are available
    
    operations = []
    
    # Initialize: all inputs start blue (in fast memory)
    blue_set = set(inputs)
    computed = set(inputs)  # vertices that have been computed
    
    # For each input, we may need to load it
    for v in inputs:
        operations.append(['load', v])
    
    # Process vertices in topological order
    # For each vertex, if all parents are available (blue or can be loaded), compute it
    # Strategy: keep vertices with high importance in fast memory
    
    # Track which vertices are in fast memory
    in_fast = set(inputs)
    
    # For each vertex in topological order
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        # If v is an input, it's already handled
        if v in inputs:
            continue
        
        # Check if all parents are available
        # A parent is available if it's in fast memory (blue) or can be loaded from slow
        parents_available = True
        for p in parents[v]:
            if p not in in_fast and p not in computed:
                parents_available = False
                break
        
        if not parents_available:
            continue
        
        # Compute v
        # First, ensure all parents are in fast memory (load if needed)
        for p in parents[v]:
            if p not in in_fast:
                # Load parent from slow memory
                operations.append(['load', p])
                in_fast.add(p)
        
        # Perform computation
        operations.append(['compute', v])
        computed.add(v)
        
        # Decide whether to keep v in fast memory
        # If we have room, keep it. Otherwise, evict the least important vertex.
        if len(in_fast) < fast_memory_budget:
            in_fast.add(v)
        else:
            # Need to evict something before adding v
            # Find the vertex in in_fast with lowest importance that is not an output
            # and not a parent of any uncomputed vertex
            uncomputed = [u for u in topo_order if u not in computed and u != v]
            
            # Vertices that are parents of uncomputed vertices should be kept
            needed_parents = set()
            for u in uncomputed:
                for p in parents[u]:
                    if p in in_fast:
                        needed_parents.add(p)
            
            # Candidates for eviction: in in_fast, not in needed_parents, not an output
            evict_candidates = [x for x in in_fast 
                               if x not in needed_parents 
                               and x not in outputs]
            
            if evict_candidates:
                # Evict the one with lowest importance
                evict_v = min(evict_candidates, key=lambda x: importance[x])
                operations.append(['delete', evict_v])
                in_fast.remove(evict_v)
                in_fast.add(v)
            else:
                # Can't evict, so don't keep v in fast memory
                # v will be in slow memory (implicitly)
                pass
    
    # At the end, all outputs must be blue (in fast memory)
    for v in outputs:
        if v not in in_fast:
            operations.append(['load', v])
            in_fast.add(v)
    
    # Remove operations that exceed budget
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END

