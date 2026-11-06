"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling with communication minimization.
    
    dag: dict or object with structure describing the DAG
         Expected keys/attributes:
         - 'vertices' or 'n': number of vertices
         - 'parents': list of lists, parents[v] = list of parent vertex indices
         - 'outputs': list of output vertex indices
         - 'inputs': list of input vertex indices (or vertices with no parents)
    
    fast_memory_budget: int, number of fast memory slots
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    # Parse DAG structure
    # Try to handle different formats
    if hasattr(dag, 'vertices'):
        n = dag.vertices
        parents = dag.parents
        outputs = dag.outputs
        inputs = getattr(dag, 'inputs', None)
    elif isinstance(dag, dict):
        n = dag.get('vertices', dag.get('n', len(dag.get('parents', []))))
        parents = dag.get('parents', [])
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', None)
    else:
        # Fallback: assume dag is a tuple or list
        if isinstance(dag, (tuple, list)):
            n = dag[0]
            parents = dag[1]
            outputs = dag[2]
            inputs = dag[3] if len(dag) > 3 else None
        else:
            raise ValueError("Unknown dag format")
    
    # Determine inputs if not specified
    if inputs is None:
        inputs = [v for v in range(n) if len(parents[v]) == 0]
    
    # Topological order (assumed to be 0..n-1 if parents are topologically ordered)
    # The problem states "topologically ordered parents", so vertex i's parents are all < i
    topo_order = list(range(n))
    
    # Compute reference count: how many times each vertex is used as a parent
    ref_count = [0] * n
    for v in range(n):
        for p in parents[v]:
            ref_count[p] += 1
    
    # Greedy pebbling:
    # Process vertices in topological order
    # Keep values in fast memory if possible
    # Track which vertices are currently in fast memory (red)
    
    fast_memory = set()  # vertices currently in fast memory
    stored = set(inputs)  # vertices already stored in slow memory (blue)
    computed = set()  # vertices already computed
    
    operations = []
    
    def elapsed():
        return time.time() - start_time
    
    for v in topo_order:
        if elapsed() > DEADLINE:
            # Return what we have so far (may be incomplete, but better than timeout)
            break
        
        # Check if v is already in fast memory
        if v in fast_memory:
            continue
        
        # Check if v is an input
        if v in stored:
            # Need to load it if not in fast memory
            # Check if we have space
            if len(fast_memory) < fast_memory_budget:
                operations.append(['load', v])
                fast_memory.add(v)
            else:
                # Need to evict something
                # Evict the vertex with lowest future reference count
                # that is not an output and has been used
                evict_candidates = []
                for u in fast_memory:
                    if u in outputs:
                        continue
                    if ref_count[u] == 0:
                        evict_candidates.append((0, u))
                    else:
                        evict_candidates.append((ref_count[u], u))
                if evict_candidates:
                    evict_candidates.sort()
                    evict_v = evict_candidates[0][1]
                    operations.append(['delete', evict_v])
                    fast_memory.discard(evict_v)
                    operations.append(['load', v])
                    fast_memory.add(v)
                else:
                    # No good eviction candidate, just load (shouldn't happen if budget >= 1)
                    operations.append(['load', v])
                    fast_memory.add(v)
            continue
        
        # v is not an input, needs to be computed
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in fast_memory:
                # Load parent
                if len(fast_memory) < fast_memory_budget:
                    operations.append(['load', p])
                    fast_memory.add(p)
                else:
                    # Evict
                    evict_candidates = []
                    for u in fast_memory:
                        if u in outputs:
                            continue
                        evict_candidates.append((ref_count[u], u))
                    if evict_candidates:
                        evict_candidates.sort()
                        evict_v = evict_candidates[0][1]
                        operations.append(['delete', evict_v])
                        fast_memory.discard(evict_v)
                        operations.append(['load', p])
                        fast_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        fast_memory.add(v)
        
        # If v is an output, store it
        if v in outputs:
            operations.append(['store', v])
            stored.add(v)
            fast_memory.discard(v)
    
    return operations

# EVOLVE-BLOCK-END

