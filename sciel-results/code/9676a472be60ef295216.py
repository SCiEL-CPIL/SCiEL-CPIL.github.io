"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    dag: dict with:
      - 'vertices': list of vertex IDs (topologically ordered)
      - 'parents': dict mapping vertex -> list of parent vertices
      - 'inputs': set of input vertices
      - 'outputs': set of output vertices
    fast_memory_budget: int, max number of red values in fast memory
    
    Returns: list of [operation, vertex]
    """
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin
    
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
    
    # Track state: 'blue' (in fast memory), 'red' (not in fast memory, needs reload/recompute)
    # For inputs: start blue (in fast memory)
    # For others: start red
    
    # Strategy: Process in topological order, but with smart eviction
    # Keep track of which vertices are in fast memory
    in_fast = set(inputs)  # inputs start in fast memory
    fast_count = len(inputs)
    
    operations = []
    
    # Count how many times each vertex is needed as a parent
    # This helps decide what to keep
    use_count = {}
    for v in vertices:
        use_count[v] = 0
    for v in vertices:
        for p in parents[v]:
            use_count[p] += 1
    
    # Process vertices in topological order
    for v in vertices:
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
        
        # If v is already in fast memory, skip
        if v in in_fast:
            continue
        
        # Check if all parents are in fast memory
        parents_in_fast = all(p in in_fast for p in parents[v])
        
        if not parents_in_fast:
            # Need to bring in missing parents
            for p in parents[v]:
                if p not in in_fast:
                    # Need to load p (if input) or recompute p
                    # For simplicity, we'll load inputs and recompute others
                    # But recomputation requires their parents...
                    # This gets complex. Let's use a simpler approach.
                    pass
        
        # Simplified approach: Process layer by layer
        # Actually, let's try a different strategy
        break
    
    # Simpler greedy strategy:
    # 1. Load all inputs
    # 2. Process in topological order
    # 3. When fast memory is full, evict the vertex with lowest future use
    
    in_fast = set()
    operations = []
    
    # Load all inputs
    for inp in inputs:
        if inp not in in_fast:
            operations.append(['load', inp])
            in_fast.add(inp)
            if len(in_fast) > fast_memory_budget:
                # Evict least useful
                # For now, just evict any non-output
                for x in list(in_fast):
                    if x not in outputs:
                        operations.append(['delete', x])
                        in_fast.remove(x)
                        break
    
    # Process in topological order
    for v in vertices:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in in_fast:
            continue
        
        # Ensure parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                # Need to get p into fast memory
                # If p is input, load it
                # If p is computed, recompute it
                # For now, just load inputs
                if p in inputs:
                    operations.append(['load', p])
                    in_fast.add(p)
                    if len(in_fast) > fast_memory_budget:
                        for x in list(in_fast):
                            if x not in outputs and x != v:
                                operations.append(['delete', x])
                                in_fast.remove(x)
                                break
        
        # Compute v
        if v not in in_fast:
            operations.append(['compute', v])
            in_fast.add(v)
            if len(in_fast) > fast_memory_budget:
                for x in list(in_fast):
                    if x not in outputs and x != v:
                        operations.append(['delete', x])
                        in_fast.remove(x)
                        break
    
    # Store all outputs
    for out in outputs:
        if out in in_fast:
            operations.append(['store', out])
        else:
            # Need to bring output into fast memory
            # This requires its parents, etc.
            # For now, just load if input
            if out in inputs:
                operations.append(['load', out])
                operations.append(['store', out])
    
    return operations
# EVOLVE-BLOCK-END

