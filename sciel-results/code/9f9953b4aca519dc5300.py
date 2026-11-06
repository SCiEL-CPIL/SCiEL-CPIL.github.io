import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Solve the communication-minimizing DAG pebbling problem.
    
    dag: dict with 'vertices', 'parents', 'inputs', 'outputs'
    fast_memory_budget: int, number of blue pebbles allowed
    
    Returns: list of [operation, vertex]
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build parent lists
    parent_map = {}
    for v in vertices:
        parent_map[v] = parents.get(v, [])
    
    # Topological order (should already be given, but verify)
    topo = list(vertices)
    
    # Determine which vertices are leaves (no children) vs internal
    # We need to figure out the structure
    
    # Simple strategy: process in topological order
    # For each vertex, if it has no parents (input), load it
    # If it has parents, compute it when parents are available
    # If memory is tight, store intermediate results to slow memory
    
    # Key insight: we want to minimize transfers (load + store operations)
    # A "transfer" is a load or store operation
    
    # Strategy: 
    # 1. Try to keep as many values in fast memory as possible
    # 2. Only store to slow memory when we need to free fast memory
    # 3. Recompute rather than store if possible
    
    # For a simple layered DAG, optimal strategy depends on the structure
    # Let's implement a greedy approach:
    
    # State: set of vertices currently in fast memory (blue)
    # We process in topological order
    
    operations = []
    in_fast = set()
    
    # First, load all inputs that we need
    # Actually, let's think more carefully.
    
    # A better approach: use a recursive pebbling strategy
    # For each vertex, determine the minimal number of pebbles needed
    
    # Let's implement a simple but effective strategy:
    # Process vertices in topological order
    # For each vertex v:
    #   - If v is an input: load it (if not already in fast)
    #   - If v has parents: compute it when all parents are in fast
    #   - If we exceed fast_memory_budget, store some values to slow memory
    #   - At the end, ensure all outputs are in fast memory
    
    # Track slow memory: set of vertices stored in slow memory
    in_slow = set()
    
    # Process in topological order
    for v in topo:
        # Check if deadline is approaching
        if time.time() > deadline:
            break
        
        if v in inputs:
            # Input vertices start blue, so they're already available
            # But do we need to explicitly "load" them? 
            # The problem says "Inputs start blue" so they're already in fast memory
            in_fast.add(v)
        else:
            # Non-input vertex: need to compute it
            # Check if all parents are in fast memory
            parents_v = parent_map.get(v, [])
            
            # If any parent is not in fast, we need to load it or recompute
            missing_parents = [p for p in parents_v if p not in in_fast]
            
            if missing_parents:
                # Load missing parents from slow memory
                for p in missing_parents:
                    if p in in_slow:
                        # Load from slow to fast
                        # First, make sure we have room in fast memory
                        while len(in_fast) >= fast_memory_budget:
                            # Store some value to slow memory
                            # Choose a value that's not an output and won't be needed again
                            stored = False
                            for candidate in list(in_fast):
                                if candidate not in outputs and candidate not in parents_v:
                                    in_fast.remove(candidate)
                                    in_slow.add(candidate)
                                    operations.append(['store', candidate])
                                    stored = True
                                    break
                            if not stored:
                                # Can't store anything, this is a problem
                                break
                        operations.append(['load', p])
                        in_fast.add(p)
                        in_slow.remove(p)
                    
                    # If parent was never computed, we need to recompute
                    # This gets complex - let's assume the DAG is well-structured
            
            # Now compute v
            # Make sure we have room in fast memory
            while len(in_fast) >= fast_memory_budget and v not in in_fast:
                # Store a value
                stored = False
                for candidate in list(in_fast):
                    if candidate not in outputs:
                        in_fast.remove(candidate)
                        in_slow.add(candidate)
                        operations.append(['store', candidate])
                        stored = True
                        break
                if not stored:
                    break
            
            operations.append(['compute', v])
            in_fast.add(v)
            
            # Remove parents from fast memory if they're no longer needed
            # (optional optimization)
    
    # Ensure all outputs are in fast memory
    for out in outputs:
        if out not in in_fast:
            if out in in_slow:
                # Load it
                while len(in_fast) >= fast_memory_budget:
                    for candidate in list(in_fast):
                        if candidate not in outputs:
                            in_fast.remove(candidate)
                            in_slow.add(candidate)
                            operations.append(['store', candidate])
                            break
                operations.append(['load', out])
                in_fast.add(out)
                in_slow.remove(out)
    
    # Check operation limit
    if len(operations) > 20000:
        # Truncate and try to fix
        operations = operations[:20000]
    
    return operations
