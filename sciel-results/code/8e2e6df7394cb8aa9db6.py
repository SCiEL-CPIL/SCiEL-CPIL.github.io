import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Args:
        dag: dict with 'parents' (topologically ordered parents per vertex) and 'outputs' (list of output vertices)
        fast_memory_budget: number of fast-memory slots available
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', 'delete'
    """
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Parse the DAG structure
    # Assume dag is a dict: dag['parents'][v] = list of parent indices, dag['outputs'] = list of output indices
    # Vertices are 0-indexed, topologically ordered (parents come before children)
    
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        # Determine number of vertices
        if 'n' in dag:
            n = dag['n']
        elif 'num_vertices' in dag:
            n = dag['num_vertices']
        else:
            # Infer from parents keys
            n = max(parents.keys()) + 1 if parents else 0
            for v in outputs:
                if v >= n:
                    n = v + 1
    else:
        # Fallback: assume dag is a list of parent lists
        parents = {}
        for i, p in enumerate(dag):
            parents[i] = list(p)
        outputs = []
        n = len(dag)
        # Outputs are vertices with no children (sinks)
        has_child = set()
        for p in parents.values():
            for c in p:
                has_child.add(c)
        for i in range(n):
            if i not in has_child:
                outputs.append(i)
    
    # Convert parents to list of lists for easier access
    parent_list = [parents.get(i, []) for i in range(n)]
    
    # Identify input vertices (no parents) and output vertices
    input_vertices = [i for i in range(n) if len(parent_list[i]) == 0]
    output_vertices = list(outputs) if outputs else [i for i in range(n) if all(j != i for j in range(n) for p in parent_list[j] if i in p)]
    # Better: outputs are given; if not, find sinks
    if not output_vertices:
        has_child = set()
        for i in range(n):
            for p in parent_list[i]:
                has_child.add(p)
        output_vertices = [i for i in range(n) if i not in has_child]
    
    # Strategy: For each output, we need it blue at the end.
    # Naive approach: compute everything in topological order, storing results.
    # Optimized: Use fast memory budget to minimize transfers.
    
    # Key insight: We can recompute values instead of storing them.
    # A value needs to be stored (transferred to slow memory) if it's used by multiple children
    # or if it's an output and we don't have enough fast memory to keep it.
    
    # Let's think about this differently:
    # - Blue = in fast memory (or already in slow memory and loaded)
    # - Red = in fast memory
    # - Actually: "Inputs start blue" means they're available. "All outputs must finish blue."
    # - "Every red value consumes one fast-memory slot"
    
    # Operations:
    # load: move value from slow to fast (becomes blue/red in fast memory)
    # store: move value from fast to slow
    # compute: compute a value from its parents (parents must be available)
    # delete: remove a value from fast memory (frees a slot)
    
    # A value is "available" if it's in fast memory (blue or red).
    # Inputs start blue (available, no cost).
    
    # Strategy: Process vertices in topological order. For each vertex, ensure its parents
    # are available, compute it, then decide whether to keep it or delete it.
    
    # To minimize transfers, we want to:
    # 1. Only load a value from slow memory if it's needed and not already in fast memory
    # 2. Only store a value to slow memory if it will be needed later and we can't keep it in fast memory
    # 3. Delete values from fast memory when they're no longer needed to free slots
    
    # Let's compute for each vertex: how many times it's needed as a parent
    use_count = [0] * n
    for i in range(n):
        for p in parent_list[i]:
            use_count[p] += 1
    
    # A vertex needs to be stored to slow memory if:
    # - It's used more than once (so we need it available multiple times) OR
    # - It's an output (must finish blue)
    # AND we can't keep it in fast memory long enough
    
    # Simpler approach: Greedy schedule
    # Track what's in fast memory
    # Process in topological order
    
    # Let me try a different, cleaner approach:
    # 
    # Phase 1: Determine which vertices need to be "materialized" (stored to slow)
    # A vertex must be stored to slow if it has multiple children (used multiple times)
    # or is an output. Otherwise, it can be computed and consumed immediately.
    #
    # Phase 2: Schedule the operations respecting fast memory budget.
    
    # For a layered DAG, the optimal strategy is often:
    # - Process layer by layer
    # - Within a layer, compute values and use them for the next layer
    # - Store intermediate values that are needed by multiple later vertices
    
    # Let me implement a general greedy algorithm:
    
    # State: fast_memory = set of vertex indices currently in fast memory
    # We process vertices in topological order (which is 0, 1, 2, ..., n-1 if properly ordered)
    
    # For each vertex v in topological order:
    #   1. Ensure all parents of v are in fast memory (load if not)
    #   2. Compute v (adds v to fast memory)
    #   3. If fast memory is full and we need to free space, delete some value
    #   4. If v needs to be stored (multiple uses or output), store it to slow
    
    # But this is tricky because we need to know future needs.
    
    # Let me try: For each vertex, determine if it should be "kept" in fast memory
    # or "spilled" to slow memory.
    
    # A vertex should be spilled to slow memory if:
    # - It's used by multiple children (use_count > 1) OR it's an output
    # - AND we can't afford to keep it in fast memory
    
    # For a simple layered DAG, the optimal is to keep one layer in fast memory.
    
    # Let me implement a straightforward algorithm:
    
    operations = []
    fast_mem = set()  # vertices currently in fast memory
    in_slow = set()   # vertices stored in slow memory
    
    # Inputs start blue (in fast memory conceptually, but let's say they're available)
    # Actually, "inputs start blue" - they're already available, no load needed
    
    # Process in topological order
    for v in range(n):
        if time.time() > deadline:
            break
        
        # Ensure parents are available in fast memory
        for p in parent_list[v]:
            if p not in fast_mem:
                if p in in_slow:
                    # Load from slow memory
                    operations.append(['load', p])
                    fast_mem.add(p)
                elif p in input_vertices:
                    # Input, should be available - add to fast mem
                    fast_mem.add(p)
                else:
                    # Should have been computed and stored
                    # If not in slow or fast, there's an issue
                    # Recompute? For now, load from slow (assume it was stored)
                    if p in in_slow:
                        operations.append(['load', p])
                        fast_mem.add(p)
                    else:
                        # This shouldn't happen in a correct schedule
                        # Force load
                        operations.append(['load', p])
                        fast_mem.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_mem.add(v)
        
        # Decide if v needs to be stored to slow memory
        needs_store = (use_count[v] > 1) or (v in output_vertices)
        
        if needs_store:
            # Store to slow memory
            operations.append(['store', v])
            in_slow.add(v)
            # After storing, we can delete from fast memory if we're running low
            # But keep it for now if we have room
            if len(fast_mem) > fast_memory_budget:
                # Need to free space - delete v or another value
                # Delete v since we just stored it
                operations.append(['delete', v])
                fast_mem.discard(v)
        else:
            # Don't store, just keep in fast memory
            # But if fast memory is full, we need to delete something
            pass
        
        # Manage fast memory size
        while len(fast_mem) > fast_memory_budget:
            # Find a value to delete that won't be needed soon
            # Prefer to delete values that are already stored in slow memory
            # or have no more uses
            candidates = []
            for u in fast_mem:
                if u in in_slow:
                    # Already stored, safe to delete
                    candidates.append((0, u))
                else:
                    # Not stored, check if still needed
                    # Count remaining uses
                    remaining = 0
                    for w in range(v + 1, n):
                        if u in parent_list[w]:
                            remaining += 1
                    if remaining == 0:
                        candidates.append((1, u))
                    else:
                        candidates.append((2, u))
            
            if candidates:
                candidates.sort()
                to_delete = candidates[0][1]
                operations.append(['delete', to_delete])
                fast_mem.discard(to_delete)
            else:
                break
    
    # Ensure all outputs are blue (in fast memory) at the end
    for v in output_vertices:
        if v not in fast_mem:
            if v in in_slow:
                operations.append(['load', v])
                fast_mem.add(v)
            else:
                # Shouldn't happen
                operations.append(['load', v])
                fast_mem.add(v)
        
        # Manage fast memory
        while len(fast_mem) > fast_memory_budget:
            candidates = []
            for u in fast_mem:
                if u in in_slow and u != v:
                    candidates.append((0, u))
                else:
                    candidates.append((1, u))
            candidates.sort()
            if candidates:
                to_delete = candidates[0][1]
                if to_delete in fast_mem and to_delete != v:
                    operations.append(['delete', to_delete])
                    fast_mem.discard(to_delete)
            else:
                break
    
    return operations
