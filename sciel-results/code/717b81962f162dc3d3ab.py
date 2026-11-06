"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimize transfers for DAG pebbling.
    
    dag structure (expected):
    - dag['vertices']: list of vertex ids
    - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
    - dag['inputs']: set/list of input vertices (start blue)
    - dag['outputs']: set/list of output vertices (must finish blue)
    
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    # Parse dag structure - handle multiple possible formats
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        # Fallback: assume dag is a tuple or has attributes
        vertices = getattr(dag, 'vertices', [])
        parents = getattr(dag, 'parents', {})
        inputs = set(getattr(dag, 'inputs', []))
        outputs = set(getattr(dag, 'outputs', []))
    
    if not vertices:
        return []
    
    # Build adjacency: children[v] = list of vertices that depend on v
    children = defaultdict(list)
    in_degree = {}
    for v in vertices:
        in_degree[v] = len(parents.get(v, []))
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Count how many times each vertex's value is needed
    # A vertex value is needed by its children
    # If a vertex has only one child and is not an output, its value can be consumed immediately
    usage_count = {}
    for v in vertices:
        # Value of v is needed by each child
        usage_count[v] = len(children[v])
        if v in outputs:
            usage_count[v] += 1  # output requires storing
    
    # Determine which vertices need to be "kept" in memory
    # Strategy: topological order processing
    # - Inputs start blue (in memory)
    # - Compute vertices when all parents are available
    # - Delete vertices when no longer needed
    
    # Track state
    # blue[v] = True if v's value is currently in fast memory
    # We need to track memory usage
    
    ops = []
    blue = set()
    memory_used = 0
    
    # Initialize: inputs are blue
    for v in inputs:
        if v in vertices:
            blue.add(v)
            memory_used += 1
    
    # Topological order processing
    # Use a queue for vertices whose parents are all blue
    ready = deque()
    pending = defaultdict(int)  # number of unprocessed parents
    
    for v in vertices:
        for p in parents.get(v, []):
            pending[v] += 1
    
    for v in vertices:
        if pending[v] == 0:
            ready.append(v)
    
    # We'll process in topological order
    # For each vertex, decide when to compute and when to delete
    
    # Key insight: we want to minimize load/store operations
    # - compute: free (no transfer)
    # - load: brings a value into memory (transfer +1)
    # - store: sends a value out of memory (transfer +1)  
    # - delete: frees memory (no transfer)
    
    # Wait - re-reading: "Inputs start blue" means inputs are already in memory.
    # "every red value consumes one fast-memory slot" - so red values in memory use slots.
    # Operations: load (red->blue? or blue->red?), store, compute, delete
    
    # Let me reconsider the semantics:
    # - load: move value from slow to fast memory (transfer)
    # - store: move value from fast to slow memory (transfer)
    # - compute: compute a new value in fast memory (no transfer, but uses a slot)
    # - delete: remove value from fast memory (no transfer)
    
    # Inputs start blue = already in fast memory
    # Outputs must finish blue = must be in fast memory at end
    
    # So the goal is to minimize load + store operations.
    
    # Strategy:
    # 1. Process in topological order
    # 2. When computing a vertex, ensure all parents are in memory
    # 3. After a vertex is no longer needed, delete it to free memory
    # 4. If memory is full, we may need to evict (store) some value and later reload it
    
    # For now, let's do a simple optimal-ish strategy:
    # - Keep track of what's needed
    # - Only store/delete when necessary
    
    # Reset
    ops = []
    blue = set()
    memory_used = 0
    
    for v in inputs:
        if v in vertices:
            blue.add(v)
            memory_used += 1
    
    # Process in topological order
    remaining = {v: len(parents.get(v, [])) for v in vertices}
    ready = deque()
    for v in vertices:
        if remaining[v] == 0:
            ready.append(v)
    
    processed = set()
    
    while ready:
        v = ready.popleft()
        if v in processed:
            continue
        processed.add(v)
        
        # Check if all parents are in memory
        all_parents_in_memory = all(p in blue for p in parents.get(v, []))
        
        if not all_parents_in_memory:
            # This shouldn't happen in topological order if we're careful
            # But let's handle it: we need to ensure parents are available
            # For simplicity, skip for now - topological order should guarantee this
            pass
        
        # Compute v
        if v not in blue:
            ops.append(['compute', v])
            blue.add(v)
            memory_used += 1
        
        # Check if memory exceeds budget
        # If so, we need to evict some values
        # But for now, let's assume we can manage
        
        # Determine when to delete v
        # v is needed by its children and if it's an output
        num_children = len(children[v])
        is_output = v in outputs
        
        # We can delete v after all its children have been processed
        # For now, we'll track this with a counter
        
        # Add children to ready when their last parent is processed
        for c in children[v]:
            remaining[c] -= 1
            if remaining[c] == 0:
                ready.append(c)
    
    # This simple approach doesn't handle memory pressure well.
    # Let me think of a better approach.
    
    # Better approach:
    # 1. Identify "critical" values that must be kept: inputs (until last use), outputs (until end)
    # 2. For intermediates, keep them only as long as needed
    # 3. When memory is tight, evict the value that won't be needed soonest
    
    # Let me try a different strategy based on usage patterns:
    
    ops = []
    blue = set()
    memory_used = 0
    
    # Inputs start in memory
    for v in inputs:
        if v in vertices:
            blue.add(v)
            memory_used += 1
    
    # We need to process vertices in topological order
    # For each vertex, when we compute it, all parents must be in memory
    # After a vertex is used by all its children, we can delete it (unless it's an output)
    
    # Let's track:
    # - For each vertex, how many times its value is still needed
    # - A vertex's value is needed by each child that hasn't been computed yet
    # - Plus once more if it's an output
    
    remaining_uses = {}
    for v in vertices:
        # Value of v is needed by each child
        remaining_uses[v] = len(children[v])
        if v in outputs:
            remaining_uses[v] += 1
    
    # Process in topological order
    remaining_parents = {v: len(parents.get(v, [])) for v in vertices}
    ready = deque()
    for v in vertices:
        if remaining_parents[v] == 0:
            ready.append(v)
    
    processed = set()
    
    while ready:
        v = ready.popleft()
        if v in processed:
            continue
        processed.add(v)
        
        # Ensure all parents are in memory
        for p in parents.get(v, []):
            if p not in blue:
                # Need to reload parent
                ops.append(['load', p])
                blue.add(p)
                memory_used += 1
        
        # Compute v
        if v not in blue:
            ops.append(['compute', v])
            blue.add(v)
            memory_used += 1
        
        # Decrement remaining uses for parents
        for p in parents.get(v, []):
            remaining_uses[p] -= 1
            # If parent is no longer needed and not an output, delete it
            if remaining_uses[p] == 0 and p not in outputs:
                if p in blue:
                    ops.append(['delete', p])
                    blue.remove(p)
                    memory_used -= 1
        
        # Handle memory pressure: if memory_used > fast_memory_budget, evict
        while memory_used > fast_memory_budget:
            # Find a vertex in blue that is not an output and can be evicted
            # Prefer vertices that won't be needed soon
            evictable = [x for x in blue if x not in outputs and remaining_uses[x] > 0]
            if not evictable:
                # Can't evict, this is a problem
                break
            # Evict the one with fewest remaining uses
            evict_target = min(evictable, key=lambda x: remaining_uses[x])
            ops.append(['store', evict_target])
            blue.remove(evict_target)
            memory_used -= 1
        
        # Add children to ready
        for c in children[v]:
            remaining_parents[c] -= 1
            if remaining_parents[c] == 0:
                ready.append(c)
    
    # Ensure all outputs are in memory at the end
    for v in outputs:
        if v not in blue:
            ops.append(['load', v])
            blue.add(v)
            memory_used += 1
    
    # Limit operations to 20000
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END

