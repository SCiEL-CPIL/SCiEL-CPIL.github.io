# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy: Layered pebbling with careful memory management.
    We process the DAG in topological layers, loading inputs, computing nodes,
    and managing fast memory to minimize total transfers.
    
    Key insight: For layered DAGs, we can process layer by layer.
    - Inputs start blue (available)
    - We need outputs to finish blue
    - Red values consume fast memory slots
    - Operations: load, store, compute, delete
    
    The baseline likely does a simple topological pass. We can improve by:
    1. Better ordering of computations within layers
    2. Pre-loading dependencies that will be reused
    3. Minimizing redundant loads
    """
    deadline = time.time() + 45  # Safety margin before 60s limit
    
    # Extract DAG structure
    # dag is expected to have:
    # - dag['vertices'] or dag['nodes']: list of vertices
    # - dag['parents']: dict or list mapping vertex -> list of parent vertices
    # - dag['outputs']: list of output vertices
    # - dag['inputs'] or dag['sources']: list of input vertices (no parents)
    
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', dag.get('nodes', [])))
        parents = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        inputs = set(dag.get('inputs', dag.get('sources', [])))
    else:
        # Try to handle other formats
        vertices = list(dag)
        parents = {}
        outputs = set()
        inputs = set()
    
    if not vertices:
        return []
    
    # Ensure parents is a dict
    if not isinstance(parents, dict):
        # Maybe it's a list of lists
        if isinstance(parents, list) and len(parents) == len(vertices):
            parent_dict = {}
            for i, v in enumerate(vertices):
                parent_dict[v] = parents[i] if parents[i] else []
            parents = parent_dict
        else:
            parents = {v: [] for v in vertices}
    
    # Build reverse mapping: children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Determine inputs (vertices with no parents)
    if not inputs:
        inputs = set(v for v in vertices if not parents.get(v, []))
    
    # Determine outputs if not specified
    if not outputs:
        # Outputs are vertices with no children
        outputs = set(v for v in vertices if not children.get(v, []))
    
    # Topological sort (Kahn's algorithm)
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = [v for v in vertices if in_degree[v] == 0]
    # Sort by vertex index for determinism
    queue.sort()
    
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
        queue.sort()
    
    if len(topo_order) != len(vertices):
        # Cycle or issue - fallback
        topo_order = list(vertices)
    
    # Compute layers based on longest path from source
    layer = {}
    for v in topo_order:
        if not parents.get(v, []):
            layer[v] = 0
        else:
            layer[v] = max(layer.get(p, 0) for p in parents.get(v, [])) + 1
    
    max_layer = max(layer.values()) if layer else 0
    
    # Group by layer
    layers = [[] for _ in range(max_layer + 1)]
    for v in topo_order:
        layers[layer[v]].append(v)
    
    # State tracking
    # blue: value is in main memory (available without load)
    # red: value is in fast memory (consumes a slot)
    # not_computed: value hasn't been computed yet
    state = {v: 'not_computed' for v in vertices}
    fast_memory = set()  # vertices currently in fast memory (red)
    operations = []
    
    def is_available(v):
        """Check if vertex value is available without load."""
        return state[v] in ('blue', 'red')
    
    def get_value(v):
        """Get the value of vertex, performing load if needed."""
        if state[v] == 'red':
            # Already in fast memory
            return v
        elif state[v] == 'blue':
            # In main memory, need to load to fast memory
            if len(fast_memory) >= fast_memory_budget:
                # Need to evict something
                # Find a value that won't be needed soon
                evict_v = _find_evict_candidate(v, fast_memory, layers, layer, topo_order, children)
                if evict_v is not None:
                    operations.append(['store', evict_v])
                    fast_memory.remove(evict_v)
                    state[evict_v] = 'blue'
                else:
                    # Emergency: just evict the first one
                    evict_v = next(iter(fast_memory))
                    operations.append(['store', evict_v])
                    fast_memory.remove(evict_v)
                    state[evict_v] = 'blue'
            operations.append(['load', v])
            fast_memory.add(v)
            state[v] = 'red'
            return v
        else:
            # Not computed - should not happen if we're doing topological order
            raise ValueError(f"Vertex {v} not computed but needed")
    
    def _find_evict_candidate(needing_v, fm, layers, layer, topo_order, children):
        """Find a vertex in fast memory to evict that minimizes future loads."""
        # Score each candidate by how soon it will be needed again
        # If never needed again, prefer to evict it
        # If needed soon, keep it
        
        # Build next-usage map
        # For simplicity, count remaining uses
        remaining_uses = {v: 0 for v in fm}
        for c in topo_order:
            if c in remaining_uses:
                for p in parents.get(c, []):
                    if p in remaining_uses:
                        remaining_uses[p] += 1
        
        # Prefer evicting vertices with fewest remaining uses
        # Break ties by preferring to evict non-outputs
        candidates = sorted(fm, key=lambda v: (remaining_uses.get(v, 0), v in outputs, v))
        if candidates:
            return candidates[0]
        return None
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        if v in inputs:
            # Inputs start blue
            state[v] = 'blue'
            continue
        
        # Get parents (dependencies)
        ps = parents.get(v, [])
        if not ps:
            # Should be an input, but handle gracefully
            state[v] = 'blue'
            continue
        
        # Load all parents into fast memory
        for p in ps:
            get_value(p)
        
        # Check deadline
        if time.time() > deadline:
            break
        
        # Compute v
        operations.append(['compute', v])
        state[v] = 'red'
        fast_memory.add(v)
        
        # Check if we can delete parents that are no longer needed
        for p in ps:
            # Check if p is still needed by any other vertex
            still_needed = False
            for c in children.get(p, []):
                if state[c] == 'not_computed' and c not in inputs:
                    still_needed = True
                    break
            
            if not still_needed and p in fast_memory and p not in outputs:
                operations.append(['delete', p])
                fast_memory.remove(p)
                state[p] = 'blue'
    
    # Ensure all outputs are blue (in main memory)
    for v in sorted(outputs):
        if time.time() > deadline:
            break
        if state[v] == 'red':
            operations.append(['store', v])
            fast_memory.remove(v)
            state[v] = 'blue'
        elif state[v] == 'not_computed':
            # Should not happen, but handle gracefully
            pass
    
    # Limit operations to 20000
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
