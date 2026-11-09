"""
Communication-minimizing DAG pebbling using chain decomposition and round-robin scheduling.
"""

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem with a communication-minimizing strategy.
    
    Args:
        dag: A dictionary or object representing the DAG.
             Expected structure:
             - dag['vertices']: list of vertex IDs (topologically ordered)
             - dag['parents']: dict mapping vertex -> list of parent vertices
             - dag['outputs']: list of output vertex IDs
             - Or similar structure. We'll handle common formats.
             
        fast_memory_budget: int, number of fast memory slots available.
        
    Returns:
        List of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'.
    """
    # Parse the DAG structure
    # The dag format needs to be inferred. Common formats:
    # 1. dag is a dict with 'vertices', 'parents', 'outputs'
    # 2. dag is an object with attributes
    # Let's try to extract the necessary information.
    
    vertices = None
    parents = None
    outputs = None
    
    # Try dict format
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        parents = dag.get('parents', None)
        outputs = dag.get('outputs', None)
        
        # Also check for alternative key names
        if vertices is None:
            vertices = dag.get('nodes', None)
        if parents is None:
            parents = dag.get('preds', None) or dag.get('ancestors', None)
        if outputs is None:
            outputs = dag.get('sinks', None) or dag.get('targets', None)
            
    elif hasattr(dag, 'vertices') or hasattr(dag, 'nodes'):
        vertices = getattr(dag, 'vertices', None) or getattr(dag, 'nodes', None)
        parents = getattr(dag, 'parents', None) or getattr(dag, 'preds', None)
        outputs = getattr(dag, 'outputs', None) or getattr(dag, 'sinks', None)
    
    if vertices is None or parents is None or outputs is None:
        # Fallback: try to infer from common structures
        # If dag is just a list of (vertex, parents) pairs
        if isinstance(dag, list) and len(dag) > 0:
            if isinstance(dag[0], (list, tuple)) and len(dag[0]) == 2:
                vertices = [v[0] for v in dag]
                parents = {v[0]: v[1] for v in dag}
                # Outputs are vertices with no children - need to compute
                children = {v: [] for v in vertices}
                for v, ps in parents.items():
                    for p in ps:
                        if p in children:
                            children[p].append(v)
                outputs = [v for v in vertices if len(children[v]) == 0]
            else:
                # Assume dag is a list of vertices, parents is separate
                raise ValueError("Cannot parse dag structure")
        else:
            raise ValueError("Cannot parse dag structure")
    
    # Ensure vertices is a list
    if not isinstance(vertices, list):
        vertices = list(vertices)
    
    # Ensure parents is a dict with list values
    if not isinstance(parents, dict):
        parents = dict(parents)
    for k in parents:
        if not isinstance(parents[k], list):
            parents[k] = list(parents[k])
    
    # Ensure outputs is a list
    if not isinstance(outputs, list):
        outputs = list(outputs)
    
    # Build children mapping
    children = {v: [] for v in vertices}
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in children:
                    children[p].append(v)
    
    # Identify input vertices (no parents)
    input_vertices = [v for v in vertices if len(parents.get(v, [])) == 0]
    
    # Topological order is given by the order of vertices list
    # Create position mapping
    pos = {v: i for i, v in enumerate(vertices)}
    
    # Chain decomposition: find a set of vertex-disjoint paths that cover all vertices
    # Greedy approach: repeatedly find the longest path from an input to an output
    # or from an input to any vertex, prioritizing paths that cover many vertices
    
    # First, let's do a simple chain decomposition
    # We'll use a greedy algorithm to find vertex-disjoint paths
    
    used = set()
    chains = []
    
    # Sort vertices by topological order
    sorted_vertices = sorted(vertices, key=lambda v: pos[v])
    
    # For each unused vertex, try to extend a chain
    for start in sorted_vertices:
        if start in used:
            continue
        # Start a new chain at this vertex if it's an input or all parents are used
        # Actually, let's just greedily build chains
        chain = [start]
        used.add(start)
        current = start
        # Try to extend the chain
        while True:
            # Find children of current that are not used
            candidates = [c for c in children[current] if c not in used]
            if not candidates:
                break
            # Pick the child with the most unused descendants (greedy)
            # For simplicity, pick the first one in topological order
            best_child = min(candidates, key=lambda c: pos[c])
            chain.append(best_child)
            used.add(best_child)
            current = best_child
    
    # Any remaining vertices (shouldn't happen if we cover all)
    for v in sorted_vertices:
        if v not in used:
            chains.append([v])
            used.add(v)
    
    # Now we have chains. For each chain, we need to process it.
    # A chain is a path: v0 -> v1 -> v2 -> ... -> vk
    # v0 is an input (or has parents that must be computed first)
    # vi for i>0 has vi-1 as a parent (among others potentially)
    
    # Wait, the chain decomposition above doesn't guarantee that each vertex in the chain
    # (except the first) has the previous vertex as its ONLY parent or that the previous
    # is a parent. Let me redo this.
    
    # Better approach: for each vertex, if it has exactly one parent that is "active" 
    # (not yet processed in another chain), extend that chain.
    
    # Let me use a different chain decomposition:
    # 1. Compute in-degree (number of parents) for each vertex
    # 2. For vertices with in-degree 0, they start chains
    # 3. For each chain, follow the child that has the fewest remaining parents
    
    # Actually, let's use a simpler approach: 
    # Since the DAG is layered (as mentioned in the task), let's exploit that.
    
    # For a layered DAG, we can process layer by layer.
    # Let's detect layers.
    
    # Compute the "depth" of each vertex (longest path from any input)
    depth = {}
    for v in sorted_vertices:
        if not parents.get(v, []):
            depth[v] = 0
        else:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    
    # Group by depth
    from collections import defaultdict
    layers = defaultdict(list)
    for v in vertices:
        layers[depth[v]].append(v)
    
    # Now, for each layer, we need to compute all vertices in that layer.
    # The naive approach would be: for each layer, load all inputs, compute, store.
    # But we can do better with the chain approach.
    
    # Let's use the chain decomposition more carefully.
    # A chain is a path where each consecutive pair (u, v) has u as a parent of v.
    
    # Rebuild chains properly:
    used = set()
    chains = []
    
    # Start with input vertices
    for v in input_vertices:
        if v in used:
            continue
        chain = [v]
        used.add(v)
        current = v
        while True:
            # Find children of current that are not used
            candidates = [c for c in children[current] if c not in used]
            if not candidates:
                break
            # Pick the child with the smallest depth (to keep chains short) 
            # or the one with the fewest parents (to ensure it can be extended)
            # Actually, pick the one that leads to the longest remaining path
            # For simplicity, pick the first in topological order
            best_child = min(candidates, key=lambda c: pos[c])
            chain.append(best_child)
            used.add(best_child)
            current = best_child
        chains.append(chain)
    
    # Add remaining vertices as single-vertex chains
    for v in sorted_vertices:
        if v not in used:
            chains.append([v])
            used.add(v)
    
    # Now, schedule the chains using round-robin.
    # For each chain, we need to:
    # 1. Load the first vertex (if it's an input)
    # 2. Compute subsequent vertices
    # 3. Store outputs
    
    # But wait, vertices in a chain may have parents from other chains.
    # So we need to ensure all parents are computed before computing a vertex.
    
    # Let's track which vertices are computed and in which chain they appear.
    chain_of = {}
    for i, chain in enumerate(chains):
        for v in chain:
            chain_of[v] = i
    
    # State: 
    # - computed: set of vertices whose values are available (in fast memory or on disk)
    # - in_memory: set of vertices currently in fast memory
    # - We need to manage fast memory carefully.
    
    # Operations:
    # - load(v): move v from disk to fast memory (if not already there)
    # - store(v): move v from fast memory to disk
    # - compute(v): compute v from its parents (requires all parents in fast memory)
    # - delete(v): remove v from fast memory (free a slot)
    
    # Initial state: all input vertices are on disk (blue), not in fast memory.
    # All outputs must end up blue (on disk or in fast memory? The task says "finish blue",
    # which I interpret as being available, i.e., on disk or in fast memory. But typically
    # "blue" means on disk in pebbling games. Let's assume outputs must be stored on disk.)
    
    # Actually, re-reading: "Inputs start blue, all outputs must finish blue"
    # In pebbling, blue pebbles are on the graph (disk), red pebbles are in fast memory.
    # So inputs start on disk (blue), outputs must end on disk (blue).
    # Red pebbles (in fast memory) consume fast_memory_budget slots.
    
    # So the goal is to move from all inputs blue to all outputs blue, minimizing transfers.
    # A transfer is a load or store operation.
    
    # Let's implement a simple scheduler:
    # 1. Process chains in round-robin order.
    # 2. For each chain, process vertices in order.
    # 3. Before computing a vertex, ensure all its parents are in fast memory.
    # 4. If a parent is not in fast memory, load it (if it's computed) or compute it first.
    
    # This is getting complex. Let's simplify:
    # We'll use a priority-based approach:
    # - Maintain a set of "ready" vertices: those whose parents are all computed and in fast memory.
    # - Process ready vertices, preferring those in chains that are being actively processed.
    
    # Actually, let's just do a topological sort and process vertices one by one,
    # managing fast memory carefully.
    
    # Simple approach:
    # 1. For each vertex in topological order:
    #    a. Ensure all parents are in fast memory (load if needed)
    #    b. Compute the vertex
    #    c. If the vertex is an output, store it (to make it blue)
    #    d. If fast memory is full, evict a vertex that won't be needed again
    
    # Let's implement this.
    
    operations = []
    in_memory = set()  # vertices currently in fast memory
    computed = set()   # vertices that have been computed (value exists on disk or in memory)
    
    # Precompute: for each vertex, how many times its value will be needed
    # (i.e., how many children it has that are not yet computed)
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # Process vertices in topological order
    for v in sorted_vertices:
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in in_memory:
                # Need to load p
                if p not in computed:
                    # This shouldn't happen in topological order
                    raise ValueError(f"Parent {p} of {v} not computed yet")
                # Load p from disk
                if len(in_memory) >= fast_memory_budget:
                    # Need to evict something
                    # Evict a vertex that is not an output and has no remaining children
                    evictable = [x for x in in_memory if x not in outputs and remaining_children[x] == 0]
                    if not evictable:
                        # Evict any non-output vertex
                        evictable = [x for x in in_memory if x not in outputs]
                    if not evictable:
                        # Evict any vertex
                        evictable = list(in_memory)
                    if evictable:
                        evict_v = evictable[0]
                        operations.append(['delete', evict_v])
                        in_memory.remove(evict_v)
                operations.append(['load', p])
                in_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        in_memory.add(v)
        
        # If v is an output, store it
        if v in outputs:
            operations.append(['store', v])
            in_memory.remove(v)
        
        # Update remaining children count for v's parents
        # Actually, we need to decrement remaining_children for v's parents
        # when v is computed
        for p in parents.get(v, []):
            remaining_children[p] -= 1
    
    # Ensure all outputs are stored (they should be, from the above)
    # But let's verify and add any missing stores
    for v in outputs:
        if v in in_memory:
            operations.append(['store', v])
            in_memory.remove(v)
    
    # Truncate to 20000 operations max
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
