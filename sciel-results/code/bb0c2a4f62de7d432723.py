"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling with communication minimization.
    
    dag: dict with keys like 'vertices', 'parents', 'inputs', 'outputs', 'topo_order'
    fast_memory_budget: int, number of fast memory slots
    Returns: list of [operation, vertex] where operation is in {load, store, compute, delete}
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        topo_order = dag.get('topo_order', [])
    else:
        # Try to handle other formats
        vertices = list(dag.keys()) if isinstance(dag, dict) else []
        parents = {}
        inputs = []
        outputs = []
        topo_order = []
    
    # If no topo_order, compute it
    if not topo_order and vertices:
        # Simple topological sort using parents
        in_degree = {v: len(parents.get(v, [])) for v in vertices}
        from collections import deque
        queue = deque([v for v in vertices if in_degree[v] == 0])
        topo_order = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            for u in vertices:
                if v in parents.get(u, []):
                    in_degree[u] -= 1
                    if in_degree[u] == 0:
                        queue.append(u)
    
    if not vertices:
        return []
    
    # Build children map for reverse traversal
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Identify input vertices (no parents) and output vertices
    input_vertices = set(v for v in vertices if len(parents.get(v, [])) == 0)
    if inputs:
        input_vertices = set(inputs)
    output_vertices = set(outputs) if outputs else set()
    
    # For layered DAGs, use a depth-first recursive pebbling strategy
    # The key idea: process the DAG in topological order, but use a stack-based
    # approach to manage memory, keeping only values that are needed soon
    
    # Strategy: Use a greedy approach with lookahead
    # - Load inputs into fast memory
    # - Compute vertices when all parents are available
    # - Store to slow memory when fast memory is full
    # - Delete values that are no longer needed
    
    # Count how many times each vertex's value is needed (out-degree in reverse)
    needed_count = {v: 0 for v in vertices}
    for v in vertices:
        for child in children.get(v, []):
            needed_count[v] += 1
    
    # Fast memory state
    fast_mem = set()  # vertices currently in fast memory
    ops = []
    
    # Track which vertices have been computed and are available
    computed = set()
    in_fast = set()
    in_slow = set()
    
    # Start by loading inputs
    for v in input_vertices:
        ops.append(['load', v])
        in_fast.add(v)
        fast_mem.add(v)
        computed.add(v)
    
    # Process vertices in topological order
    # Use a priority: process vertices whose values will be needed soon
    # to minimize the number of times we need to reload from slow memory
    
    # Build a schedule: for each vertex, determine when to compute it
    # and manage memory
    
    # Simple greedy: process in topo order, load parents if needed,
    # compute, and manage memory by evicting least-needed values
    
    remaining = list(topo_order)
    # Remove inputs from remaining (already loaded)
    remaining = [v for v in remaining if v not in input_vertices]
    
    # For each vertex in topo order, ensure parents are available, compute, manage memory
    for v in remaining:
        if time.time() > deadline:
            break
            
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                # Need to load p from slow memory
                if p in in_slow or p in computed:
                    # Evict if necessary
                    if len(fast_mem) >= fast_memory_budget:
                        # Find a value to evict - prefer values with low needed_count
                        # that are not parents of v
                        evict_candidates = [x for x in fast_mem if x != v and x not in parents.get(v, [])]
                        if evict_candidates:
                            # Evict the one with lowest needed_count
                            evict_target = min(evict_candidates, key=lambda x: needed_count.get(x, 0))
                            ops.append(['delete', evict_target])
                            fast_mem.discard(evict_target)
                            in_fast.discard(evict_target)
                            in_slow.add(evict_target)
                        else:
                            # Must evict a parent - this is bad but necessary
                            evict_target = parents.get(v, [])[0]
                            ops.append(['delete', evict_target])
                            fast_mem.discard(evict_target)
                            in_fast.discard(evict_target)
                            in_slow.add(evict_target)
                    
                    ops.append(['load', p])
                    in_fast.add(p)
                    fast_mem.add(p)
                else:
                    # p hasn't been computed yet - this shouldn't happen in topo order
                    pass
        
        # Compute v
        ops.append(['compute', v])
        computed.add(v)
        in_fast.add(v)
        fast_mem.add(v)
        
        # If v is an output, keep it in fast memory
        # Otherwise, if memory is tight, consider storing to slow memory
        if v not in output_vertices and len(fast_mem) > fast_memory_budget:
            # Store v to slow memory and remove from fast
            ops.append(['store', v])
            fast_mem.discard(v)
            in_fast.discard(v)
            in_slow.add(v)
    
    # Ensure outputs are in fast memory at the end
    for v in output_vertices:
        if v not in in_fast:
            if v in computed:
                # Need to load it back
                if len(fast_mem) >= fast_memory_budget:
                    evict_candidates = [x for x in fast_mem if x not in output_vertices]
                    if evict_candidates:
                        evict_target = min(evict_candidates, key=lambda x: needed_count.get(x, 0))
                        ops.append(['delete', evict_target])
                        fast_mem.discard(evict_target)
                        in_fast.discard(evict_target)
                        in_slow.add(evict_target)
                ops.append(['load', v])
                in_fast.add(v)
                fast_mem.add(v)
    
    return ops
# EVOLVE-BLOCK-END

