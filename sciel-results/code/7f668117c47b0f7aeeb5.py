"""
Communication-minimizing DAG pebbling solver.

Strategy:
- Parse the DAG structure (topologically ordered parents, output vertices).
- Use a time-ordered pebbling strategy that minimizes communication
  (load/store operations) by keeping computed values in fast memory as long as
  they are needed, and only storing to slow memory when fast memory is full.
- Recomputation is allowed and can be used to reduce communication.
- Track elapsed time and return best result before deadline.

The score is mean transfers / naive schedule. Lower is better.
"""

def pebble(dag, fast_memory_budget):
    import time
    import sys
    
    START_TIME = time.time()
    DEADLINE = 90.0  # seconds, must return before this
    
    def check_time():
        return time.time() - START_TIME < DEADLINE - 2.0  # 2 second safety margin
    
    # Parse DAG structure
    # dag is expected to be a dict with:
    #   'nodes': dict mapping node_id -> {'parents': [...], 'input': bool, 'output': bool}
    #   or similar structure. Let's handle multiple possible formats.
    
    if isinstance(dag, dict):
        nodes = dag.get('nodes', {})
        output_vertices = set(dag.get('outputs', []))
        if not output_vertices:
            # Try to find output vertices
            for nid, data in nodes.items():
                if isinstance(data, dict) and data.get('output'):
                    output_vertices.add(nid)
                elif isinstance(data, dict) and not data.get('parents') and not data.get('input'):
                    # Could be output if no children reference it
                    pass
    else:
        # Fallback: assume dag is a list of node descriptors
        nodes = {}
        output_vertices = set()
        for i, item in enumerate(dag):
            if isinstance(item, dict):
                nodes[i] = item
                if item.get('output'):
                    output_vertices.add(i)
            else:
                nodes[i] = {'parents': []}
    
    # Build adjacency: children map
    children = {nid: [] for nid in nodes}
    for nid, data in nodes.items():
        if isinstance(data, dict):
            parents = data.get('parents', [])
            for p in parents:
                if p in children:
                    children[p].append(nid)
    
    # Determine topological order (should already be given, but verify)
    # Assume nodes are already topologically ordered if they have integer keys
    # or if the list is in topo order
    if isinstance(dag, dict) and 'nodes' in dag:
        topo_order = list(nodes.keys())
    else:
        topo_order = sorted(nodes.keys())
    
    # Identify input nodes
    input_nodes = set()
    for nid, data in nodes.items():
        if isinstance(data, dict):
            if data.get('input'):
                input_nodes.add(nid)
            elif not data.get('parents') and nid not in output_vertices:
                # No parents and not marked as output -> likely input
                input_nodes.add(nid)
    
    # If no input nodes found, treat nodes with no parents as inputs
    if not input_nodes:
        for nid, data in nodes.items():
            if isinstance(data, dict) and not data.get('parents'):
                input_nodes.add(nid)
    
    # State tracking
    # fast_memory: set of node ids currently in fast memory (red values)
    # slow_memory: set of node ids currently in slow memory (blue values)
    # operations: list of [op, vertex]
    
    # Strategy: Process nodes in topological order.
    # For each node:
    #   - If it's an input: it starts blue (in slow memory)
    #   - If it needs to be computed: ensure parents are available (load if needed)
    #   - Compute: result goes to fast memory
    #   - Manage fast memory: if full, store least-used value to slow memory
    #   - If node is output: ensure it ends up blue (store to slow memory at end)
    
    # Better strategy: Use a reference-counting approach.
    # Count how many children each node has (out-degree).
    # When a value is no longer needed by any unprocessed child, it can be evicted.
    
    out_degree = {nid: len(children.get(nid, [])) for nid in nodes}
    
    # Process in topological order
    operations = []
    fast_mem = set()  # node ids in fast memory
    slow_mem = set()  # node ids in slow memory (blue)
    
    # Inputs start blue
    for nid in input_nodes:
        slow_mem.add(nid)
    
    # Track remaining uses: how many more children need this node
    remaining_uses = dict(out_degree)
    
    # Process each node in topo order
    for nid in topo_order:
        if not check_time():
            break
        
        data = nodes.get(nid, {})
        if not isinstance(data, dict):
            continue
        
        parents = data.get('parents', [])
        is_input = nid in input_nodes
        is_output = nid in output_vertices
        
        if is_input:
            # Already in slow memory (blue)
            # If we need it for computation (as a parent of some node), it'll be loaded then
            continue
        
        # Need to compute this node
        # Ensure all parents are in fast memory
        for p in parents:
            if p not in fast_mem:
                # Need to load from slow memory
                if p in slow_mem:
                    operations.append(['load', p])
                    fast_mem.add(p)
                else:
                    # Parent not in slow memory? Shouldn't happen if topo order is correct
                    # Recompute? For now, try to load anyway
                    operations.append(['load', p])
                    fast_mem.add(p)
        
        # Compute
        operations.append(['compute', nid])
        fast_mem.add(nid)
        
        # Now check if fast memory is over budget
        # Evict values that are no longer needed
        while len(fast_mem) > fast_memory_budget:
            # Find a value that can be evicted
            # Prefer values with remaining_uses == 0
            evicted = None
            for candidate in list(fast_mem):
                if candidate == nid:
                    continue  # Don't evict the just-computed value
                if remaining_uses.get(candidate, 0) == 0:
                    evicted = candidate
                    break
            if evicted is None:
                # Evict any value except nid
                for candidate in list(fast_mem):
                    if candidate != nid:
                        evicted = candidate
                        break
            if evicted is not None:
                operations.append(['store', evicted])
                fast_mem.remove(evicted)
                slow_mem.add(evicted)
            else:
                break
        
        # Decrement remaining uses for parents
        for p in parents:
            remaining_uses[p] = remaining_uses.get(p, 0) - 1
        
        # If this node is an output, it must end up blue
        # We'll handle finalization at the end
    
    # Finalize: all outputs must be blue
    # Ensure all non-output nodes in fast memory are stored
    for nid in list(fast_mem):
        if nid not in output_vertices:
            operations.append(['store', nid])
            fast_mem.remove(nid)
            slow_mem.add(nid)
    
    # Ensure outputs are in slow memory (blue)
    for nid in output_vertices:
        if nid in fast_mem:
            operations.append(['store', nid])
            fast_mem.remove(nid)
            slow_mem.add(nid)
    
    # Check operation limit
    if len(operations) > 20000:
        # Truncate (shouldn't happen with reasonable DAG sizes)
        operations = operations[:20000]
    
    return operations


# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    import time
    
    START_TIME = time.time()
    DEADLINE = 85.0  # seconds, safety margin
    
    def check_time():
        return time.time() - START_TIME < DEADLINE
    
    # Parse DAG structure
    if isinstance(dag, dict):
        nodes = dag.get('nodes', {})
        output_vertices = set(dag.get('outputs', []))
        if not output_vertices:
            for nid, data in nodes.items():
                if isinstance(data, dict) and data.get('output'):
                    output_vertices.add(nid)
    else:
        nodes = {}
        output_vertices = set()
        if isinstance(dag, (list, tuple)):
            for i, item in enumerate(dag):
                if isinstance(item, dict):
                    nodes[i] = item
                    if item.get('output'):
                        output_vertices.add(i)
                else:
                    nodes[i] = {'parents': []}
        else:
            return []
    
    if not nodes:
        return []
    
    # Build children map
    children = {nid: [] for nid in nodes}
    for nid, data in nodes.items():
        if isinstance(data, dict):
            parents = data.get('parents', [])
            if parents:
                for p in parents:
                    if p in children:
                        children[p].append(nid)
    
    # Topological order
    if isinstance(dag, dict) and 'nodes' in dag:
        topo_order = list(nodes.keys())
    else:
        topo_order = sorted(nodes.keys())
    
    # Identify input nodes
    input_nodes = set()
    for nid, data in nodes.items():
        if isinstance(data, dict):
            if data.get('input'):
                input_nodes.add(nid)
            elif not data.get('parents') and nid not in output_vertices:
                input_nodes.add(nid)
    
    if not input_nodes:
        for nid, data in nodes.items():
            if isinstance(data, dict) and not data.get('parents'):
                input_nodes.add(nid)
    
    # Count out-degrees
    out_degree = {nid: len(children.get(nid, [])) for nid in nodes}
    
    operations = []
    fast_mem = set()
    slow_mem = set(input_nodes)
    remaining_uses = dict(out_degree)
    
    # Process in topological order
    for nid in topo_order:
        if not check_time():
            break
        
        data = nodes.get(nid, {})
        if not isinstance(data, dict):
            continue
        
        parents = data.get('parents', [])
        is_input = nid in input_nodes
        is_output = nid in output_vertices
        
        if is_input:
            continue
        
        # Load parents into fast memory
        for p in parents:
            if p not in fast_mem:
                if p in slow_mem:
                    operations.append(['load', p])
                    fast_mem.add(p)
                else:
                    operations.append(['load', p])
                    fast_mem.add(p)
        
        # Compute
        operations.append(['compute', nid])
        fast_mem.add(nid)
        
        # Evict if over budget
        while len(fast_mem) > fast_memory_budget:
            evicted = None
            for candidate in list(fast_mem):
                if candidate == nid:
                    continue
                if remaining_uses.get(candidate, 0) == 0:
                    evicted = candidate
                    break
            if evicted is None:
                for candidate in list(fast_mem):
                    if candidate != nid:
                        evicted = candidate
                        break
            if evicted is not None:
                operations.append(['store', evicted])
                fast_mem.remove(evicted)
                slow_mem.add(evicted)
            else:
                break
        
        for p in parents:
            remaining_uses[p] = remaining_uses.get(p, 0) - 1
    
    # Finalize
    for nid in list(fast_mem):
        if nid not in output_vertices:
            operations.append(['store', nid])
            fast_mem.remove(nid)
            slow_mem.add(nid)
    
    for nid in output_vertices:
        if nid in fast_mem:
            operations.append(['store', nid])
            fast_mem.remove(nid)
            slow_mem.add(nid)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
