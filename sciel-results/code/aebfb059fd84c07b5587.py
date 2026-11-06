"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy: Recursive compute-and-delete.
    - Inputs are permanently blue (free).
    - Outputs must end blue.
    - Internal vertices: compute, use, delete.
    - Recompute as needed when memory is tight.
    """
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill
    
    # Parse DAG structure
    # dag format: assume dag has 'parents' (dict/list mapping vertex -> list of parent vertices)
    # and 'outputs' (list of output vertices)
    # Inputs are vertices with no parents or explicitly marked.
    
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    all_vertices = set(parents.keys()) | set(outputs)
    for p_list in parents.values():
        all_vertices.update(p_list)
    
    # Identify inputs: vertices that are not computed (no parents) or are leaf inputs
    # Actually, inputs are vertices with no parents in the DAG
    inputs = set()
    for v in all_vertices:
        if v not in parents or len(parents[v]) == 0:
            inputs.add(v)
    
    # Count how many times each vertex is needed as a parent
    need_count = {}
    for v in all_vertices:
        if v in parents:
            for p in parents[v]:
                need_count[p] = need_count.get(p, 0) + 1
    
    # Track remaining needs
    remaining_needs = dict(need_count)
    
    # Memory state
    in_memory = set(inputs)  # inputs start blue
    operations = []
    
    def elapsed():
        return time.time() - start_time
    
    def is_deadline():
        return elapsed() > DEADLINE
    
    def compute_vertex(v):
        """Ensure vertex v is in memory (blue). Returns list of operations."""
        if v in in_memory:
            return []
        if is_deadline():
            return None  # signal deadline
        
        ops = []
        
        # Ensure all parents are in memory
        if v in parents:
            for p in parents[v]:
                if p not in in_memory:
                    p_ops = compute_vertex(p)
                    if p_ops is None:
                        return None
                    ops.extend(p_ops)
        
        # Compute v
        ops.append(['compute', v])
        in_memory.add(v)
        
        # After computing v, we can delete parents that are no longer needed
        # and are not outputs
        if v in parents:
            for p in parents[v]:
                remaining_needs[p] = remaining_needs.get(p, 0) - 1
                if (remaining_needs.get(p, 0) == 0 and 
                    p not in outputs and 
                    p not in inputs):
                    ops.append(['delete', p])
                    in_memory.discard(p)
        
        return ops
    
    # Process all vertices in topological order
    # First, compute a topological ordering
    # Since parents are given, we can do a DFS-based topo sort
    topo_order = []
    visited = set()
    
    def topo_sort(v):
        if v in visited:
            return
        visited.add(v)
        if v in parents:
            for p in parents[v]:
                topo_sort(p)
        topo_order.append(v)
    
    for v in all_vertices:
        topo_sort(v)
    
    # Now process in topological order
    for v in topo_order:
        if v in in_memory:
            continue
        if is_deadline():
            break
        
        ops = compute_vertex(v)
        if ops is None:
            break
        operations.extend(ops)
    
    # Ensure all outputs are in memory
    for out in outputs:
        if out not in in_memory:
            if is_deadline():
                break
            ops = compute_vertex(out)
            if ops is None:
                break
            operations.extend(ops)
    
    # Cap operations at 20000
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
