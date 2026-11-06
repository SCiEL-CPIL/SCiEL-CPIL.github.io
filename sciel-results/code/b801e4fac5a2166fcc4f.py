"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Pebble a layered DAG to minimize communication (transfers).
    
    Returns list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'.
    """
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # Parse the DAG structure - handle multiple possible formats
    if isinstance(dag, dict):
        layers = dag.get('layers', dag.get('layered_vertices', dag.get('vertex_layers', dag.get('strata', None))))
        parents = dag.get('parents', dag.get('parent_list', dag.get('predecessors', dag.get('deps', None))))
        inputs = set(dag.get('inputs', dag.get('input_vertices', [])))
        outputs = set(dag.get('outputs', dag.get('output_vertices', [])))
    else:
        layers = getattr(dag, 'layers', getattr(dag, 'layered_vertices', None))
        parents = getattr(dag, 'parents', getattr(dag, 'parent_list', None))
        inputs = set(getattr(dag, 'inputs', getattr(dag, 'input_vertices', [])))
        outputs = set(getattr(dag, 'outputs', getattr(dag, 'output_vertices', [])))
    
    if layers is None:
        if isinstance(dag, dict) and 'vertices' in dag:
            layers = [list(dag['vertices'])]
        elif hasattr(dag, 'vertices'):
            layers = [list(dag.vertices)]
        else:
            layers = []
    
    if parents is None:
        parent_map = defaultdict(list)
    elif isinstance(parents, dict):
        parent_map = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_map = {i: list(parents[i]) for i in range(len(parents))}
    else:
        parent_map = defaultdict(list)
    
    # Build layer index for each vertex
    all_vertices = set()
    for layer in layers:
        for v in layer:
            all_vertices.add(v)
    
    vertex_layer = {}
    for li, layer in enumerate(layers):
        for v in layer:
            vertex_layer[v] = li
    
    # Count how many children each vertex has
    child_count = defaultdict(int)
    for v in all_vertices:
        for p in parent_map.get(v, []):
            child_count[p] += 1
    
    # Determine inputs and outputs if not specified
    if not inputs:
        inputs = set(v for v in all_vertices if not parent_map.get(v, []))
    if not outputs:
        outputs = set(v for v in all_vertices if child_count[v] == 0)
    
    # Track state
    in_fast_memory = set()  # vertices currently red (in fast memory)
    computed = set()  # vertices that have been computed at least once
    stored = set()  # vertices that have been stored to slow memory
    
    operations = []
    
    def load_vertex(v):
        operations.append(['load', v])
        in_fast_memory.add(v)
    
    def store_vertex(v):
        operations.append(['store', v])
        in_fast_memory.discard(v)
        stored.add(v)
    
    def compute_vertex(v):
        operations.append(['compute', v])
        in_fast_memory.add(v)
        computed.add(v)
    
    def delete_vertex(v):
        operations.append(['delete', v])
        in_fast_memory.discard(v)
    
    def is_deadline():
        return (time.time() - start_time) > DEADLINE
    
    # For each vertex, determine how many times it's used as a parent
    # by vertices in later layers
    future_usage = defaultdict(int)
    for v in all_vertices:
        for p in parent_map.get(v, []):
            future_usage[p] += 1
    
    # Process each layer
    for li, layer in enumerate(layers):
        if is_deadline():
            break
        
        # First, load any inputs in this layer
        for v in layer:
            if is_deadline():
                break
            if v in inputs:
                if v not in in_fast_memory:
                    load_vertex(v)
                continue
        
        # Now compute non-input vertices in this layer
        for v in layer:
            if is_deadline():
                break
            if v in inputs:
                continue
            
            # Ensure all parents are in fast memory
            par = parent_map.get(v, [])
            for p in par:
                if p not in in_fast_memory:
                    # Check if parent was computed before
                    if p in computed or p in inputs:
                        load_vertex(p)
                    else:
                        # Parent not computed yet - skip (shouldn't happen in topological order)
                        pass
            
            # Check memory budget and evict if needed
            while len(in_fast_memory) >= fast_memory_budget:
                # Find a vertex to evict: prefer those with no future uses
                evicted = False
                for p in list(in_fast_memory):
                    if p in inputs:
                        continue
                    # Check if p is needed by any uncomputed vertex
                    needed = False
                    for v2 in all_vertices:
                        if v2 not in computed and v2 not in inputs:
                            if p in parent_map.get(v2, []):
                                needed = True
                                break
                    if not needed:
                        delete_vertex(p)
                        evicted = True
                        break
                if not evicted:
                    # Evict the least useful vertex
                    break
            
            # Compute v
            compute_vertex(v)
        
        # After layer, store vertices not needed in future
        if li < len(layers) - 1:
            for v in layer:
                if is_deadline():
                    break
                if v in in_fast_memory and v not in inputs:
                    # Check if needed in future layers
                    future_need = False
                    for v2 in all_vertices:
                        if vertex_layer.get(v2, 0) > li and v in parent_map.get(v2, []):
                            future_need = True
                            break
                    if not future_need:
                        store_vertex(v)
    
    # Final: store all outputs and inputs to make them blue
    for v in outputs:
        if v in in_fast_memory:
            store_vertex(v)
    for v in inputs:
        if v in in_fast_memory:
            store_vertex(v)
    
    # Trim to max 20000 operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
