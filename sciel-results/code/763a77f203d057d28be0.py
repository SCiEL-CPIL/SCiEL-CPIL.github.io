"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    inputs = set(dag.get('inputs', []))
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Identify input nodes
    input_nodes = set()
    for v in vertices:
        if not parents.get(v, []):
            input_nodes.add(v)
    input_nodes |= inputs
    
    # Assign layers
    layer = {}
    for v in vertices:
        if v in input_nodes:
            layer[v] = 0
        else:
            layer[v] = 1 + max(layer[p] for p in parents[v])
    
    max_layer = max(layer.values()) if layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in vertices:
        layers[layer[v]].append(v)
    
    operations = []
    fast = set()
    
    def op(o, v):
        operations.append([o, v])
    
    def load(v):
        op('load', v)
        fast.add(v)
    
    def store(v):
        op('store', v)
        fast.discard(v)
    
    def delete(v):
        op('delete', v)
        fast.discard(v)
    
    def compute(v):
        op('compute', v)
        fast.add(v)
    
    # Process each layer
    for i in range(max_layer + 1):
        # Ensure all parents of vertices in this layer are in fast memory
        # Parents are in layer i-1
        
        # Load inputs in this layer
        for v in layers[i]:
            if v in input_nodes and v not in fast:
                if len(fast) < fast_memory_budget:
                    load(v)
                else:
                    # Evict non-output vertices from earlier layers
                    for u in list(fast):
                        if u not in outputs and layer[u] < i:
                            delete(u)
                            if len(fast) < fast_memory_budget:
                                break
                    if v not in fast:
                        load(v)
        
        # Compute non-input vertices in this layer
        for v in layers[i]:
            if v in input_nodes:
                continue
            # Ensure parents are in fast memory
            for p in parents[v]:
                if p not in fast:
                    if len(fast) < fast_memory_budget:
                        load(p)
                    else:
                        # Evict
                        for u in list(fast):
                            if u not in outputs:
                                delete(u)
                                if len(fast) < fast_memory_budget:
                                    break
                        if p not in fast:
                            load(p)
            
            if len(fast) >= fast_memory_budget:
                for u in list(fast):
                    if u not in outputs and u != v:
                        delete(u)
                        if len(fast) < fast_memory_budget:
                            break
            compute(v)
        
        # Clean up: store outputs, delete non-outputs that are no longer needed
        for v in layers[i]:
            if v in fast:
                if v in outputs:
                    store(v)
                else:
                    # Delete if no children in future layers
                    if not any(layer[c] > i for c in children[v]):
                        delete(v)
    
    return operations
# EVOLVE-BLOCK-END
