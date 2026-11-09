"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 80.0
    
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
    
    # Determine layers
    layer = {}
    for v in vertices:
        if v in inputs:
            layer[v] = 0
        else:
            max_pl = -1
            for p in parents[v]:
                if p in layer:
                    max_pl = max(max_pl, layer[p])
            layer[v] = max_pl + 1
    
    max_layer = max(layer.values()) if layer else 0
    
    layers = {}
    for v in vertices:
        l = layer[v]
        if l not in layers:
            layers[l] = []
        layers[l].append(v)
    
    fast_memory = set()
    slow_memory = set(inputs)
    
    operations = []
    
    def do_load(v):
        operations.append(['load', v])
        slow_memory.discard(v)
        fast_memory.add(v)
    
    def do_store(v):
        operations.append(['store', v])
        fast_memory.discard(v)
        slow_memory.add(v)
    
    def do_compute(v):
        operations.append(['compute', v])
        fast_memory.add(v)
    
    def do_delete(v):
        operations.append(['delete', v])
        fast_memory.discard(v)
    
    for l in range(max_layer + 1):
        if time.time() > deadline:
            break
        if l not in layers:
            continue
        
        layer_vertices = layers[l]
        
        # Load parents that are in slow memory
        for v in layer_vertices:
            if time.time() > deadline:
                break
            for p in parents[v]:
                if p not in fast_memory and p in slow_memory:
                    do_load(p)
        
        # Compute vertices
        for v in layer_vertices:
            if time.time() > deadline:
                break
            do_compute(v)
        
        # Manage fast memory after this layer
        if l == max_layer:
            for v in layer_vertices:
                if v in outputs and v in fast_memory:
                    do_store(v)
        else:
            # Determine which fast memory values are needed in future
            needed_in_future = set()
            for v in fast_memory:
                for c in children[v]:
                    if layer[c] > l:
                        needed_in_future.add(v)
                        break
            
            # Values not needed in future: delete them
            # Values needed in future: keep if possible, else store
            if len(needed_in_future) <= fast_memory_budget:
                # Can keep all needed values
                for v in list(fast_memory):
                    if v not in needed_in_future:
                        do_delete(v)
            else:
                # Can't keep all; prioritize by future use count
                future_use = {}
                for v in needed_in_future:
                    count = sum(1 for c in children[v] if layer[c] > l)
                    future_use[v] = count
                
                sorted_needed = sorted(needed_in_future, key=lambda x: future_use.get(x, 0), reverse=True)
                to_keep = set(sorted_needed[:fast_memory_budget])
                
                for v in list(fast_memory):
                    if v in to_keep:
                        continue
                    elif v in needed_in_future:
                        do_store(v)  # Store so we can reload later
                    else:
                        do_delete(v)
    
    # Final: ensure outputs are in slow memory
    for v in outputs:
        if v in fast_memory:
            do_store(v)
    
    return operations
# EVOLVE-BLOCK-END
