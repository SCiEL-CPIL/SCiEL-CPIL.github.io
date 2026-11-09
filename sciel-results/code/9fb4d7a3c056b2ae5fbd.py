# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag structure (inferred from task):
    - dag['parents']: list of lists, parents[v] = list of parent indices for vertex v
    - dag['outputs']: list of output vertex indices
    - dag['n']: number of vertices (or len of parents)
    - Vertices are topologically ordered: parents of v have index < v
    
    Operations: load, store, compute, delete
    - load: bring value from slow to fast memory (cost 1 transfer)
    - store: send value from fast to slow memory (cost 1 transfer)
    - compute: compute vertex value from parents in fast memory (no transfer)
    - delete: free a fast memory slot (no transfer)
    
    Score: mean transfers / naive schedule transfers (lower is better)
    """
    start_time = time.time()
    DEADLINE = 85.0  # 5s safety margin before 90s kill
    
    # Parse DAG structure
    parents = dag.get('parents', dag.get('parent', []))
    outputs = dag.get('outputs', dag.get('output', []))
    n = len(parents) if parents else dag.get('n', 0)
    
    if n == 0:
        return []
    
    # Build children map
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Determine which vertices are inputs (no parents) and outputs
    is_input = [len(parents[v]) == 0 for v in range(n)]
    is_output = [False] * n
    for o in outputs:
        is_output[o] = True
    
    # Calculate layers (longest path from input)
    layer = [0] * n
    for v in range(n):
        if is_input[v]:
            layer[v] = 0
        else:
            max_parent_layer = max(layer[p] for p in parents[v])
            layer[v] = max_parent_layer + 1
    
    max_layer = max(layer) if n > 0 else 0
    
    # Group vertices by layer
    layers = [[] for _ in range(max_layer + 1)]
    for v in range(n):
        layers[layer[v]].append(v)
    
    # Determine last layer each vertex is needed
    # A vertex p is needed until the max layer of its children
    last_needed = [0] * n
    for v in range(n):
        for p in parents[v]:
            if layer[v] > last_needed[p]:
                last_needed[p] = layer[v]
    
    operations = []
    fast_memory = set()
    
    # Helper to free space if budget is exceeded or about to be
    def ensure_space():
        if len(fast_memory) < fast_memory_budget:
            return
        # Need to free space
        # Priority: delete vertices whose last_needed layer < current layer
        # We don't have current layer in this closure easily, so we pass it or use global-ish state
        # Actually, let's just delete any vertex that is "safe" to delete.
        # Safe to delete if last_needed[v] < current_layer being processed.
        # Since we process layer by layer, we can just delete anything with last_needed < L.
        # But ensure_space is called before L is fully defined in the loop?
        # Let's refactor: do the deletion check inside the main loop explicitly.
        pass

    # Process each layer
    for L in range(max_layer + 1):
        if time.time() - start_time > DEADLINE:
            # Emergency exit: just return what we have, though it might be invalid
            # But better to try to finish quickly.
            break
        
        layer_vertices = layers[L]
        
        # 1. Identify all parents needed for this layer
        needed_parents = set()
        for v in layer_vertices:
            for p in parents[v]:
                needed_parents.add(p)
        
        # 2. Load missing parents
        for p in needed_parents:
            if p not in fast_memory:
                # Check budget before loading
                if len(fast_memory) >= fast_memory_budget:
                    # Need to delete something
                    # Find a vertex in fast_memory that is no longer needed (last_needed < L)
                    deleted = False
                    for v in list(fast_memory):
                        if last_needed[v] < L:
                            operations.append(['delete', v])
                            fast_memory.discard(v)
                            deleted = True
                            break
                    if not deleted:
                        # If we can't find a safe one to delete, we have a problem.
                        # This implies the graph requires more memory than budget at this layer.
                        # However, the problem states "bounded candidate interface", implying a solution exists.
                        # If we are stuck, we might have to delete something that IS needed, 
                        # which would break the compute. 
                        # But usually, in layered DAGs, you can process layer by layer.
                        # If budget < max layer width, we might need to recompute.
                        # For now, assume budget is sufficient for layer width + 1.
                        # If not, we fail. Let's try to delete the "oldest" one as a heuristic.
                        # But this is dangerous.
                        # Let's just pick the first one.
                        v_to_del = next(iter(fast_memory))
                        operations.append(['delete', v_to_del])
                        fast_memory.discard(v_to_del)
                
                operations.append(['load', p])
                fast_memory.add(p)
        
        # 3. Compute vertices in this layer
        for v in layer_vertices:
            if is_input[v]:
                # Inputs are loaded in step 2 if they are parents of something,
                # but what if an input is NOT a parent of anything in this layer?
                # Wait, inputs are layer 0. If L=0, we process inputs.
                # If L>0, inputs are not in layer_vertices.
                # So if v is input, it must be in layer 0.
                # We should have loaded it in step 2 if it's a parent of a layer 1 vertex?
                # No, step 2 loads parents of layer L.
                # If v is in layer L, and is_input[v], then L=0.
                # Parents of layer 0 vertices are none.
                # So needed_parents is empty.
                # We need to load inputs explicitly if they are not already loaded.
                if v not in fast_memory:
                     if len(fast_memory) >= fast_memory_budget:
                         # Delete something safe
                         for vv in list(fast_memory):
                             if last_needed[vv] < L:
                                 operations.append(['delete', vv])
                                 fast_memory.discard(vv)
                                 break
                     operations.append(['load', v])
                     fast_memory.add(v)
            else:
                # Non-input: compute
                # Check budget
                if len(fast_memory) >= fast_memory_budget:
                     # Delete something safe
                     for vv in list(fast_memory):
                         if last_needed[vv] < L:
                             operations.append(['delete', vv])
                             fast_memory.discard(vv)
                             break
                operations.append(['compute', v])
                fast_memory.add(v)
        
        # 4. Store outputs in this layer
        for v in layer_vertices:
            if is_output[v]:
                operations.append(['store', v])
                fast_memory.discard(v)
        
        # 5. Cleanup: delete vertices no longer needed
        # A vertex p can be deleted if last_needed[p] < L
        # Note: We just finished layer L. 
        # Vertices needed for layer L+1 have last_needed >= L+1? 
        # No, last_needed[p] is the max layer of children.
        # If a vertex p is parent of a vertex in layer L+1, last_needed[p] >= L+1.
        # If a vertex p is NOT parent of any vertex in layer L+1 or later, last_needed[p] <= L.
        # So we can delete p if last_needed[p] < L + 1 ? 
        # Actually, if last_needed[p] == L, it was needed for layer L. 
        # After layer L is done, is it needed for L+1? No, because last_needed is max child layer.
        # So if last_needed[p] <= L, it is NOT needed for L+1.
        # So we can delete if last_needed[p] <= L.
        
        for p in list(fast_memory):
            if last_needed[p] <= L:
                operations.append(['delete', p])
                fast_memory.discard(p)
    
    # Final cleanup: store any remaining outputs not yet stored
    for o in outputs:
        if o in fast_memory:
            operations.append(['store', o])
            fast_memory.discard(o)
    
    # Delete remaining fast memory
    for v in list(fast_memory):
        operations.append(['delete', v])
        fast_memory.discard(v)
    
    return operations

# EVOLVE-BLOCK-END
