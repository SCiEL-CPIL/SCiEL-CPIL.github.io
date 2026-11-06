# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Minimize communication for DAG pebbling.
    
    dag: dict with keys:
        - 'vertices': list of vertex IDs (topologically ordered)
        - 'parents': dict mapping vertex -> list of parent vertices
        - 'inputs': list of input vertex IDs (start blue)
        - 'outputs': list of output vertex IDs (must finish blue)
    
    fast_memory_budget: number of fast memory slots available
    
    Returns: list of [operation, vertex] where operation is in {load, store, compute, delete}
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    if not vertices:
        return []
    
    # Build children mapping
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Precompute the next use index for each vertex to help with eviction
    # next_use[v] = index of the next vertex in topological order that requires v
    # If v is not used again, next_use[v] = infinity
    next_use = {}
    # We need to traverse in reverse topological order to compute "next use"
    # Or simply, for each vertex v, find the min index of its children that are not yet processed.
    # Since we process in topological order, we can maintain a pointer or precompute.
    
    # Precompute the list of children sorted by topological index (they are already in 'vertices' order)
    # Actually, 'vertices' is topologically ordered.
    # For each vertex v, its children are at indices > index(v).
    
    # Let's create an index map
    vertex_index = {v: i for i, v in enumerate(vertices)}
    
    # For each vertex, find the minimum topological index of its children
    # This helps in "Least Recently Used" or "Farthest Next Use" heuristics
    farthest_next_use = {}
    for v in vertices:
        if not children[v]:
            farthest_next_use[v] = float('inf')
        else:
            # Find the child with the largest index? No, we want to know when it's *next* needed.
            # Actually, for eviction, we want to evict the node that is needed *least soon* or *never*.
            # So we want to maximize the index of the next use.
            # next_use[v] = min( index(c) for c in children[v] )
            # If we evict a node with a large next_use index, we keep it for longer.
            # If we evict a node with a small next_use index, we will need to reload it soon.
            # So, to minimize reloads, we should evict the node with the MAXIMUM next_use index (farthest in the future).
            # Wait, if I have limited memory, I should keep the nodes I need *soonest*.
            # So I should evict the nodes I need *latest*.
            
            min_child_idx = min(vertex_index[c] for c in children[v])
            farthest_next_use[v] = min_child_idx
            
    # State
    in_memory = set()
    # Inputs start blue. But if len(inputs) > fast_memory_budget, we have a problem.
    # The contract says "Inputs start blue". This implies they are in fast memory.
    # If budget is too small, the problem might be unsolvable or we assume we can hold them.
    # Let's assume the test cases respect the budget for inputs or we must handle it.
    # If we can't hold all inputs, we might need to load/unload them.
    # But "start blue" is a strong constraint. Let's assume len(inputs) <= fast_memory_budget.
    
    # Initialize in_memory with inputs
    for i in inputs:
        in_memory.add(i)
        
    operations = []
    computed = set(inputs)
    
    # We need to be careful: if we add inputs to in_memory, do we need to "load" them?
    # No, they start blue.
    
    for v in vertices:
        if time.time() > deadline:
            # Emergency stop: return best effort? 
            # If we break, outputs might not be stored. 
            # Better to finish quickly or return empty if invalid.
            # Let's try to finish the current step if possible, otherwise break.
            break
            
        if v in inputs:
            # Input is already in memory (or should be). 
            # If it was evicted, we need to load it.
            if v not in in_memory:
                # Need to load v
                if len(in_memory) >= fast_memory_budget:
                    # Evict someone
                    # Find best candidate to evict
                    candidates = []
                    for x in in_memory:
                        # We don't want to evict v (we are loading it)
                        # We don't want to evict parents of v if they are needed right now? 
                        # Actually, if v is an input, it has no parents to compute.
                        # But other nodes might need x.
                        candidates.append((farthest_next_use.get(x, float('inf')), x))
                    if candidates:
                        candidates.sort(reverse=True) # Evict the one needed farthest in future
                        to_delete = candidates[0][1]
                        operations.append(['delete', to_delete])
                        in_memory.remove(to_delete)
                operations.append(['load', v])
                in_memory.add(v)
            continue
        
        # v is not an input. It needs to be computed.
        
        # 1. Ensure all parents are in memory
        for p in parents.get(v, []):
            if p not in in_memory:
                # Need to load p
                if len(in_memory) >= fast_memory_budget:
                    # Need to free space
                    # Evict the node with the farthest next use
                    candidates = []
                    for x in in_memory:
                        if x == p:
                            continue
                        # We are loading p, so we don't want to evict p (it's not in memory anyway)
                        # We want to evict x.
                        # Heuristic: Evict x with max farthest_next_use[x]
                        candidates.append((farthest_next_use.get(x, float('inf')), x))
                    
                    if candidates:
                        candidates.sort(reverse=True)
                        to_delete = candidates[0][1]
                        operations.append(['delete', to_delete])
                        in_memory.remove(to_delete)
                    else:
                        # Should not happen if budget >= 1 and p not in memory
                        pass
                
                operations.append(['load', p])
                in_memory.add(p)
        
        # 2. Compute v
        # Does compute require space? 
        # "every red value consumes one fast-memory slot, including newly computed results"
        # So if in_memory is full, we might need to delete something BEFORE computing?
        # Or does compute add to memory? Yes.
        # If len(in_memory) == fast_memory_budget, we need to free space for v?
        # Usually, compute writes to a register. If the register is full, we must evict.
        # Let's assume we need space for v.
        
        if len(in_memory) >= fast_memory_budget:
             # Evict someone to make room for v
             # But we just ensured parents are in memory. 
             # We can evict a parent if it's not needed again? 
             # Or evict an unrelated node.
             # Let's pick the one with farthest next use.
             candidates = []
             for x in in_memory:
                 # We must keep parents of v? No, we already used them.
                 # If a parent is not needed again, it's a good candidate.
                 candidates.append((farthest_next_use.get(x, float('inf')), x))
             
             if candidates:
                 candidates.sort(reverse=True)
                 to_delete = candidates[0][1]
                 operations.append(['delete', to_delete])
                 in_memory.remove(to_delete)
        
        operations.append(['compute', v])
        in_memory.add(v)
        computed.add(v)
        
        # 3. Optional: Delete v immediately if it's not an output and not needed again?
        # This saves space.
        # If v is not an output and farthest_next_use[v] == inf, delete it.
        if v not in outputs and farthest_next_use.get(v, float('inf')) == float('inf'):
             # It won't be used again. Delete it to free space.
             # But maybe we want to keep it if memory is not full? 
             # Deleting it reduces memory usage, which is good for avoiding future evictions.
             # However, "delete" is an operation. Does it cost? 
             # The score is "mean transfers". Delete is a transfer? 
             # Usually load/store/delete are transfers. Compute is not.
             # If delete is a transfer, we want to minimize total transfers.
             # Keeping it in memory costs 0 transfers. Deleting it costs 1.
             # But if we delete it, we might avoid an eviction (which is a delete) later?
             # No, if we delete it now, we free space. If we keep it, we might have to delete it later.
             # Total deletes might be similar.
             # However, if we keep it, and it's not used, it's just sitting there.
             # If memory is tight, keeping unused nodes forces evictions of useful nodes.
             # So, if a node is not used again, it's best to delete it immediately to maximize space for active nodes.
             # BUT, if we delete it, we pay a transfer cost.
             # If we don't delete it, we don't pay now, but might pay later if we need to evict it.
             # If we never need to evict it (because we have enough space), then deleting it is a waste.
             # So, only delete if we need to free space?
             # The code above already evicts when full.
             # So, let's NOT delete immediately unless it's an output? No, outputs must stay.
             # Let's rely on the eviction logic.
            
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in in_memory:
            # Need to load v
            if len(in_memory) >= fast_memory_budget:
                # Evict someone
                candidates = []
                for x in in_memory:
                    if x == v:
                        continue
                    candidates.append((farthest_next_use.get(x, float('inf')), x))
                if candidates:
                    candidates.sort(reverse=True)
                    to_delete = candidates[0][1]
                    operations.append(['delete', to_delete])
                    in_memory.remove(to_delete)
            
            operations.append(['load', v])
            in_memory.add(v)
            
    # Truncate to max 20000 operations
    if len(operations) > 20000:
        operations = operations[:20000]
        
    return operations
# EVOLVE-BLOCK-END
