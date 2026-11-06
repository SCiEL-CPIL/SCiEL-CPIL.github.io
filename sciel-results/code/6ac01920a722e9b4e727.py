"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Find vertices that can reach an output (needed vertices)
    needed = set(outputs)
    changed = True
    while changed:
        changed = False
        for v in list(needed):
            for p in parents.get(v, []):
                if p not in needed:
                    needed.add(p)
                    changed = True
    
    # Topological order of needed vertices
    v_index = {v: i for i, v in enumerate(vertices)}
    needed_sorted = sorted(needed, key=lambda v: v_index[v])
    
    # Which inputs do we actually need to load?
    # An input needs to be loaded if it has a child in the needed set
    # (i.e., it's used to compute something)
    inputs_to_load = set()
    for v in inputs & needed:
        if any(c in needed for c in children.get(v, [])):
            inputs_to_load.add(v)
    
    fast_memory = set()
    ops = []
    computed = set()
    
    # Remaining needed children count
    rem_children = {v: len([c for c in children.get(v, []) if c in needed]) for v in needed}
    
    def free_space():
        nonlocal fast_memory, ops
        while len(fast_memory) >= fast_memory_budget:
            # Delete a vertex with no remaining needed children
            freed = False
            for fm_v in list(fast_memory):
                if fm_v in computed and rem_children.get(fm_v, 0) == 0:
                    ops.append(['delete', fm_v])
                    fast_memory.discard(fm_v)
                    freed = True
                    break
            if not freed:
                for fm_v in list(fast_memory):
                    if fm_v not in outputs:
                        ops.append(['delete', fm_v])
                        fast_memory.discard(fm_v)
                        break
            if len(fast_memory) < fast_memory_budget:
                break
            # Emergency
            if fast_memory:
                fm_v = next(iter(fast_memory))
                ops.append(['delete', fm_v])
                fast_memory.discard(fm_v)
            else:
                break
    
    for v in needed_sorted:
        if time.time() > deadline:
            break
        
        if v in inputs:
            if v in inputs_to_load and v not in fast_memory:
                free_space()
                ops.append(['load', v])
                fast_memory.add(v)
            computed.add(v)
        else:
            # Load parents
            for p in parents.get(v, []):
                if p in needed and p not in fast_memory:
                    free_space()
                    ops.append(['load', p])
                    fast_memory.add(p)
            
            ops.append(['compute', v])
            fast_memory.add(v)
            computed.add(v)
            
            for p in parents.get(v, []):
                if p in rem_children:
                    rem_children[p] -= 1
    
    # Store outputs
    for v in outputs:
        if time.time() > deadline:
            break
        if v not in fast_memory:
            free_space()
            ops.append(['load', v])
            fast_memory.add(v)
        ops.append(['store', v])
        ops.append(['delete', v])
        fast_memory.discard(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
