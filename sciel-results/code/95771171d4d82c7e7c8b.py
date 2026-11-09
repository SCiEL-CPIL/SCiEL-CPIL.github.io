"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling with communication minimization.
    """
    start_time = time.time()
    DEADLINE = 85
    
    parents = dag['parents']
    n = len(parents)
    outputs = set(dag.get('outputs', []))
    
    children = defaultdict(list)
    for i, par_list in enumerate(parents):
        for p in par_list:
            children[p].append(i)
    
    remaining_uses = [len(children[i]) for i in range(n)]
    state = ['blue'] * n
    fast_mem = set()
    ops = []
    
    for i in range(n):
        if time.time() - start_time > DEADLINE or len(ops) >= 19995:
            break
        
        # Load parents if not in fast memory
        for p in parents[i]:
            if state[p] != 'red':
                # Need to load
                if len(fast_mem) >= fast_memory_budget:
                    # Need to evict something
                    # Find a value that's no longer needed or least useful
                    evicted = None
                    for v in list(fast_mem):
                        if remaining_uses[v] == 0 and v not in outputs:
                            evicted = v
                            break
                    if evicted is None:
                        # Evict the one with fewest remaining uses
                        evicted = min(fast_mem, key=lambda v: remaining_uses[v])
                    ops.append(['store', evicted])
                    state[evicted] = 'blue'
                    fast_mem.discard(evicted)
                
                ops.append(['load', p])
                state[p] = 'red'
                fast_mem.add(p)
        
        # Compute
        ops.append(['compute', i])
        state[i] = 'red'
        fast_mem.add(i)
        
        # Decrement uses of parents
        for p in parents[i]:
            remaining_uses[p] -= 1
        
        # If i has no more uses and isn't an output, consider freeing it
        if remaining_uses[i] == 0 and i not in outputs:
            if len(fast_mem) > fast_memory_budget:
                ops.append(['store', i])
                state[i] = 'blue'
                fast_mem.discard(i)
    
    # Store all outputs
    for o in outputs:
        if state[o] == 'red':
            ops.append(['store', o])
            state[o] = 'blue'
            fast_mem.discard(o)
    
    return ops[:20000]

# EVOLVE-BLOCK-END
