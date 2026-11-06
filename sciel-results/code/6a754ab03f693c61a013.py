# EVOLVE-BLOCK-START
import time
from collections import OrderedDict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    # Robustly extract DAG components
    try:
        parents = dag['parents']
        outputs = set(dag['outputs'])
        inputs = set(dag['inputs'])
        num_vertices = len(parents)
    except Exception as e:
        # Fallback: Try to infer
        # If it fails, return empty list? No, must be valid.
        # Assume standard structure.
        raise e

    in_fast = OrderedDict()
    operations = []
    
    def load(v):
        operations.append(['load', v])
        if v in in_fast:
            in_fast.move_to_end(v)
        else:
            in_fast[v] = True
            
    def store(v):
        operations.append(['store', v])
        if v in in_fast:
            del in_fast[v]
            
    def compute(v):
        operations.append(['compute', v])
        if v in in_fast:
            in_fast.move_to_end(v)
        else:
            in_fast[v] = True
            
    # Process vertices in topological order
    # Assuming 0..N-1 is topo order
    for v in range(num_vertices):
        if v in inputs:
            if v not in in_fast:
                load(v)
        else:
            # Load parents if not in fast memory
            for p in parents[v]:
                if p not in in_fast:
                    load(p)
            
            compute(v)
            
            if v in outputs:
                store(v)
            
            # Evict if over budget
            while len(in_fast) > fast_memory_budget:
                # Evict LRU
                lru_v, _ = in_fast.popitem(last=False)
                store(lru_v)
                
    return operations
# EVOLVE-BLOCK-END
