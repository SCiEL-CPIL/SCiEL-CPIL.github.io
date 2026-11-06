"""DAG pebbling: minimize communication (load/store) operations."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize communication (load + store operations).
    Returns list of [operation, vertex].
    """
    start_time = time.time()
    DEADLINE = 55.0
    
    # Parse DAG
    if isinstance(dag, dict):
        n = dag.get('vertices', dag.get('n', dag.get('num_vertices', len(dag.get('parents', [])))))
        parents = dag.get('parents', [[] for _ in range(n)])
        children = dag.get('children', None)
        outputs = dag.get('outputs', [])
        layers = dag.get('layers', None)
    else:
        n = getattr(dag, 'n', getattr(dag, 'num_vertices', len(dag)))
        parents = getattr(dag, 'parents', [[] for _ in range(n)])
        children = getattr(dag, 'children', None)
        outputs = getattr(dag, 'outputs', [])
        layers = getattr(dag, 'layers', None)
    
    if children is None:
        children = [[] for _ in range(n)]
        for v in range(n):
            for p in parents[v]:
                children[p].append(v)
    
    if not outputs:
        outputs = [v for v in range(n) if not children[v]]
    
    # State: 0=blue, 1=red, 2=white
    state = [0] * n
    fast_memory = set()
    ops = []
    op_count = 0
    MAX_OPS = 20000
    
    def can_do_op():
        return op_count < MAX_OPS and (time.time() - start_time) < DEADLINE
    
    def do_op(op, v):
        nonlocal op_count
        ops.append([op, v])
        op_count += 1
    
    def load(v):
        if state[v] == 0:
            state[v] = 1
            fast_memory.add(v)
            do_op('load', v)
            return True
        return False
    
    def store(v):
        if state[v] == 1:
            state[v] = 0
            fast_memory.discard(v)
            do_op('store', v)
            return True
        return False
    
    def compute(v):
        for p in parents[v]:
            if state[p] != 1:
                return False
        state[v] = 1
        fast_memory.add(v)
        do_op('compute', v)
        return True
    
    def delete(v):
        if state[v] == 1:
            state[v] = 2
            fast_memory.discard(v)
            do_op('delete', v)
            return True
        return False
    
    def ensure_red(v, depth=0):
        """Ensure vertex v is red (in fast memory)."""
        if depth > 500:
            return False
        if state[v] == 1:
            return True
        if state[v] == 0:
            if len(fast_memory) >= fast_memory_budget:
                if not free_space():
                    return False
            return load(v)
        # state[v] == 2, recompute
        for p in parents[v]:
            if not ensure_red(p, depth + 1):
                return False
        return compute(v)
    
    def free_space():
        """Free fast memory space."""
        # Delete vertices that are no longer needed
        # Heuristic: delete vertices whose children are all computed
        for fm_v in list(fast_memory):
            all_children_done = True
            for c in children[fm_v]:
                if state[c] != 1:
                    all_children_done = False
                    break
            if all_children_done:
                delete(fm_v)
                return True
        # If nothing safe, delete an input (no parents)
        for fm_v in list(fast_memory):
            if not parents[fm_v]:
                delete(fm_v)
                return True
        # Last resort: delete first
        if fast_memory:
            delete(next(iter(fast_memory)))
            return True
        return False
    
    # Strategy: Topological order with smart memory management
    # Key: keep vertices with multiple children in fast memory
    
    for v in range(n):
        if not can_do_op():
            break
        
        # Ensure all parents are red
        for p in parents[v]:
            if not ensure_red(p):
                # Fallback: try harder
                if state[p] == 0:
                    free_space()
                    load(p)
                elif state[p] == 2:
                    # Recompute
                    for pp in parents[p]:
                        ensure_red(pp)
                    compute(p)
        
        # Compute v
        if state[v] != 1:
            compute(v)
        
        # Decide: keep or delete v
        num_children = len(children[v])
        if v in outputs:
            # Keep for final store
            pass
        elif num_children >= 2 and len(fast_memory) <= fast_memory_budget:
            # Keep in fast memory (multiple uses)
            pass
        elif num_children >= 2:
            # Want to keep but no space - free space
            free_space()
            # v is already red, it's in fast_memory
        else:
            # Single use or no use: delete
            delete(v)
    
    # Store all outputs
    for v in outputs:
        if state[v] == 1:
            store(v)
        elif state[v] == 2:
            ensure_red(v)
            if state[v] == 1:
                store(v)
    
    # Final safety
    for v in outputs:
        if state[v] == 1:
            store(v)
    
    return ops

# EVOLVE-BLOCK-END
