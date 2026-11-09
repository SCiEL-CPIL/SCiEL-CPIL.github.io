"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using topological greedy recompute.
    """
    start_time = time.time()
    DEADLINE = 85  # seconds
    
    n = dag['n']
    parents = dag['parents']
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    # Build children map
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    # remaining[p] = number of children of p not yet processed
    remaining = [len(children[p]) for p in range(n)]
    
    in_memory = set(inputs)  # inputs start in fast memory
    ops = []
    op_count = 0
    MAX_OPS = 20000
    
    def make_room():
        """Free memory if needed by deleting vertices that are no longer needed."""
        nonlocal ops, op_count, in_memory
        while len(in_memory) >= fast_memory_budget:
            candidate = None
            for v in in_memory:
                if v not in outputs and remaining[v] == 0:
                    candidate = v
                    break
            if candidate is None:
                return False
            ops.append(['delete', candidate])
            op_count += 1
            in_memory.discard(candidate)
        return True
    
    for i in range(n):
        if op_count >= MAX_OPS:
            break
        if time.time() - start_time > DEADLINE:
            break
        
        if i in in_memory:
            # Already in memory, nothing to do
            continue
        
        # Need to compute i. Ensure all parents are in memory.
        for p in parents[i]:
            if p not in in_memory:
                if not make_room():
                    break
                ops.append(['load', p])
                op_count += 1
                in_memory.add(p)
        
        # Compute i
        ops.append(['compute', i])
        op_count += 1
        in_memory.add(i)
        
        # After computing i, decrement remaining count for its parents
        # because i is now processed (one child of each parent is done)
        for p in parents[i]:
            remaining[p] -= 1
    
    return ops

# EVOLVE-BLOCK-END
