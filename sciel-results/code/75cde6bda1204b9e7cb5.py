"""Pebbling algorithm for DAG pebbling with bounded fast memory."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize transfers.
    
    Args:
        dag: dict with 'parents' (list of lists), 'outputs' (list), 'inputs' (list or None)
        fast_memory_budget: int, max red values in fast memory
    
    Returns:
        list of [operation, vertex] where operation in {'load', 'store', 'compute', 'delete'}
    """
    start_time = time.time()
    DEADLINE = 60.0
    SAFETY = 5.0
    
    n = len(dag['parents'])
    parents = dag['parents']
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    if not inputs:
        inputs = {v for v in range(n) if len(parents[v]) == 0}
    
    # Build children map
    children = defaultdict(list)
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # State: fast_mem (set of vertices in fast memory), stored (set in slow mem)
    fast_mem = set()
    stored = set()
    ops = []
    
    def make_available(v):
        """Ensure vertex v is in fast memory. Returns True if it was already there."""
        if v in fast_mem:
            return True
        if v in stored:
            # Need to load
            if len(fast_mem) >= fast_memory_budget:
                # Evict something
                ev = evict()
                if ev is not None:
                    ops.append(['delete', ev])
                    fast_mem.discard(ev)
            ops.append(['load', v])
            fast_mem.add(v)
            return True
        # v not in fast_mem and not in stored - must compute
        # This shouldn't happen if we process in topo order, but handle it
        return False
    
    def evict():
        """Find a vertex to evict from fast memory."""
        # Prefer non-output vertices
        candidates = [v for v in fast_mem if v not in outputs]
        if not candidates:
            candidates = list(fast_mem)
        if not candidates:
            return None
        # Evict the one with fewest remaining children that need it
        def score(v):
            # Count children not yet computed/stored
            s = 0
            for c in children[v]:
                if c not in stored and c not in fast_mem:
                    s += 1
            return s
        candidates.sort(key=score)
        return candidates[0]
    
    # Process in topological order
    for v in range(n):
        if time.time() - start_time > DEADLINE - SAFETY:
            break
        
        # Ensure all parents are available
        for p in parents[v]:
            if not make_available(p):
                # Parent not available and not stored - need to compute it
                # This is a problem; should not happen in topo order
                pass
        
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # If output, store it
        if v in outputs:
            ops.append(['store', v])
            stored.add(v)
            # Can keep in fast_mem or delete; let's delete to save space
            if len(fast_mem) > fast_memory_budget:
                ev = evict()
                if ev is not None and ev != v:
                    ops.append(['delete', ev])
                    fast_mem.discard(ev)
    
    # Clean up any remaining fast memory
    while fast_mem and len(ops) < 20000:
        v = evict()
        if v is None:
            break
        ops.append(['delete', v])
        fast_mem.discard(v)
    
    return ops[:20000]

# EVOLVE-BLOCK-END
