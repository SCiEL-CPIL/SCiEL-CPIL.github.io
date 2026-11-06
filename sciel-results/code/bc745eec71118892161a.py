"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag: dict with keys:
        - 'n': number of vertices
        - 'parents': list of lists, parents[i] = list of parent vertex indices
        - 'outputs': list of output vertex indices
    fast_memory_budget: int, number of red pebbles (fast memory slots)
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    deadline = time.time() + 50  # safety margin
    
    n = dag['n']
    parents = dag['parents']
    outputs = dag['outputs']
    budget = fast_memory_budget
    
    # Build children mapping
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Identify inputs (vertices with no parents)
    inputs = [v for v in range(n) if len(parents[v]) == 0]
    
    # Topological order (Kahn's algorithm)
    in_degree = [len(parents[v]) for v in range(n)]
    queue = [v for v in range(n) if in_degree[v] == 0]
    topo_order = []
    q_idx = 0
    while q_idx < len(queue):
        v = queue[q_idx]
        q_idx += 1
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != n:
        # Cycle detected, fallback to naive
        return _naive_schedule(dag, fast_memory_budget)
    
    # State: 
    #   -1: unpebbled (not in fast or slow memory)
    #    0: in slow memory (blue)
    #    1: in fast memory (red)
    state = [-1] * n
    
    # All inputs start blue (in slow memory)
    for v in inputs:
        state[v] = 0
    
    # Operations list
    ops = []
    
    # Track which vertices are in fast memory
    fast_set = set()
    
    # For each vertex in topological order, we need to compute it
    # We need its parents to be available (in fast or slow memory)
    
    def load(v):
        """Load vertex v from slow to fast memory"""
        if state[v] == 0:
            ops.append(['load', v])
            state[v] = 1
            fast_set.add(v)
            return True
        elif state[v] == 1:
            return True
        else:
            return False
    
    def store(v):
        """Store vertex v from fast to slow memory"""
        if state[v] == 1:
            ops.append(['store', v])
            state[v] = 0
            fast_set.discard(v)
            return True
        return False
    
    def delete(v):
        """Delete vertex v from fast memory (not saved to slow)"""
        if state[v] == 1:
            ops.append(['delete', v])
            state[v] = -1
            fast_set.discard(v)
            return True
        return False
    
    def compute(v):
        """Compute vertex v (requires all parents in fast memory)"""
        ops.append(['compute', v])
        state[v] = 1
        fast_set.add(v)
    
    # Greedy pebbling strategy:
    # Process vertices in topological order.
    # Before computing a vertex, ensure all its parents are in fast memory.
    # If fast memory is full, decide which vertex to evict (store or delete).
    
    # For each vertex, count how many times it's needed as a parent
    use_count = [0] * n
    for v in range(n):
        for p in parents[v]:
            use_count[p] += 1
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        # If this is an input, it's already in slow memory
        if len(parents[v]) == 0:
            # Input: it's in slow memory (state 0). Load it if we need it in fast.
            # We'll load it when we need to compute a child, or if it's an output.
            # For now, leave it in slow memory.
            continue
        
        # Need to compute v. First, ensure all parents are in fast memory.
        for p in parents[v]:
            if state[p] != 1:
                # Need to load p into fast memory
                if len(fast_set) >= budget:
                    # Need to evict something
                    # Find a vertex in fast memory that can be evicted
                    # Prefer to evict vertices with low future use
                    evicted = False
                    # Try to find a vertex to evict
                    candidates = list(fast_set)
                    # Sort by use_count (ascending) - evict least used first
                    candidates.sort(key=lambda x: use_count[x])
                    for cand in candidates:
                        # Check if cand is still needed as a parent for some future vertex
                        # Actually, we can always evict - if needed later, we reload
                        if cand == v:
                            continue
                        # Evict cand
                        # If cand is an output, we must store it (it needs to finish blue)
                        if cand in outputs:
                            store(cand)
                        else:
                            # Decide: store or delete?
                            # If use_count[cand] > 0, it might be needed again
                            # But we can always reload, so delete is cheaper if we don't need it much
                            # Heuristic: store if use_count is high, delete otherwise
                            if use_count[cand] > 1:
                                store(cand)
                            else:
                                delete(cand)
                        evicted = True
                        break
                    
                    if not evicted:
                        # Fallback: store any vertex
                        for cand in list(fast_set):
                            if cand == v:
                                continue
                            store(cand)
                            evicted = True
                            break
                    
                    if not evicted:
                        # Can't evict, this shouldn't happen
                        break
                
                load(p)
        
        # Now compute v
        if len(fast_set) >= budget:
            # Need to make room
            # Evict a vertex that is not v and not a parent of v
            candidates = [x for x in fast_set if x != v and x not in parents[v]]
            if candidates:
                candidates.sort(key=lambda x: use_count[x])
                for cand in candidates:
                    if cand in outputs:
                        store(cand)
                    else:
                        if use_count[cand] > 1:
                            store(cand)
                        else:
                            delete(cand)
                    break
            else:
                # All fast memory is parents of v, we can't evict them
                # This is a problem - we need to compute v but no room
                # We'll have to store a parent temporarily
                for p in parents[v]:
                    if p in fast_set:
                        store(p)
                        break
        
        compute(v)
        
        # After computing v, if v is an output, we need it to finish blue
        # We can store it now or later
        # For now, keep it in fast memory; we'll handle outputs at the end
    
    # Ensure all outputs finish blue
    for v in outputs:
        if state[v] == 1:
            store(v)
        elif state[v] == -1:
            # Output was deleted, need to recompute? 
            # This shouldn't happen if we handled outputs properly
            # For safety, we can't easily recompute, so this is a problem
            pass
        # state[v] == 0 is fine (already blue)
    
    # Check if we have too many operations
    if len(ops) > 20000:
        # Fallback to naive
        return _naive_schedule(dag, fast_memory_budget)
    
    return ops


def _naive_schedule(dag, fast_memory_budget):
    """Naive baseline: load each input, compute each vertex, store each output."""
    n = dag['n']
    parents = dag['parents']
    outputs = dag['outputs']
    
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Topological order
    in_degree = [len(parents[v]) for v in range(n)]
    queue = [v for v in range(n) if in_degree[v] == 0]
    topo_order = []
    q_idx = 0
    while q_idx < len(queue):
        v = queue[q_idx]
        q_idx += 1
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    ops = []
    # Load all inputs
    for v in topo_order:
        if len(parents[v]) == 0:
            ops.append(['load', v])
        else:
            ops.append(['compute', v])
    
    # Store all outputs
    for v in outputs:
        ops.append(['store', v])
    
    return ops
# EVOLVE-BLOCK-END
