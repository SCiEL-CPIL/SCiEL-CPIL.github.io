"""DAG pebbling with communication minimization for layered DAGs."""
import time

def pebble(dag, fast_memory_budget):
    """
    dag: dict with keys:
        'num_vertices': int
        'layers': list of lists of vertex indices per layer
        'parents': list of lists, parents[v] = list of parent vertex indices
        'outputs': list of output vertex indices
    fast_memory_budget: int
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85  # seconds
    
    num_vertices = dag['num_vertices']
    layers = dag['layers']
    parents = dag['parents']
    outputs_set = set(dag['outputs'])
    
    # Build children map
    children = [[] for _ in range(num_vertices)]
    for v in range(num_vertices):
        for p in parents[v]:
            children[p].append(v)
    
    # remaining_uses[v] = number of children of v that have not yet been processed
    remaining_uses = [len(children[v]) for v in range(num_vertices)]
    
    # State tracking
    in_fast = [False] * num_vertices
    computed = [False] * num_vertices
    fast_set = set()
    
    ops = []
    
    def try_add_to_fast(v):
        """Add v to fast memory, evicting if necessary. Returns True if successful."""
        nonlocal fast_set, in_fast, ops
        if v in fast_set:
            return True
        if len(fast_set) < fast_memory_budget:
            fast_set.add(v)
            in_fast[v] = True
            return True
        # Need to evict
        # Prefer evicting vertices with fewest remaining uses, not outputs, not v
        candidates = [c for c in fast_set if c != v and c not in outputs_set]
        if not candidates:
            # Must evict something, even an output (shouldn't happen if we're careful)
            candidates = [c for c in fast_set if c != v]
        if not candidates:
            return False
        best = min(candidates, key=lambda c: remaining_uses[c])
        ops.append(['store', best])
        in_fast[best] = False
        fast_set.discard(best)
        fast_set.add(v)
        in_fast[v] = True
        return True
    
    def evict_if_full(prefer_not=None):
        """Evict a vertex if fast memory is full. prefer_not is a set of vertices to avoid evicting."""
        nonlocal fast_set, in_fast, ops
        if len(fast_set) < fast_memory_budget:
            return
        candidates = [c for c in fast_set if c not in (prefer_not or set())]
        if not candidates:
            candidates = list(fast_set)
        if not candidates:
            return
        best = min(candidates, key=lambda c: remaining_uses[c])
        ops.append(['store', best])
        in_fast[best] = False
        fast_set.discard(best)
    
    # Process vertices in topological order
    all_vertices = []
    for layer in layers:
        all_vertices.extend(layer)
    
    for v in all_vertices:
        if time.time() - start_time > DEADLINE:
            break
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if not in_fast[p]:
                # Need to load p
                # First, make sure we have space
                if len(fast_set) >= fast_memory_budget:
                    evict_if_full(prefer_not=set(parents[v]))
                ops.append(['load', p])
                in_fast[p] = True
                fast_set.add(p)
        
        # Compute v
        ops.append(['compute', v])
        computed[v] = True
        
        # Add v to fast memory
        if not try_add_to_fast(v):
            # Should not happen, but if it does, we're in trouble
            pass
        
        # Update remaining uses for parents
        for p in parents[v]:
            remaining_uses[p] -= 1
            # If parent has no more uses and is not an output, delete it
            if remaining_uses[p] == 0 and p not in outputs_set and in_fast[p]:
                ops.append(['delete', p])
                in_fast[p] = False
                fast_set.discard(p)
    
    # Ensure all outputs are in fast memory
    for v in sorted(outputs_set):
        if time.time() - start_time > DEADLINE:
            break
        if not in_fast[v] and computed[v]:
            if len(fast_set) < fast_memory_budget:
                ops.append(['load', v])
                in_fast[v] = True
                fast_set.add(v)
            else:
                # Evict a non-output
                candidates = [c for c in fast_set if c not in outputs_set]
                if candidates:
                    best = min(candidates, key=lambda c: remaining_uses[c])
                    ops.append(['store', best])
                    in_fast[best] = False
                    fast_set.discard(best)
                    ops.append(['load', v])
                    in_fast[v] = True
                    fast_set.add(v)
                else:
                    # All fast memory is outputs, can't evict
                    # This is a problem, but we'll try anyway
                    ops.append(['load', v])
                    in_fast[v] = True
                    fast_set.add(v)
    
    return ops
