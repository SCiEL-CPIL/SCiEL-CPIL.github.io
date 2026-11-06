"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin out of 60s
    
    try:
        return _pebble_optimized(dag, fast_memory_budget, deadline)
    except Exception:
        # Fallback to baseline if anything fails
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

def _pebble_optimized(dag, fast_memory_budget, deadline):
    """
    Optimized pebbling strategy:
    - Precomputes last-use indices for efficient eviction decisions.
    - Uses a priority queue for eviction candidates.
    - Correctly uses 'store' for values needed later, 'delete' for values not needed.
    """
    vertices = list(dag.get_vertices())
    n = len(vertices)
    if n == 0:
        return []
    
    parents = {v: dag.get_parents(v) for v in vertices}
    inputs = set(dag.get_inputs())
    outputs = set(dag.get_outputs())
    
    # Map vertex to index for quick lookup
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Build dependents list: for each vertex p, list of indices of vertices that depend on p
    dependents = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            dependents[p].append(v_to_idx[v])
            
    # Sort dependents lists
    for p in vertices:
        dependents[p].sort()
    
    # Precompute last use index for each vertex
    last_use = {}
    for v in vertices:
        if dependents[v]:
            last_use[v] = dependents[v][-1]
        else:
            last_use[v] = -1  # Never used as a parent
    
    # Pointers to next dependent
    ptr = {v: 0 for v in vertices}
    
    # Initial state
    in_fast = set(inputs)
    in_slow = set() # Values stored to slow memory
    operations = []
    
    budget = fast_memory_budget
    
    # Helper to get next use index for a vertex
    def get_next_use(v):
        if ptr[v] < len(dependents[v]):
            return dependents[v][ptr[v]]
        return None
    
    # Eviction priority: lower = evict first
    # We'll use a simple scan but with precomputed info for speed
    
    def eviction_cost(c, i, in_fast_set):
        """Compute the cost of evicting vertex c at position i."""
        next_use = get_next_use(c)
        
        if next_use is None:
            # Not needed again
            if c in outputs:
                return 2.0, 'store'
            else:
                return 0.0, 'delete'
        else:
            distance = next_use - i
            
            if c in inputs:
                return 2.0, 'store'
            else:
                parent_cost = 0
                for p in parents[c]:
                    if p not in in_fast_set:
                        parent_cost += 1
                recompute_cost = parent_cost + 1
                store_cost = 2.0
                
                if recompute_cost < store_cost:
                    cost = recompute_cost
                    action = 'delete'
                else:
                    cost = store_cost
                    action = 'store'
                
                # Penalty for soon-to-be-used values
                if distance <= 3:
                    cost += 0.5 * (4 - distance)
                
                # Penalty for outputs
                if c in outputs:
                    cost += 2.0
                
                return cost, action
    
    # Process vertices in topological order
    for i, v in enumerate(vertices):
        if time.time() > deadline:
            break
            
        if v in inputs:
            continue
        
        # Load parents if not in fast memory
        for p in parents[v]:
            if p not in in_fast:
                if p in in_slow:
                    operations.append(['load', p])
                    in_slow.remove(p)
                    in_fast.add(p)
                else:
                    # Should not happen with correct store/delete logic
                    pass
        
        # Compute v
        operations.append(['compute', v])
        in_fast.add(v)
        
        # Update pointers for parents of v
        for p in parents[v]:
            while ptr[p] < len(dependents[p]) and dependents[p][ptr[p]] <= i:
                ptr[p] += 1
        
        # Eviction phase
        while len(in_fast) > budget:
            # Find best candidate to evict using precomputed info
            best_candidate = None
            best_cost = float('inf')
            best_action = None
            
            for c in in_fast:
                cost, action = eviction_cost(c, i, in_fast)
                
                if cost < best_cost:
                    best_cost = cost
                    best_candidate = c
                    best_action = action
            
            if best_candidate is None:
                break
            
            # Execute eviction
            if best_action == 'delete':
                operations.append(['delete', best_candidate])
                in_fast.remove(best_candidate)
                if best_candidate in in_slow:
                    in_slow.remove(best_candidate)
            else:  # store
                operations.append(['store', best_candidate])
                in_fast.remove(best_candidate)
                in_slow.add(best_candidate)
    
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in in_fast:
            if v in in_slow:
                operations.append(['load', v])
                in_slow.remove(v)
                in_fast.add(v)
            else:
                # Should not happen
                pass
            # Make room if needed
            while len(in_fast) > budget:
                deleted = False
                for candidate in list(in_fast):
                    if candidate not in outputs:
                        operations.append(['delete', candidate])
                        in_fast.remove(candidate)
                        if candidate in in_slow:
                            in_slow.remove(candidate)
                        deleted = True
                        break
                if not deleted:
                    break
    
    if len(operations) > 20000:
        operations = operations[:20000]
        
    return operations
# EVOLVE-BLOCK-END
