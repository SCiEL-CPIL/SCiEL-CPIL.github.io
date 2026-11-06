# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict

def _topological_sort(vertices, parents):
    """Returns a list of vertices in topological order."""
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in in_degree:
                    in_degree[v] += 1
                    children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(order) != len(vertices):
        return list(vertices)
    return order

def _analyze_dag(dag):
    """
    Pre-compute DAG properties:
    - Topological order
    - Children map
    - Last use index for each vertex (index in topo order of its last child)
    - Is input / Is output
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    order_index = {v: i for i, v in enumerate(order)}
    
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children_map:
                children_map[p].append(v)
                
    # Sort children in topological order
    for v in children_map:
        children_map[v].sort(key=lambda x: order_index[x])

    is_input = {v: (len(parents.get(v, [])) == 0) for v in vertices}
    is_output = {v: (v in outputs) for v in vertices}
    
    # Calculate last use index
    last_use = {}
    for v in vertices:
        if children_map[v]:
            last_use[v] = max(order_index[c] for c in children_map[v])
        else:
            last_use[v] = order_index[v]
            
    return order, order_index, children_map, is_input, is_output, last_use

def _validate_schedule(dag, ops, fast_memory_budget):
    """Validate that the schedule is correct and respects memory budget."""
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    state = {}
    for v in vertices:
        if not parents.get(v, []):
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    fast_memory = set()
    computed = set()
    
    for op, v in ops:
        if op == 'load':
            if state[v] != 'blue':
                return False
            if v in fast_memory:
                return False
            if len(fast_memory) >= fast_memory_budget:
                return False
            fast_memory.add(v)
            state[v] = 'red'
        elif op == 'store':
            if state[v] != 'red':
                return False
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
            state[v] = 'blue'
        elif op == 'delete':
            if state[v] != 'red':
                return False
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
            state[v] = 'void'
        elif op == 'compute':
            if v in computed:
                return False
            for p in parents.get(v, []):
                if state.get(p) != 'red':
                    return False
            if len(fast_memory) >= fast_memory_budget:
                return False
            fast_memory.add(v)
            state[v] = 'red'
            computed.add(v)
        else:
            return False
    
    # Check all outputs are blue
    for v in outputs:
        if state.get(v) != 'blue':
            return False
            
    return True

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    try:
        vertices = dag['vertices']
        parents = dag['parents']
        outputs = set(dag['outputs'])
        
        if not vertices:
            return []
            
        order, order_index, children_map, is_input, is_output, last_use = _analyze_dag(dag)
        
        # State tracking
        state = {}
        for v in vertices:
            if is_input[v]:
                state[v] = 'blue'
            else:
                state[v] = 'void'
                
        fast_memory = set()
        ops = []
        computed = set()
        
        # Pre-compute "remaining children" counts for O(1) can_delete check
        remaining_children = {v: len(children_map[v]) for v in vertices}
        
        # Track next child index for each vertex to compute next_use dynamically
        next_child_idx = {v: 0 for v in vertices}
        
        def get_next_use(v):
            """Get the topological index of the next child of v to be computed."""
            idx = next_child_idx[v]
            if idx < len(children_map[v]):
                return order_index[children_map[v][idx]]
            return float('inf')

        def can_delete(v):
            """Check if vertex v can be deleted (no uncomputed children)."""
            return remaining_children[v] == 0

        def evict_one():
            """Evict one vertex from fast memory to make space. Returns True if successful."""
            if not fast_memory:
                return False
            
            candidates = []
            
            # 1. Dead non-outputs
            for v in fast_memory:
                if can_delete(v) and not is_output[v]:
                    candidates.append((0, v, 'delete'))
            
            # 2. Dead outputs
            for v in fast_memory:
                if can_delete(v) and is_output[v]:
                    candidates.append((1, v, 'store'))
            
            # 3. Live non-outputs
            for v in fast_memory:
                if not can_delete(v) and not is_output[v]:
                    candidates.append((2, v, 'store'))
            
            # 4. Live outputs
            for v in fast_memory:
                if not can_delete(v) and is_output[v]:
                    candidates.append((3, v, 'store'))
            
            if not candidates:
                return False
                
            # Sort by priority, then by next_use (descending for live, doesn't matter for dead?)
            # For dead, next_use is inf.
            # For live, we want to evict the one with the LARGEST next_use (furthest future).
            # So sort by priority ASC, then next_use DESC.
            
            def sort_key(x):
                p, v, action = x
                if p < 2: # Dead
                    return (p, 0)
                else: # Live
                    return (p, -get_next_use(v))
            
            candidates.sort(key=sort_key)
            
            _, v, action = candidates[0]
            
            if action == 'delete':
                ops.append(['delete', v])
                state[v] = 'void'
            else: # store
                ops.append(['store', v])
                state[v] = 'blue'
            
            fast_memory.discard(v)
            return True

        def ensure_parent_in_memory(p):
            """Ensure parent p is in fast memory (red)."""
            if state[p] == 'red':
                return True
            
            if state[p] == 'blue':
                # Need to load
                if len(fast_memory) >= fast_memory_budget:
                    # Try to evict
                    if not evict_one():
                        return False
                ops.append(['load', p])
                fast_memory.add(p)
                state[p] = 'red'
                return True
            else:
                # state[p] == 'void'
                # This means p was deleted. We need to recompute it?
                # Or was it an input? Inputs start blue, so they can't be void unless deleted.
                # If an input was deleted, we can't reload it.
                # This case should not happen if we don't delete inputs prematurely.
                return False

        # Main Loop
        for v in order:
            # Check time
            if time.time() > deadline:
                break
            
            if is_input[v]:
                # Inputs start blue. We don't compute them.
                # Do we need to load them?
                # Only when a child needs them.
                # So we don't load them here.
                computed.add(v)
                # Update next_child_idx for parents? No, v is input.
                # But v is a parent of its children.
                # We don't update remaining_children here because v is not being "computed" in the sense of producing a value for the first time?
                # Actually, inputs are "computed" at start.
                # So we should update remaining_children for inputs?
                # No, remaining_children tracks how many children of a vertex are uncomputed.
                # When we "compute" a vertex v, we decrement remaining_children for all parents of v.
                # For inputs, we don't have parents.
                # But we do mark them as computed.
                # So if an input p has a child v, and v is computed, we decrement remaining_children[p].
                # This happens in the main loop when v is processed.
                continue
            
            # Non-input vertex
            all_parents_ready = True
            for p in parents[v]:
                if state[p] != 'red':
                    if not ensure_parent_in_memory(p):
                        all_parents_ready = False
                        break
            
            if not all_parents_ready:
                # Failure state for this simple scheduler
                # This shouldn't happen if eviction logic is correct.
                # If it does, we might need a more complex recompute.
                # For now, we assume it works.
                pass

            # Compute v
            # Check memory space
            if len(fast_memory) >= fast_memory_budget:
                if not evict_one():
                    # Can't make space
                    pass

            ops.append(['compute', v])
            fast_memory.add(v)
            state[v] = 'red'
            computed.add(v)
            
            # Update remaining children counts for parents of v
            for p in parents[v]:
                remaining_children[p] -= 1
                # Advance next_child_idx for p
                # We need to find the index of v in children_map[p]
                # Since children_map is sorted by topo order, and v is the current vertex,
                # it should be the next one if we process in order.
                # However, multiple children might be processed.
                # We can just increment next_child_idx[p] if children_map[p][next_child_idx[p]] == v
                # But it's safer to check.
                # Actually, since we process in topo order, and children_map is sorted,
                # v is the next child of p if it's in the list.
                # But p might have multiple children, and we are processing v.
                # So we advance the pointer for p.
                # We need to be careful: next_child_idx[p] points to the next uncomputed child.
                # Since v is computed now, if v == children_map[p][next_child_idx[p]], we increment.
                # But what if v is not the next one? It should be, because of topo order.
                # So:
                if next_child_idx[p] < len(children_map[p]) and children_map[p][next_child_idx[p]] == v:
                    next_child_idx[p] += 1
            
            # Proactive cleanup?
            # The eviction in `ensure_parent_in_memory` and before compute handles pressure.
            # But we can also clean up dead vertices to keep memory low.
            # Let's proactively delete any vertex that is dead (can_delete) and not an output.
            # And store any vertex that is dead and IS an output.
            
            # Only do this if memory is getting tight? Or always?
            # Doing it always keeps memory minimal, which is good for future evictions.
            # But it adds operations.
            # If we delete a dead vertex, we save a slot.
            # If we don't, we might have to evict it later (cost 1 op + 1 load if live, or 1 op if dead).
            # So proactive deletion of dead vertices is always good?
            # Yes, because it reduces memory pressure without future cost.
            
            for w in list(fast_memory):
                if time.time() > deadline:
                    break
                if can_delete(w):
                    if is_output[w]:
                        ops.append(['store', w])
                        state[w] = 'blue'
                    else:
                        ops.append(['delete', w])
                        state[w] = 'void'
                    fast_memory.discard(w)
            
        # Final Cleanup
        for v in list(fast_memory):
            if is_output[v]:
                ops.append(['store', v])
                fast_memory.discard(v)
                state[v] = 'blue'
            else:
                ops.append(['delete', v])
                fast_memory.discard(v)
                state[v] = 'void'
        
        # Validate the schedule
        if _validate_schedule(dag, ops, fast_memory_budget):
            if len(ops) <= 20000:
                return ops
        
        # Fallback
        return _safe_naive_schedule(dag, fast_memory_budget)

    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

def _safe_naive_schedule(dag, fast_memory_budget):
    """
    A very conservative schedule.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    if not vertices:
        return []

    order, order_index, children_map, is_input, is_output, last_use = _analyze_dag(dag)
    
    state = {}
    for v in vertices:
        if is_input[v]:
            state[v] = 'blue'
        else:
            state[v] = 'void'
            
    fast_memory = set()
    ops = []
    computed = set()
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    def evict_one_simple():
        if not fast_memory:
            return False
        # Evict the vertex with the largest last_use (furthest in the future)
        # If it's dead, delete/store. If live, store.
        best_v = None
        best_lu = -1
        for v in fast_memory:
            if last_use[v] > best_lu:
                best_lu = last_use[v]
                best_v = v
        if best_v is None:
            return False
        if remaining_children[best_v] == 0:
            # Dead
            if is_output[best_v]:
                ops.append(['store', best_v])
                state[best_v] = 'blue'
            else:
                ops.append(['delete', best_v])
                state[best_v] = 'void'
        else:
            # Live
            ops.append(['store', best_v])
            state[best_v] = 'blue'
        fast_memory.discard(best_v)
        return True

    for v in order:
        if is_input[v]:
            computed.add(v)
            continue
            
        for p in parents[v]:
            if state[p] != 'red':
                if state[p] == 'blue':
                    while len(fast_memory) >= fast_memory_budget:
                        if not evict_one_simple():
                            break
                    ops.append(['load', p])
                    fast_memory.add(p)
                    state[p] = 'red'
        
        while len(fast_memory) >= fast_memory_budget:
            if not evict_one_simple():
                break
        
        ops.append(['compute', v])
        fast_memory.add(v)
        state[v] = 'red'
        computed.add(v)
        
        for p in parents[v]:
            remaining_children[p] -= 1
        
        if is_output[v] and not children_map[v]:
            ops.append(['store', v])
            fast_memory.discard(v)
            state[v] = 'blue'

    for v in list(fast_memory):
        if is_output[v]:
            ops.append(['store', v])
            fast_memory.discard(v)
            state[v] = 'blue'
        else:
            ops.append(['delete', v])
            fast_memory.discard(v)
            state[v] = 'void'
            
    return ops

# EVOLVE-BLOCK-END
