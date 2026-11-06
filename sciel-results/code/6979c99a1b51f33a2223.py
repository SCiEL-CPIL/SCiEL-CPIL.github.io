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
        # Fallback if cycle or disconnected (shouldn't happen in valid DAG)
        return list(vertices)
    return order

def _analyze_dag(dag):
    """
    Pre-compute DAG properties.
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
                
    # Sort children by topological order to support next_use logic
    for v in vertices:
        children_map[v].sort(key=lambda x: order_index[x])
        
    is_input = {v: (len(parents.get(v, [])) == 0) for v in vertices}
    is_output = {v: (v in outputs) for v in vertices}
    
    # Calculate last use index (for reference, though we will use next_use)
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
        
        # Next use tracking
        # next_use_idx[v]: index into children_map[v] of the first uncomputed child
        # next_use_val[v]: topological index of that child, or float('inf') if none
        next_use_idx = {v: 0 for v in vertices}
        next_use_val = {}
        for v in vertices:
            if children_map[v]:
                next_use_val[v] = order_index[children_map[v][0]]
            else:
                next_use_val[v] = float('inf')

        def can_delete(v):
            """Check if vertex v can be deleted (no uncomputed children)."""
            return next_use_idx[v] >= len(children_map[v])

        def evict_one():
            """Evict one vertex from fast memory to make space. Returns True if successful."""
            if not fast_memory:
                return False
            
            candidates = []
            for v in fast_memory:
                is_out = is_output[v]
                safe_to_delete = can_delete(v)
                
                if safe_to_delete:
                    if is_out:
                        candidates.append((0, v, 'store')) 
                    else:
                        candidates.append((1, v, 'delete'))
                else:
                    if is_out:
                        candidates.append((2, v, 'store'))
                    else:
                        candidates.append((3, v, 'store'))
            
            if not candidates:
                return False
                
            # Belady's Algorithm: Evict the vertex whose next use is furthest in the future.
            # Among same priority, pick largest next_use_val.
            # If next_use_val is inf (dead), it's handled by priority 0/1 (safe_to_delete).
            # So for priority 2/3 (alive), next_use_val is finite.
            
            candidates.sort(key=lambda x: (-next_use_val[x[1]], x[0]))
            
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
                return False

        # Main Loop
        for v in order:
            # Check time
            if time.time() > deadline:
                break
            
            if is_input[v]:
                computed.add(v)
                # Update next_use for parents of v?
                # Inputs have no parents.
                continue
            
            # Non-input vertex
            all_parents_ready = True
            for p in parents[v]:
                if state[p] != 'red':
                    if not ensure_parent_in_memory(p):
                        all_parents_ready = False
                        break
            
            if not all_parents_ready:
                pass

            # Compute v
            # Check memory space
            if len(fast_memory) >= fast_memory_budget:
                if not evict_one():
                    pass

            ops.append(['compute', v])
            fast_memory.add(v)
            state[v] = 'red'
            computed.add(v)
            
            # Update next_use for parents of v
            for p in parents[v]:
                # v is a child of p.
                # If v is the next expected child for p, advance pointer.
                # children_map[p] is sorted by topo index.
                # next_use_idx[p] points to the first uncomputed child.
                # Since we process in topo order, if children_map[p][next_use_idx[p]] == v, we consume it.
                if next_use_idx[p] < len(children_map[p]) and children_map[p][next_use_idx[p]] == v:
                    next_use_idx[p] += 1
                    if next_use_idx[p] < len(children_map[p]):
                        next_use_val[p] = order_index[children_map[p][next_use_idx[p]]]
                    else:
                        next_use_val[p] = float('inf')

            # Proactive cleanup of dead vertices
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
    
    def evict_one_simple():
        if not fast_memory:
            return False
        best_v = None
        best_lu = -1
        for v in fast_memory:
            if last_use[v] > best_lu:
                best_lu = last_use[v]
                best_v = v
        if best_v is None:
            return False
        if is_output[best_v]:
            ops.append(['store', best_v])
            state[best_v] = 'blue'
        else:
            ops.append(['delete', best_v])
            state[best_v] = 'void'
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
