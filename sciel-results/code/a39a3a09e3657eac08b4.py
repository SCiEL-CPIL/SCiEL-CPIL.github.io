"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

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

def _build_graph_info(dag):
    """Pre-compute graph information for scheduling."""
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children_map:
                children_map[p].append(v)
    
    # Determine input vertices (no parents)
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    return order, children_map, outputs, inputs

def _improved_pebbling(dag, fast_memory_budget):
    """
    Improved pebbling strategy with recomputation support:
    1. Process vertices in topological order.
    2. Use a priority-based eviction strategy that avoids deleting needed vertices.
    3. Minimize transfers by keeping frequently-needed vertices in memory.
    4. Properly handle recomputation when memory is constrained.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order, children_map, outputs_set, inputs = _build_graph_info(dag)
    
    # Compute depth for priority
    depth = {v: 0 for v in vertices}
    for v in order:
        if parents.get(v, []):
            max_parent_depth = max(depth[p] for p in parents[v])
            depth[v] = max_parent_depth + 1
    
    # Compute "next use" position for each vertex (first child in topological order)
    # This helps us decide what to keep in memory
    next_use_pos = {v: len(order) for v in vertices}
    pos_in_order = {v: i for i, v in enumerate(order)}
    for v in vertices:
        for c in children_map[v]:
            if pos_in_order[c] < next_use_pos[v]:
                next_use_pos[v] = pos_in_order[c]
    
    # State tracking
    state = {}
    for v in vertices:
        if v in inputs:
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    fast_memory = set()
    ops = []
    
    # Track remaining children count for each vertex
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    def get_eviction_priority(v):
        """Get eviction priority for vertex v (lower = evict first)."""
        # Priority: 1) not an output, 2) later next use, 3) fewer remaining uses, 4) lower depth
        is_output = 1 if v in outputs else 0
        next_use = next_use_pos[v]
        remaining_uses = remaining_children[v]
        return (is_output, next_use, remaining_uses, depth[v])
    
    def evict_vertex(v):
        """Evict vertex v from fast memory."""
        if v not in fast_memory:
            return
        fast_memory.discard(v)
        if v in outputs:
            ops.append(['store', v])
            state[v] = 'blue'
        else:
            ops.append(['delete', v])
            state[v] = 'void'
    
    def free_space_if_needed():
        """Free space in fast memory if needed."""
        if len(fast_memory) < fast_memory_budget:
            return True
        
        # Get evictable vertices (those with no remaining children)
        evictable = [v for v in fast_memory if remaining_children[v] == 0]
        
        if not evictable:
            return len(fast_memory) < fast_memory_budget
        
        # Sort by eviction priority
        evictable.sort(key=get_eviction_priority)
        
        for v in evictable:
            evict_vertex(v)
            if len(fast_memory) < fast_memory_budget:
                return True
        
        return len(fast_memory) < fast_memory_budget
    
    def recompute_vertex(v):
        """
        Recompute vertex v if it was deleted. This requires recomputing all its ancestors
        that are also deleted. Returns True if successful.
        """
        # Find all ancestors that need to be recomputed
        to_recompute = set()
        stack = [v]
        while stack:
            curr = stack.pop()
            if curr in inputs:
                continue
            if state[curr] == 'void':
                to_recompute.add(curr)
                for p in parents.get(curr, []):
                    stack.append(p)
        
        if not to_recompute:
            return True
        
        # Recompute in topological order
        to_recompute_list = [v for v in order if v in to_recompute]
        
        for rv in to_recompute_list:
            # Ensure all parents are in memory
            for p in parents.get(rv, []):
                if state[p] != 'red':
                    if state[p] == 'blue':
                        # Load from slow memory
                        if not free_space_if_needed():
                            return False
                        ops.append(['load', p])
                        fast_memory.add(p)
                        state[p] = 'red'
                    elif state[p] == 'void':
                        # Should not happen if we're recomputing in topological order
                        return False
            
            # Compute rv
            if len(fast_memory) >= fast_memory_budget:
                if not free_space_if_needed():
                    return False
            
            ops.append(['compute', rv])
            fast_memory.add(rv)
            state[rv] = 'red'
            
            # Update remaining children count for parents
            for p in parents.get(rv, []):
                if p in remaining_children:
                    remaining_children[p] -= 1
        
        return True
    
    def ensure_parents_in_memory(v):
        """Ensure all parents of v are in fast memory. Returns True if successful."""
        for p in parents.get(v, []):
            if state[p] != 'red':
                if state[p] == 'blue':
                    # Load from slow memory
                    if not free_space_if_needed():
                        return False
                    ops.append(['load', p])
                    fast_memory.add(p)
                    state[p] = 'red'
                elif state[p] == 'void':
                    # Parent was deleted - we need to recompute it
                    if not recompute_vertex(p):
                        return False
        return True
    
    # Process vertices in topological order
    for v in order:
        if v in inputs:
            continue
        
        # Ensure parents are in memory
        if not ensure_parents_in_memory(v):
            # Can't compute this vertex; skip (shouldn't happen with good eviction)
            continue
        
        # Compute v
        if len(fast_memory) >= fast_memory_budget:
            if not free_space_if_needed():
                continue
        
        ops.append(['compute', v])
        fast_memory.add(v)
        state[v] = 'red'
        
        # Update remaining children count for parents
        for p in parents.get(v, []):
            if p in remaining_children:
                remaining_children[p] -= 1
        
        # If v is an output and has no more children, store it immediately
        if v in outputs and remaining_children[v] == 0:
            evict_vertex(v)
        
        # Free space if needed
        if len(fast_memory) >= fast_memory_budget:
            free_space_if_needed()
    
    # Final cleanup: store all remaining outputs, delete all other vertices
    for v in list(fast_memory):
        evict_vertex(v)
    
    return ops

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

def _count_transfers(ops):
    """Count the number of transfers (loads + stores) in the schedule."""
    transfers = 0
    for op, v in ops:
        if op in ('load', 'store'):
            transfers += 1
    return transfers

def _naive_schedule(dag, fast_memory_budget):
    """Simple naive schedule for comparison/fallback with recomputation support."""
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order, children_map, outputs_set, inputs = _build_graph_info(dag)
    
    # Compute depth for priority
    depth = {v: 0 for v in vertices}
    for v in order:
        if parents.get(v, []):
            max_parent_depth = max(depth[p] for p in parents[v])
            depth[v] = max_parent_depth + 1
    
    # Compute next use position
    next_use_pos = {v: len(order) for v in vertices}
    pos_in_order = {v: i for i, v in enumerate(order)}
    for v in vertices:
        for c in children_map[v]:
            if pos_in_order[c] < next_use_pos[v]:
                next_use_pos[v] = pos_in_order[c]
    
    state = {}
    for v in vertices:
        if v in inputs:
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    fast_memory = set()
    ops = []
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    def get_eviction_priority(v):
        """Get eviction priority for vertex v (lower = evict first)."""
        is_output = 1 if v in outputs else 0
        next_use = next_use_pos[v]
        remaining_uses = remaining_children[v]
        return (is_output, next_use, remaining_uses, depth[v])
    
    def evict_vertex(v):
        """Evict vertex v from fast memory."""
        if v not in fast_memory:
            return
        fast_memory.discard(v)
        if v in outputs:
            ops.append(['store', v])
            state[v] = 'blue'
        else:
            ops.append(['delete', v])
            state[v] = 'void'
    
    def free_space_if_needed():
        """Free space in fast memory if needed."""
        if len(fast_memory) < fast_memory_budget:
            return True
        
        evictable = [v for v in fast_memory if remaining_children[v] == 0]
        
        if not evictable:
            return len(fast_memory) < fast_memory_budget
        
        evictable.sort(key=get_eviction_priority)
        
        for v in evictable:
            evict_vertex(v)
            if len(fast_memory) < fast_memory_budget:
                return True
        
        return len(fast_memory) < fast_memory_budget
    
    def recompute_vertex(v):
        """Recompute vertex v if it was deleted."""
        to_recompute = set()
        stack = [v]
        while stack:
            curr = stack.pop()
            if curr in inputs:
                continue
            if state[curr] == 'void':
                to_recompute.add(curr)
                for p in parents.get(curr, []):
                    stack.append(p)
        
        if not to_recompute:
            return True
        
        to_recompute_list = [v for v in order if v in to_recompute]
        
        for rv in to_recompute_list:
            for p in parents.get(rv, []):
                if state[p] != 'red':
                    if state[p] == 'blue':
                        if not free_space_if_needed():
                            return False
                        ops.append(['load', p])
                        fast_memory.add(p)
                        state[p] = 'red'
                    elif state[p] == 'void':
                        return False
            
            if len(fast_memory) >= fast_memory_budget:
                if not free_space_if_needed():
                    return False
            
            ops.append(['compute', rv])
            fast_memory.add(rv)
            state[rv] = 'red'
            
            for p in parents.get(rv, []):
                if p in remaining_children:
                    remaining_children[p] -= 1
        
        return True
    
    # Process vertices in topological order
    for v in order:
        if v in inputs:
            continue
        
        # Ensure all parents are in memory
        parents_ok = True
        for p in parents.get(v, []):
            if state[p] != 'red':
                if state[p] == 'blue':
                    if not free_space_if_needed():
                        parents_ok = False
                        break
                    ops.append(['load', p])
                    fast_memory.add(p)
                    state[p] = 'red'
                elif state[p] == 'void':
                    if not recompute_vertex(p):
                        parents_ok = False
                        break
        
        if not parents_ok:
            continue
        
        # Compute v
        if len(fast_memory) >= fast_memory_budget:
            if not free_space_if_needed():
                continue
        
        ops.append(['compute', v])
        fast_memory.add(v)
        state[v] = 'red'
        
        # Update remaining children count for parents
        for p in parents.get(v, []):
            if p in remaining_children:
                remaining_children[p] -= 1
        
        # If v is an output and has no more children, store it immediately
        if v in outputs and remaining_children[v] == 0:
            evict_vertex(v)
        
        # Free space if needed
        if len(fast_memory) >= fast_memory_budget:
            free_space_if_needed()
    
    # Final cleanup
    for v in list(fast_memory):
        evict_vertex(v)
    
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    try:
        # Try the improved pebbling schedule first
        schedule = _improved_pebbling(dag, fast_memory_budget)
        
        # Validate it
        if _validate_schedule(dag, schedule, fast_memory_budget):
            if len(schedule) <= 20000:
                return schedule
        
        # Check time
        if time.time() > deadline - 10:
            # Not enough time for fallback, return what we have or empty
            return []
        
        # Fallback to naive schedule
        schedule = _naive_schedule(dag, fast_memory_budget)
        
        if _validate_schedule(dag, schedule, fast_memory_budget):
            if len(schedule) <= 20000:
                return schedule
        
        # Last resort: return an empty schedule
        return []
        
    except Exception as e:
        # Fallback to baseline if available
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

# EVOLVE-BLOCK-END
