# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _topo_sort(vertices, parents_map):
    """Compute topological order given vertices and parents."""
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    return order

def _greedy_pebble_improved(dag, fast_memory_budget, deadline):
    """
    Improved greedy pebbling with better eviction policy.
    """
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
        
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    input_set = set(input_vertices)
    output_set = set(outputs)
    
    # Build children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
            
    # remaining_usage: how many children still need this vertex
    remaining_usage = {v: len(children_map[v]) for v in vertices}
    
    order = _topo_sort(vertices, parents_map)
    order_index = {v: i for i, v in enumerate(order)}
    
    # Fast memory state
    fast_mem = set()
    ops = []
    state = {v: 'in_slow' if v in input_set else 'gone' for v in vertices}
    
    # Pre-compute next use index for each vertex
    # next_use[v] = the order_index of the first child of v that hasn't been processed yet
    # We'll update this as we go
    
    def get_next_use(v):
        """Get the order index of the next child of v that needs v."""
        for c in children_map[v]:
            # If child hasn't been processed yet, this is a future use
            # We need to track which children have been processed
            pass
        return float('inf')
    
    # Track processed vertices
    processed = set()
    
    def get_eviction_priority(v):
        """
        Returns a priority for eviction. Lower priority = evict first.
        We want to keep high-value vertices.
        Value = remaining_usage * weight1 + (0 if input else weight2) + recompute_cost * weight3
        """
        r = remaining_usage.get(v, 0)
        is_input = 1 if v in input_set else 0
        recompute_cost = len(parents_map.get(v, []))
        # Higher value = better to keep
        value = r * 100 + (0 if is_input else 10) + recompute_cost * 5
        return -value  # Negate for min()
    
    def evict_one():
        if not fast_mem:
            return
        victim = min(fast_mem, key=get_eviction_priority)
        
        if remaining_usage.get(victim, 0) == 0:
            ops.append(['delete', victim])
        elif victim in input_set:
            ops.append(['delete', victim])
        else:
            ops.append(['store', victim])
        
        fast_mem.discard(victim)
        state[victim] = 'in_slow' if victim in input_set else 'stored'
    
    def ensure_in_memory(v):
        if v in fast_mem:
            return
        
        while len(fast_mem) >= fast_memory_budget:
            evict_one()
            
        if v in input_set:
            ops.append(['load', v])
            fast_mem.add(v)
            state[v] = 'in_fast'
        else:
            if state.get(v) == 'stored':
                ops.append(['load', v])
                fast_mem.add(v)
                state[v] = 'in_fast'
            else:
                for p in parents_map.get(v, []):
                    ensure_in_memory(p)
                ops.append(['compute', v])
                fast_mem.add(v)
                state[v] = 'in_fast'
    
    # Process in Topological Order
    for v in order:
        if time.time() > deadline:
            break
            
        ensure_in_memory(v)
        processed.add(v)
        
        # Update remaining_usage of parents
        for p in parents_map.get(v, []):
            remaining_usage[p] -= 1
    
    # Ensure outputs are in memory
    for o in outputs:
        if o not in fast_mem:
            ensure_in_memory(o)
            
    return ops

def _count_transfers(operations):
    return sum(1 for op, v in operations if op in ('load', 'store'))

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80
    
    try:
        result = _greedy_pebble_improved(dag, fast_memory_budget, deadline)
        if result and len(result) <= 20000:
            return result
        else:
            return result if result else []
    except Exception as e:
        return []
# EVOLVE-BLOCK-END
