# EVOLVE-BLOCK-START
import time
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

def _greedy_pebble_optimized(dag, fast_memory_budget, deadline):
    """
    Optimized greedy pebbling.
    Strategy: Keep nodes in fast memory as long as possible, evicting the node
    with the lowest 'Benefit' when space is needed.
    Benefit(v) = 1 + remaining_uses(v) for computed nodes.
    Benefit(v) = remaining_uses(v) for input nodes.
    """
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    
    # Identify outputs
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    output_set = set(outputs)
    
    # Identify inputs
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    input_set = set(input_vertices)
    
    # Pre-compute children for usage counting
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # Initial usage counts (how many times this vertex is needed as a parent)
    remaining_uses = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            remaining_uses[p] += 1
    
    fast_memory = set()
    operations = []
    
    # Topological order
    order = _topo_sort(vertices, parents_map)
    
    def get_benefit(v):
        """Calculate the benefit of keeping v in fast memory."""
        uses = remaining_uses.get(v, 0)
        if v in input_set:
            # Inputs don't have a 'store' cost to save, only 'load' cost per use.
            return uses
        else:
            # Computed nodes save 1 store + 1 load per use.
            # Actually, saving the store is 1. Saving the load is 1 per use.
            # So benefit is 1 + uses.
            return 1 + uses
    
    def evict_one():
        """Evict the node with the lowest benefit."""
        if not fast_memory:
            return
        # Find victim with min benefit
        victim = min(fast_memory, key=lambda v: get_benefit(v))
        if victim in input_set:
            operations.append(['delete', victim])
        else:
            operations.append(['store', victim])
        fast_memory.discard(victim)
        
    def ensure_in_memory(v):
        """Ensure v is in fast memory, evicting if necessary."""
        if v in fast_memory:
            return
            
        # If v is a computed node, we must compute it.
        # But first, its parents must be in memory.
        if v not in input_set:
            for p in parents_map.get(v, []):
                ensure_in_memory(p)
            operations.append(['compute', v])
            fast_memory.add(v)
        else:
            # v is an input, we need to load it
            while len(fast_memory) >= fast_memory_budget:
                evict_one()
            if len(fast_memory) < fast_memory_budget:
                operations.append(['load', v])
                fast_memory.add(v)
            else:
                # Should not happen if budget > 0 and we can always evict
                pass
                
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
            
        # Ensure v is computed/loaded
        ensure_in_memory(v)
        
        # Update usage counts for parents
        for p in parents_map.get(v, []):
            remaining_uses[p] -= 1
            
        # Note: We do NOT proactively evict v here.
        # We only evict when we need to load a new parent for a future vertex.
        # This maximizes the time v stays in memory, capturing all potential savings.
        
    # Ensure outputs are in fast memory at the end?
    # The problem states: "all outputs must finish blue" (in fast memory).
    # Our loop processes all vertices. If an output was evicted, we need to bring it back.
    for o in outputs:
        if o not in fast_memory:
            ensure_in_memory(o)
            
    return operations

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    try:
        result = _greedy_pebble_optimized(dag, fast_memory_budget, deadline)
        
        if result and len(result) <= 20000:
            return result
        else:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
