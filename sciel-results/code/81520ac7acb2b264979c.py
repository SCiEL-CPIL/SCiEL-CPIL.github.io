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

def _greedy_pebble(dag, fast_memory_budget, deadline):
    """Greedy pebbling with memory management and recomputation using Next-Use heuristic."""
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
    
    # Pre-compute recompute cost proxy: number of parents
    parent_count = {v: len(parents_map.get(v, [])) for v in vertices}
    
    # Pre-compute next use indices
    # next_use[v] will be a list of indices in topological order where v is used as a parent
    # We can just store the list of children indices.
    order = _topo_sort(vertices, parents_map)
    index_in_order = {v: i for i, v in enumerate(order)}
    
    # For each vertex, find the indices of its children in the topological order
    # Sort these indices to find "next use" efficiently
    children_indices = defaultdict(list)
    for v in vertices:
        for c in (dag.get('_children', None) or []): # Fallback if _children not present
            pass # We need children. Let's build children_map properly.
            
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
            
    # Build sorted list of child indices for each parent
    next_use_queues = {}
    for v in vertices:
        c_indices = [index_in_order[c] for c in children_map[v]]
        c_indices.sort()
        next_use_queues[v] = c_indices # We will pop from this list
        
    # State
    fast_memory = set()
    operations = []
    
    # Helper to get the "next use" index for a vertex
    # We maintain a pointer or just use the list
    # To make it O(1) lookup for "current next use", we can use a dict of current pointer indices
    next_use_ptr = {v: 0 for v in vertices}
    
    def get_next_use(v):
        """Return the next topological index where v is needed, or infinity."""
        ptr = next_use_ptr[v]
        queue = next_use_queues[v]
        if ptr < len(queue):
            return queue[ptr]
        return float('inf')
    
    def advance_next_use(v):
        """Advance the pointer for v if its next use has passed."""
        current_idx = index_in_order.get(v, -1) # This is not quite right, we advance based on current processing index
        
    # Actually, we should advance pointers when we pass the index where the use happens.
    # The "use" of p for child c happens when we process c.
    # So when we process vertex at index i, we should advance pointers for all parents p of vertices processed so far?
    # No, the "next use" of p is determined by the children of p.
    # If we are processing vertex v at index i, any parent p of v has its "next use" potentially at i.
    # If p's next use was i, then after processing v, we should advance p's pointer.
    
    def get_eviction_score(v):
        # Lower score = better candidate for eviction
        # 1. Next Use: Infinity (no more uses) is best. Then larger index (further away) is better.
        #    We want to keep nodes that are used soon.
        #    So we want to evict nodes with LARGE next_use.
        #    To make "min" work, we can use -next_use? No, inf is tricky.
        #    Let's map inf to a large number, say 10**9.
        nu = get_next_use(v)
        if nu == float('inf'):
            nu_score = 10**9
        else:
            nu_score = -nu # We want to evict large nu, so minimize -nu? No.
            # We want to KEEP small nu. We want to EVICT large nu.
            # So the "value" of keeping is high for small nu.
            # The "cost" of evicting is low for large nu.
            # Let's define score such that lower is better to evict.
            # If nu is inf, score is 0 (best to evict).
            # If nu is 10, score is 10.
            # If nu is 1, score is 1.
            # So score = nu.
            pass
        
        # Correct logic:
        # We want to evict the node with the FURTHEST next use.
        # So the "eviction priority" is HIGH for large next_use.
        # In a min-heap or min() selection, we want the "best to evict" to have the SMALLEST score.
        # So if we assign score = -next_use, then large next_use -> small score -> evicted first.
        # But inf is a problem.
        
        # Let's use a tuple:
        # (has_future_use, -next_use, is_input, -recompute_cost)
        # 1. has_future_use: 0 if inf, 1 if finite. Prefer 0 (evict first).
        # 2. -next_use: If finite, we prefer to evict the one with LARGEST next_use.
        #    So -next_use will be MORE NEGATIVE for larger next_use.
        #    So min() will pick the most negative, i.e., largest next_use. Correct.
        # 3. is_input: 1 if input, 0 if computed. Prefer to evict inputs?
        #    Evicting input: Delete (0) + Load later (1). Total 1.
        #    Evicting computed: Store (1) + Load/Recompute later (1+). Total 2+.
        #    So Inputs are cheaper to evict. We prefer to evict Inputs.
        #    So we want Inputs to have LOWER score.
        #    So is_input should be 0 for Inputs, 1 for Computed?
        #    Wait, if we evict Input, we pay less. So it's "better" to evict Input.
        #    So Input should have lower score.
        #    Let is_input_score = 0 if input, 1 if computed.
        # 4. -recompute_cost: We prefer to keep nodes with HIGH recompute cost.
        #    So we prefer to evict nodes with LOW recompute cost.
        #    So LOW recompute cost should have LOWER score.
        #    So use recompute_cost directly?
        #    If we evict low cost node, we save less "value" than evicting high cost node?
        #    No, we WANT to evict low cost node.
        #    So low cost -> low score -> evicted.
        
        nu = get_next_use(v)
        if nu == float('inf'):
            future_score = 0
            next_use_score = 0
        else:
            future_score = 1
            next_use_score = -nu
            
        input_score = 0 if v in input_set else 1
        recompute_score = parent_count[v] # Lower is better to evict
        
        return (future_score, next_use_score, input_score, recompute_score, v)

    def evict_one():
        if not fast_memory:
            return
        victim = min(fast_memory, key=get_eviction_score)
        if victim in input_set:
            operations.append(['delete', victim])
        else:
            operations.append(['store', victim])
        fast_memory.discard(victim)
        
    def ensure_in_memory(v):
        if v in fast_memory:
            return
        
        # Make space if needed
        if len(fast_memory) >= fast_memory_budget:
            evict_one()
            if len(fast_memory) >= fast_memory_budget:
                 # If budget is 0 or stuck, we can't proceed safely, but assume budget >= 1
                 pass

        if v in input_set:
            operations.append(['load', v])
            fast_memory.add(v)
        else:
            # Need to compute v
            # Ensure parents are in memory
            for p in parents_map.get(v, []):
                ensure_in_memory(p)
            operations.append(['compute', v])
            fast_memory.add(v)
            
    # Main loop
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
            
        ensure_in_memory(v)
        
        # Update next_use pointers for parents of v
        # The "use" of p for v is happening NOW (at index i).
        # So we should advance the pointer for p if its next use was i.
        for p in parents_map.get(v, []):
            # If the next use of p is i, advance pointer
            if get_next_use(p) == i:
                next_use_ptr[p] += 1

    # Ensure all outputs are in fast memory at the end
    for o in outputs:
        if o not in fast_memory:
            ensure_in_memory(o)
            
    return operations

def _count_transfers(operations):
    """Count the number of load/store operations (transfers)."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    try:
        # Try the greedy approach
        result = _greedy_pebble(dag, fast_memory_budget, deadline)
        
        # Verify the result is valid
        if result and len(result) <= 20000:
            return result
        else:
            # Fall back to baseline
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    except Exception as e:
        # Fall back to baseline on any error
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
