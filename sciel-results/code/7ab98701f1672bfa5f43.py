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

def _greedy_pebble_optimized(dag, fast_memory_budget, deadline):
    """
    Optimized Greedy Pebbling:
    - Processes vertices in topological order.
    - Keeps values in fast memory as long as possible to avoid re-loads.
    - Evicts based on remaining usage count (LRU-like).
    - Ensures outputs are stored (blue) at the end.
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
    
    input_vertices = set(v for v in vertices if not parents_map.get(v, []))
    output_set = set(outputs)
    
    # Count how many times each vertex is needed as a parent
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
            
    # remaining_usage will be decremented as we process children
    remaining_usage = dict(usage_count)
    
    fast_memory = set()
    operations = []
    
    # Track if a vertex in fast_memory is 'clean' (already stored) or 'dirty' (needs store if evicted)
    # Actually, if it's in fast memory, it's red. If we evict it, we must store it unless it's an input that we don't care about?
    # Inputs are in slow memory. If we load them, they are in fast. If we delete them, they are gone from fast. They are still in slow.
    # Computed vertices are NOT in slow memory until we store them. So if we evict a computed vertex, we MUST store it.
    
    # Helper to evict a vertex
    def evict_vertex(v):
        """Evict v from fast memory. Returns the operation performed."""
        fast_memory.discard(v)
        if v in input_vertices:
            # Input is still in slow memory. Just delete from fast.
            return ['delete', v]
        else:
            # Computed vertex. Must store to slow memory.
            return ['store', v]

    order = _topo_sort(vertices, parents_map)
    
    for v in order:
        if time.time() > deadline:
            break
            
        # 1. Ensure all parents are in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_memory:
                # Need to load p. Make room if necessary.
                while len(fast_memory) >= fast_memory_budget:
                    # Find best victim to evict
                    # Prefer vertices with 0 remaining usage
                    # Then vertices with lowest remaining usage
                    # Avoid evicting outputs if possible? Outputs must be in fast? No, they must finish blue.
                    # If we evict an output, we store it. That's fine.
                    
                    best_victim = None
                    min_usage = float('inf')
                    
                    for x in fast_memory:
                        # Don't evict the parent p we are trying to load? It's not in fast_memory anyway.
                        # Don't evict vertices that are currently needed for THIS vertex's computation?
                        # We are loading p. Other parents might already be in fast_memory.
                        # We should not evict a parent of v that is already in fast_memory.
                        if x in parents_map.get(v, []):
                            continue
                            
                        usage = remaining_usage.get(x, 0)
                        if usage < min_usage:
                            min_usage = usage
                            best_victim = x
                        elif usage == min_usage:
                            # Tie-breaker: prefer evicting inputs (cheaper?) or non-outputs
                            if best_victim in output_set and x not in output_set:
                                best_victim = x
                            elif best_victim not in input_vertices and x in input_vertices:
                                best_victim = x

                    if best_victim is None:
                        # All vertices in fast_memory are parents of v?
                        # This implies fast_memory is full of parents of v.
                        # We can't evict them. We are stuck?
                        # This should not happen if budget >= max_in_degree.
                        # If it does, we have a problem. Break to avoid infinite loop.
                        break
                        
                    # Evict best_victim
                    op = evict_vertex(best_victim)
                    operations.append(op)
                    # Note: remaining_usage for best_victim is NOT changed here.
                    # It is changed when we process the vertex that USES it.
                    # But wait, if we evict it, and it's a parent of a FUTURE vertex, we will have to load it again.
                    # That's the cost of eviction.
                
                if len(fast_memory) < fast_memory_budget:
                    operations.append(['load', p])
                    fast_memory.add(p)
                else:
                    # Should not happen if budget is sufficient for max in-degree
                    pass
        
        # 2. Compute v
        operations.append(['compute', v])
        fast_memory.add(v)
        
        # 3. Update usage counts for parents
        for p in parents_map.get(v, []):
            remaining_usage[p] = remaining_usage.get(p, 0) - 1
            
        # Note: We do NOT store v immediately. We leave it in fast_memory.
        # It will be evicted later if needed.
        
    # 4. Finalize: Ensure all outputs are Blue (in slow memory)
    # Currently, outputs are in fast_memory (Red). We must store them.
    # Also, any other vertices in fast_memory can be left or stored?
    # The requirement is "all outputs must finish blue".
    # It doesn't say other vertices must be blue.
    # But to be safe and clean, we might store everything?
    # No, storing everything increases transfer count.
    # Only outputs MUST be blue.
    # So, for each output in fast_memory, we must store it.
    
    for o in outputs:
        if o in fast_memory:
            operations.append(['store', o])
            fast_memory.discard(o)
            
    # What about non-output vertices in fast_memory?
    # They can stay Red? The contract says "every red value consumes one fast-memory slot".
    # It doesn't forbid ending with red values, as long as outputs are blue.
    # However, if we leave them, we save store operations.
    # So we leave them.
    
    return operations

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    try:
        result = _greedy_pebble_optimized(dag, fast_memory_budget, deadline)
        
        # Basic validation: check length and structure
        if result and len(result) <= 20000:
            return result
        else:
            # Fallback to baseline if optimized version fails or is too long
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
