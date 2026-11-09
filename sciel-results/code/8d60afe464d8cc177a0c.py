"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from heapq import heappush, heappop
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 80.0  # seconds, safety margin before 90s kill
    
    # Parse the DAG structure
    if hasattr(dag, 'num_vertices'):
        n = dag.num_vertices
    elif hasattr(dag, 'n'):
        n = dag.n
    elif isinstance(dag, dict) and 'n' in dag:
        n = dag['n']
    else:
        if hasattr(dag, 'vertices'):
            n = len(dag.vertices)
        else:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    
    def get_parents(v):
        if hasattr(dag, 'parents'):
            return dag.parents[v]
        elif isinstance(dag, dict) and 'parents' in dag:
            return dag['parents'][v]
        else:
            return []
    
    def get_outputs():
        if hasattr(dag, 'outputs'):
            return set(dag.outputs)
        elif isinstance(dag, dict) and 'outputs' in dag:
            return set(dag['outputs'])
        else:
            return {n-1}
    
    outputs = get_outputs()
    
    # Build efficient data structures
    parents_list = [get_parents(v) for v in range(n)]
    
    # Compute children for each vertex
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents_list[v]:
            children[p].append(v)
    
    # Compute remaining uses (how many children still need this vertex)
    remaining_uses = [0] * n
    for v in range(n):
        remaining_uses[v] = len(children[v])
    
    # Compute depth (topological level) for hierarchical decomposition
    depth = [0] * n
    for v in range(n):
        if parents_list[v]:
            depth[v] = max(depth[p] for p in parents_list[v]) + 1
    
    # Group vertices by depth for hierarchical processing
    depth_groups = defaultdict(list)
    for v in range(n):
        depth_groups[depth[v]].append(v)
    
    max_depth = max(depth) if depth > 0 else 0
    
    red = set()  # vertices currently in fast memory
    operations = []
    
    # Priority queue for eviction: (priority_key, vertex)
    # Lower priority_key = evict first
    # We use a lazy deletion approach: push entries, ignore stale ones
    evict_heap = []
    evict_version = [0] * n  # version number to detect stale entries
    
    # Precompute: for each vertex, the minimum depth of its children (next need depth)
    next_need_depth = [max_depth + 1] * n  # infinity if no children
    for v in range(n):
        if children[v]:
            next_need_depth[v] = min(depth[c] for c in children[v])
    
    def get_eviction_priority(v):
        """
        Compute eviction priority for vertex v.
        Lower value = should be evicted sooner.
        
        Strategy:
        - If remaining_uses[v] == 0: very low priority (evict first)
        - Otherwise: consider reload cost and urgency
          * More remaining uses = more expensive to reload = keep longer
          * Sooner next need = more urgent = keep longer
          * Output vertices = must keep until stored
        """
        ru = remaining_uses[v]
        if ru == 0:
            return (0, 0, 0)  # Evict immediately
        
        # If this is an output vertex, it's critical
        is_output = v in outputs
        
        # Reload cost: number of future loads needed if we evict now
        reload_cost = ru
        
        # Urgency: how soon will this vertex be needed again?
        # Lower next_need_depth means sooner = more urgent = keep longer
        urgency = max_depth + 1 - next_need_depth[v]  # Higher = more urgent
        
        # Combined score: we want to KEEP vertices with high reload_cost and high urgency
        # So eviction priority (lower = evict first) should be NEGATIVE of (reload_cost + urgency)
        # Add a large bonus for outputs
        output_bonus = 1000 if is_output else 0
        
        # We want lower priority_key to mean evict first
        # So: priority_key = -(reload_cost * 10 + urgency * 5 + output_bonus)
        # Vertices with high (reload_cost + urgency) get very negative priority_key = keep
        # Vertices with low (reload_cost + urgency) get less negative/positive = evict
        
        combined = reload_cost * 10 + urgency * 5 + output_bonus
        return (-combined, -ru, -urgency)
    
    def update_evict_priority(v):
        """Update the eviction priority for vertex v in the heap"""
        evict_version[v] += 1
        priority = get_eviction_priority(v)
        heappush(evict_heap, (priority[0], priority[1], priority[2], evict_version[v], v))
    
    def evict_lowest_priority():
        """Evict vertex with lowest priority using the heap"""
        if not red:
            return None
        
        # Find the valid entry with lowest priority
        while evict_heap:
            p0, p1, p2, ver, v = heappop(evict_heap)
            if v not in red:
                continue  # Already evicted
            if evict_version[v] != ver:
                continue  # Stale entry
            # This is the best candidate
            operations.append(['store', v])
            red.discard(v)
            return v
        
        # Fallback: if heap is empty but red is not, evict arbitrarily
        if red:
            v = next(iter(red))
            operations.append(['store', v])
            red.discard(v)
            return v
        return None
    
    def ensure_space():
        """Ensure fast memory has space for one more vertex"""
        while len(red) >= fast_memory_budget:
            evict_lowest_priority()
    
    def load_vertex(v):
        """Load vertex v into fast memory if not already there"""
        if v not in red:
            ensure_space()
            if v not in red:
                operations.append(['load', v])
                red.add(v)
                update_evict_priority(v)
    
    # Hierarchical pebbling: process by depth levels
    # At each level, we can keep vertices in fast memory for use in next level
    # Strategy: keep vertices with high remaining uses and high urgency in fast memory
    
    for d in range(max_depth + 1):
        if d not in depth_groups:
            continue
        
        level_vertices = depth_groups[d]
        
        # Process vertices at this depth
        for v in level_vertices:
            if time.time() - start_time > deadline - 10:
                break
            
            if not parents_list[v]:
                # Input vertex: load from slow to fast
                load_vertex(v)
            else:
                # Has parents: need all parents in fast memory to compute
                for p in parents_list[v]:
                    if p not in red:
                        load_vertex(p)
                
                # Compute v
                operations.append(['compute', v])
                red.add(v)
                update_evict_priority(v)
            
            # Decrement remaining uses for parents and update their priorities
            # Immediately evict parents that are no longer needed
            for p in parents_list[v]:
                remaining_uses[p] -= 1
                if p in red:
                    if remaining_uses[p] == 0:
                        # Evict immediately
                        operations.append(['store', p])
                        red.discard(p)
                    else:
                        update_evict_priority(p)
        
        # After processing each level, if still over budget, evict lowest priority
        while len(red) >= fast_memory_budget:
            evict_lowest_priority()
    
    # Ensure all outputs are blue (stored back to slow memory)
    # Process in reverse topological order to minimize conflicts
    for v in sorted(outputs, key=lambda x: depth[x], reverse=True):
        if v in red:
            operations.append(['store', v])
            red.discard(v)
    
    # Check time and operation limit
    if len(operations) > 20000:
        # Try to trim non-essential operations
        # Keep all compute and store operations for outputs
        essential = []
        for op in operations:
            if op[0] == 'compute' or (op[0] == 'store' and op[1] in outputs):
                essential.append(op)
            elif len(essential) < 20000:
                essential.append(op)
        operations = essential[:20000]
        # Ensure outputs are still stored
        for v in sorted(outputs, key=lambda x: depth[x], reverse=True):
            if v in red:
                operations.append(['store', v])
    
    return operations
# EVOLVE-BLOCK-END
