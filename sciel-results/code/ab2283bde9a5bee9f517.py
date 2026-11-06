"""Communication-minimizing DAG pebbling from scratch."""
import heapq
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    pebble(dag, fast_memory_budget) -> list of [operation, vertex]
    Operations: 'load', 'store', 'compute', 'delete'
    dag: dict with:
      - 'vertices': list of vertex ids (or similar)
      - 'parents': dict vertex -> list of parent vertices (topologically ordered)
      - 'outputs': list of output vertex ids
    Inputs (no parents) start blue (in fast memory).
    All outputs must finish blue.
    Every red value consumes one fast-memory slot.
    At most 20000 operations.
    """
    # Parse the DAG structure
    vertices = list(dag.get('vertices', []))
    parents = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    
    if not vertices:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological order
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topo order doesn't include all vertices, just use vertex list
    if len(topo_order) != len(vertices):
        topo_order = vertices
    
    # Identify inputs (no parents)
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    # Strategy: We want to minimize transfers (load/store operations).
    # A simple but effective strategy:
    # 1. Process vertices in topological order.
    # 2. For each vertex, ensure its parents are in fast memory (load if not).
    # 3. Compute the vertex (if it has parents, it's a compute; if no parents, it's an input already blue).
    # 4. After computing, if the vertex is not an output and we're over budget, we may need to store some values.
    # 5. Actually, the key insight: inputs start blue. We need all outputs blue at the end.
    # 
    # Let's think about this more carefully. The "naive schedule" likely loads everything, computes, stores non-outputs.
    # To beat it, we should minimize loads/stores by keeping values in fast memory as long as possible.
    #
    # Better approach: Use a priority-based eviction strategy.
    # - Process in topological order.
    # - Keep track of what's in fast memory (blue).
    # - When we need to compute a vertex, ensure all parents are blue (load if red).
    # - Compute the vertex (it becomes blue).
    # - If fast memory is full and we need to store something, evict the vertex that is least likely to be needed again.
    #   Priority: outputs should never be evicted. Vertices with more remaining children should be kept longer.
    
    # Let's implement a heuristic:
    # 1. Precompute for each vertex the number of "remaining uses" in the topological order.
    # 2. Process in topological order.
    # 3. Maintain a set of vertices currently in fast memory (blue).
    # 4. For each vertex v in topo order:
    #    a. If v is an input, it's already blue.
    #    b. If v has parents, ensure all parents are blue (load if needed).
    #    c. Compute v (if it has parents). If no parents, it's just an input.
    #    d. v is now blue.
    #    e. If fast memory size exceeds budget, evict some vertices (store them).
    #       - Never evict outputs.
    #       - Evict the vertex with the fewest remaining uses (children not yet processed).
    
    # Precompute remaining children count at each step
    # Actually, let's just track which vertices have been processed.
    
    operations = []
    in_fast_memory = set()
    
    # Inputs start blue
    for v in inputs:
        in_fast_memory.add(v)
    
    # Count how many times each vertex is needed (as a parent of unprocessed vertices)
    # We'll track remaining_uses dynamically.
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # Process in topological order
    for v in topo_order:
        # If v has parents, ensure they're in fast memory
        if parents.get(v, []):
            for p in parents[v]:
                if p not in in_fast_memory:
                    operations.append(['load', p])
                    in_fast_memory.add(p)
            
            # Compute v
            operations.append(['compute', v])
            in_fast_memory.add(v)
        else:
            # v is an input, already blue
            pass
        
        # Update remaining_children for v's children
        # Actually, we should decrement remaining_children for v's parents after they're no longer needed?
        # No, remaining_children[v] is the number of children of v that haven't been processed yet.
        # When we process v, we should decrement remaining_children for... wait, no.
        # Let's redefine: remaining_uses[v] = number of unprocessed vertices that have v as a parent.
        # Initially, remaining_uses[v] = len(children[v]).
        # When we process a vertex u, for each parent p of u, we decrement remaining_uses[p].
        
        # Let's fix this:
        # remaining_uses should be tracked properly.
    
    # Let me redo this more carefully.
    
    operations = []
    in_fast_memory = set()
    
    # Inputs start blue
    for v in inputs:
        in_fast_memory.add(v)
    
    # remaining_uses[v] = number of unprocessed vertices that depend on v
    remaining_uses = {v: len(children[v]) for v in vertices}
    
    # Process in topological order
    for v in topo_order:
        # Ensure all parents of v are in fast memory
        if parents.get(v, []):
            for p in parents[v]:
                if p not in in_fast_memory:
                    operations.append(['load', p])
                    in_fast_memory.add(p)
            
            # Compute v
            operations.append(['compute', v])
            in_fast_memory.add(v)
        
        # Now, after processing v, decrement remaining_uses for v's parents
        # Because v is now processed, so it no longer "needs" its parents
        for p in parents.get(v, []):
            remaining_uses[p] -= 1
        
        # Check if we're over budget
        while len(in_fast_memory) > fast_memory_budget:
            # Need to evict someone
            # Find the best candidate to evict
            # Never evict outputs
            # Evict the one with lowest remaining_uses
            candidates = [x for x in in_fast_memory if x not in outputs]
            if not candidates:
                # All in fast memory are outputs, can't evict
                break
            
            # Find the one with lowest remaining_uses
            best_to_evict = min(candidates, key=lambda x: (remaining_uses[x], x))
            
            operations.append(['store', best_to_evict])
            in_fast_memory.remove(best_to_evict)
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v not in in_fast_memory:
            operations.append(['load', v])
            in_fast_memory.add(v)
    
    # Truncate if over 20000 operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
