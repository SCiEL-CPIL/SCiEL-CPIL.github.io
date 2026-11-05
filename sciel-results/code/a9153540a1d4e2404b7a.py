import sys
import time

def pebble(dag, fast_memory_budget):
    """
    Minimizes communication (transfers) for a given DAG pebbling problem.
    
    Args:
        dag: A dictionary where keys are vertices and values are lists of parent vertices.
             Vertices are assumed to be topologically ordered in the input structure or will be sorted.
             The DAG includes all vertices, including outputs.
        fast_memory_budget: An integer representing the maximum number of red values in fast memory.
    
    Returns:
        A list of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'.
    """
    start_time = time.time()
    deadline = 90.0  # 90 seconds wall-clock limit
    
    # 1. Preprocessing: Topological Sort and Analysis
    # We assume 'dag' is a dict: {vertex: [parent1, parent2, ...]}
    # We need to identify roots (no parents) and outputs (no children).
    # We also need to compute the topological order.
    
    all_vertices = set(dag.keys())
    for parents in dag.values():
        all_vertices.update(parents)
    
    # Build adjacency list for children
    children = {v: [] for v in all_vertices}
    in_degree = {v: 0 for v in all_vertices}
    
    for v, parents in dag.items():
        for p in parents:
            children[p].append(v)
            in_degree[v] += 1
    
    # Topological Sort (Kahn's Algorithm)
    from collections import deque
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topological sort didn't include all vertices, there's a cycle (shouldn't happen for DAG)
    if len(topo_order) != len(all_vertices):
        # Fallback to simple order if cycle detected (though DAG is guaranteed)
        topo_order = list(all_vertices)
    
    # 2. Recursive Pebbling Strategy
    # We use a recursive approach to compute each vertex.
    # State:
    # - fast_mem: set of vertices currently in fast memory
    # - ops: list of operations
    
    # To optimize, we can use a greedy strategy:
    # - Process vertices in topological order.
    # - For each vertex, ensure its parents are in fast memory.
    # - If a parent is not in fast memory, load it (or re-compute if it was deleted).
    # - After computing a vertex, decide whether to keep it in fast memory or store it back.
    
    # A more effective strategy for minimizing transfers is to use a "look-ahead" 
    # to determine which vertices will be needed in the near future.
    
    # Let's implement a simpler but effective heuristic:
    # - Keep vertices in fast memory as long as they are needed for future computations.
    # - If the budget is exceeded, store the vertex that is least likely to be needed soon.
    
    # We'll pre-compute the "next use" for each vertex.
    # next_use[v] = the index in topo_order of the next vertex that uses v as a parent.
    
    next_use = {v: len(topo_order) for v in all_vertices}
    for i, v in enumerate(topo_order):
        for p in dag.get(v, []):
            if next_use[p] == len(topo_order):
                next_use[p] = i
    
    # Now, we'll simulate the pebbling process.
    fast_mem = set()
    ops = []
    
    # We need to handle the fact that some vertices might be computed multiple times
    # if they are evicted from fast memory. We'll track which vertices are currently
    # "valid" in fast memory (i.e., their value is correct and up-to-date).
    
    # For simplicity, we'll assume that once a vertex is computed, its value is fixed.
    # We just need to ensure it's in fast memory when needed.
    
    for v in topo_order:
        # Check if we're close to the deadline
        if time.time() > deadline - 5.0:
            break
        
        # Ensure all parents of v are in fast memory
        parents = dag.get(v, [])
        for p in parents:
            if p not in fast_mem:
                # Load p into fast memory
                ops.append(['load', p])
                fast_mem.add(p)
        
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # Check if fast memory is full
        while len(fast_mem) > fast_memory_budget:
            # Find the vertex in fast_mem that is least likely to be needed soon
            # We can use the next_use index: the larger the index, the later it's needed.
            # If next_use is len(topo_order), it's never needed again.
            
            # We should not evict v (just computed) if possible, but we might have to.
            # Actually, we should evict the vertex with the largest next_use index.
            
            # Exclude v from eviction candidates if possible? No, we must evict something.
            # But v is just computed, so it's likely to be needed soon if it has children.
            # Let's just evict the one with the largest next_use.
            
            # Find the vertex with the largest next_use
            evict_candidate = None
            max_next_use = -1
            for u in fast_mem:
                if next_use[u] > max_next_use:
                    max_next_use = next_use[u]
                    evict_candidate = u
            
            if evict_candidate is not None:
                ops.append(['store', evict_candidate])
                fast_mem.remove(evict_candidate)
    
    return ops
