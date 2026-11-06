"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag: object with attributes:
        - parents: list of lists, parents[i] = list of parent vertex indices for vertex i
        - outputs: list of output vertex indices
        - num_vertices or len(parents): total number of vertices
    
    fast_memory_budget: max number of blue (in-fast-memory) values allowed
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # Extract DAG structure
    parents = dag.parents
    outputs = dag.outputs
    n = len(parents)
    
    # Build children map
    children = defaultdict(list)
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Determine which vertices are inputs (no parents) and outputs
    inputs = [v for v in range(n) if len(parents[v]) == 0]
    
    # Compute topological order (already given by index order if DAG is topologically ordered)
    topo_order = list(range(n))
    
    # Compute remaining uses: how many children will still need this vertex's value
    # For each vertex, count how many times it's referenced as a parent
    remaining_uses = [0] * n
    for v in range(n):
        for p in parents[v]:
            remaining_uses[p] += 1
    
    # Track state: which vertices are currently blue (in fast memory)
    # We need to simulate the pebbling
    
    # Strategy: Greedy topological computation with smart eviction
    # - Process vertices in topological order
    # - Before computing a vertex, ensure all its parents are blue
    # - After computing, the new vertex is blue
    # - When a vertex's value is no longer needed (all children computed), delete it
    # - When fast memory is full and we need to load, evict a vertex that won't be needed soon
    
    # First, let's identify which vertices are "leaves" in the dependency sense
    # and track when each vertex's value is last needed
    
    # last_needed[v] = the latest topological position where v's value is needed
    last_needed = [0] * n
    for v in range(n):
        last_needed[v] = v  # at minimum, needed at its own position
        for c in children[v]:
            last_needed[v] = max(last_needed[v], c)
    
    # Simulate the pebbling
    # State: set of blue vertices (in fast memory)
    # Initially, all inputs are blue
    blue = set()
    for v in inputs:
        blue.add(v)
    
    operations = []
    fast_mem_count = len(blue)
    
    # We need to track for each vertex whether it's been computed
    computed = [False] * n
    for v in inputs:
        computed[v] = True
    
    # For non-input vertices, we need to compute them
    # Process in topological order
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if computed[v]:
            # Already computed, check if we should delete it
            # Delete if no more children need it
            if remaining_uses[v] == 0 and v in blue:
                operations.append(['delete', v])
                blue.discard(v)
                fast_mem_count -= 1
            continue
        
        # Need to compute vertex v
        # First, ensure all parents are blue
        for p in parents[v]:
            if p not in blue:
                # Need to load p
                if fast_mem_count >= fast_memory_budget:
                    # Need to evict something
                    # Evict the blue vertex with the highest last_needed that is not a parent of v
                    # and has remaining_uses == 0 or will be needed least soon
                    candidates = [b for b in blue if b != p and b not in parents[v]]
                    if candidates:
                        # Evict the one with highest last_needed (won't be needed for longest)
                        # Actually, evict the one that is least likely to be needed again
                        # Prefer evicting vertices with remaining_uses == 0
                        no_use = [b for b in candidates if remaining_uses[b] == 0]
                        if no_use:
                            # Pick the one with highest last_needed
                            evict = max(no_use, key=lambda x: last_needed[x])
                        else:
                            # Pick the one with highest last_needed
                            evict = max(candidates, key=lambda x: last_needed[x])
                        operations.append(['store', evict])
                        blue.discard(evict)
                        fast_mem_count -= 1
                    
                    operations.append(['load', p])
                    blue.add(p)
                    fast_mem_count += 1
        
        # Now compute v
        operations.append(['compute', v])
        computed[v] = True
        blue.add(v)
        fast_mem_count += 1
        
        # Update remaining_uses for parents
        for p in parents[v]:
            remaining_uses[p] -= 1
    
    # Ensure all outputs are blue at the end
    for v in outputs:
        if v not in blue:
            if fast_mem_count >= fast_memory_budget:
                # Need to evict
                candidates = [b for b in blue if b != v]
                if candidates:
                    evict = max(candidates, key=lambda x: last_needed[x])
                    operations.append(['store', evict])
                    blue.discard(evict)
                    fast_mem_count -= 1
            operations.append(['load', v])
            blue.add(v)
            fast_mem_count += 1
    
    # Cap at 20000 operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END

