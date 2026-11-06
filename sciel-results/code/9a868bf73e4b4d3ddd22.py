# EVOLVE-BLOCK-START
import time
import heapq
from functools import lru_cache

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag: dict with keys:
        - 'n': number of vertices
        - 'parents': list of lists, parents[i] = list of parent indices of vertex i
        - 'outputs': list of output vertex indices
        - 'inputs': list of input vertex indices (start blue)
    
    fast_memory_budget: int, max number of red vertices allowed simultaneously
    
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    n = dag['n']
    parents = dag['parents']
    outputs = dag['outputs']
    inputs = dag.get('inputs', [])
    
    # Build adjacency: children of each vertex
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    # Topological order (parents should come before children)
    # Assume vertices are already in topological order based on task description
    
    # Naive schedule: compute each vertex in topological order
    # For each vertex, load all parents, compute, store result, delete parents
    def naive_schedule():
        ops = []
        # Load all inputs
        for inp in inputs:
            ops.append(['load', inp])
        
        # Process vertices in topological order (0 to n-1)
        for v in range(n):
            if v in inputs:
                continue  # already loaded
            
            # Load all parents that aren't already in memory
            for p in parents[v]:
                # Check if parent is still in fast memory (we don't track state here for naive)
                # In naive, we assume we need to load each parent
                ops.append(['load', p])
            
            # Compute vertex v
            ops.append(['compute', v])
            
            # Store vertex v (move to slow memory)
            ops.append(['store', v])
            
            # Delete parents that are no longer needed
            for p in parents[v]:
                # Check if p is still needed by other children
                still_needed = False
                for c in children[p]:
                    if c > v:  # c comes after v in topo order
                        still_needed = True
                        break
                if not still_needed:
                    ops.append(['delete', p])
        
        # Store all outputs
        for out in outputs:
            # Check if already stored
            pass
        
        return ops
    
    # For the baseline, just use naive schedule
    # The score is mean transfers / naive schedule, so baseline = 1.0
    # To improve, we need fewer transfers than naive
    
    # Let's implement a smarter strategy:
    # 1. Identify critical path vertices that should stay in memory
    # 2. Use a greedy approach to minimize loads/stores
    
    # Count how many times each vertex is needed as a parent
    parent_count = [0] * n
    for i in range(n):
        for p in parents[i]:
            parent_count[p] += 1
    
    # Vertices that are parents to multiple children are expensive to reload
    # Priority: keep high-fanout vertices in memory
    
    # Greedy strategy:
    # Process in topological order
    # Maintain a set of vertices currently in fast memory
    # For each vertex v:
    #   - Load any parent not in memory
    #   - Compute v
    #   - Decide what to keep in memory based on future needs and memory budget
    
    in_memory = set(inputs)
    ops = []
    
    # Load inputs
    for inp in inputs:
        ops.append(['load', inp])
    
    # Track which vertices have been computed
    computed = set(inputs)
    
    # For each vertex in topological order
    for v in range(n):
        if v in computed:
            continue
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
        
        # Load parents not in memory
        for p in parents[v]:
            if p not in in_memory:
                ops.append(['load', p])
                in_memory.add(p)
        
        # Compute v
        ops.append(['compute', v])
        in_memory.add(v)
        computed.add(v)
        
        # If memory is over budget, evict some vertices
        while len(in_memory) > fast_memory_budget:
            # Evict the vertex with lowest future need
            # Future need = number of remaining children that need this vertex
            min_need = float('inf')
            evict_candidate = None
            for vertex in in_memory:
                # Count remaining children that haven't been processed
                remaining_children = 0
                for c in children[vertex]:
                    if c not in computed:
                        remaining_children += 1
                if remaining_children < min_need:
                    min_need = remaining_children
                    evict_candidate = vertex
            
            if evict_candidate is not None and min_need == 0:
                in_memory.discard(evict_candidate)
                ops.append(['delete', evict_candidate])
            else:
                # No vertex can be safely evicted, break
                break
    
    # At the end, store all outputs
    for out in outputs:
        if out in in_memory:
            ops.append(['store', out])
            in_memory.discard(out)
    
    # If we didn't finish due to deadline, return what we have
    # But we need all outputs to be blue at the end
    # For safety, if we're over deadline, just return the naive schedule
    
    if time.time() - start_time > DEADLINE:
        return naive_schedule()
    
    # Verify: all outputs should be stored (blue)
    # The problem says "all outputs must finish blue"
    # Blue means in slow memory (stored) or not in fast memory
    
    # Count transfers: load and store operations
    transfers = sum(1 for op in ops if op[0] in ('load', 'store'))
    
    return ops

# EVOLVE-BLOCK-END
