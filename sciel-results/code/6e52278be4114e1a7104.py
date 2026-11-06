"""Communication-minimizing DAG pebbling."""
import time
import heapq
from collections import defaultdict, deque

# EVOLVE-BLOCK-START

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize communication (load/store operations).
    
    dag: dict with keys:
        - 'n': number of vertices
        - 'parents': list of lists, parents[i] = list of parent vertices of vertex i
        - 'outputs': list of output vertex indices
        - 'inputs': list of input vertex indices (start blue)
    fast_memory_budget: int, max number of red values in fast memory
    
    Returns: list of [operation, vertex] where operation in {'load','store','compute','delete'}
    """
    deadline = time.time() + 80  # 80s safety margin
    
    n = dag['n']
    parents = dag['parents']
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    # Compute topological order
    topo = _topo_order(n, parents)
    
    # Compute children
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Identify which vertices need to be computed (have parents)
    # and which are inputs
    is_input = [v in inputs for v in range(n)]
    is_output = [v in outputs for v in range(n)]
    has_parents = [len(parents[v]) > 0 for v in range(n)]
    
    # Count how many times each vertex's value is needed
    # A vertex's value is needed by each of its children
    # For outputs, the value must be stored at the end
    
    # Strategy: Use a greedy approach based on reference counting
    # We process vertices in topological order, deciding when to compute/store/delete
    
    # First, let's understand the naive schedule:
    # Naive: compute each vertex once, store all intermediate results, load/store as needed
    # This gives us a baseline to beat
    
    # Key insight: We want to minimize load/store operations.
    # - 'compute' at vertex v: requires all parents of v to be in fast memory (blue)
    # - 'load' at vertex v: move v from slow to fast memory (costs 1 transfer)
    # - 'store' at vertex v: move v from fast to slow memory (costs 1 transfer)
    # - 'delete' at vertex v: remove v from fast memory (no cost, frees a slot)
    
    # A vertex can be:
    # - In fast memory (blue): available for computation, costs a slot
    # - In slow memory (red): available but needs load to use, no slot cost
    # - Not yet computed: needs to be computed first
    
    # Optimal strategy considerations:
    # 1. Compute vertices as late as possible to keep memory usage low
    # 2. Store values that will be needed again
    # 3. Delete values that won't be needed again
    # 4. Recompute if it's cheaper than storing/loading
    
    # Let's use a dynamic programming / greedy approach:
    # Process in reverse topological order to determine when each vertex needs to be available
    
    # For each vertex, compute:
    # - last_use[v]: the latest point in the schedule where v's value is needed
    # - use_count[v]: how many times v's value is needed (by children + output requirement)
    
    # Actually, let me think about this differently.
    # The DAG is processed in topological order. Each vertex is computed once (or recomputed).
    # After computing v, its value can be:
    #   - Kept in fast memory (uses a slot) until needed
    #   - Stored to slow memory (1 transfer), then loaded when needed (1 transfer per use)
    #   - Deleted and recomputed when needed (computation cost + parent availability cost)
    
    # For a tree-like structure, we can use a recursive strategy.
    # For general DAGs, we need a more careful approach.
    
    # Let's try a greedy heuristic:
    # 1. Process vertices in topological order
    # 2. When computing vertex v, ensure all parents are in fast memory
    # 3. After computing v, decide whether to keep in fast memory or store
    # 4. Keep in fast memory if it will be used again soon and memory allows
    # 5. Store if it will be used again later or if it's an output
    # 6. Delete if it won't be used again
    
    # Better approach: Use reference counting and a priority-based eviction
    
    # Let me implement a more principled approach:
    # 
    # Phase 1: Determine the "liveness" of each vertex
    #   - A vertex is live from when it's computed until its last use
    #   - Last use = max over all children of (time child is computed)
    #   - For outputs, last use = end of schedule
    #
    # Phase 2: Schedule computations to minimize peak memory usage
    #   - Use a greedy algorithm that processes vertices in topological order
    #   - At each step, compute the vertex with the earliest "deadline" (last use)
    #   - This is similar to scheduling with deadlines to minimize memory
    
    # Let me implement this step by step.
    
    # Compute last use for each vertex
    # last_use[v] = the latest topological index where v's value is needed
    # For a vertex v, it's needed by its children when they are computed
    # So last_use[v] = max over children c of v of (position of c in topo order)
    # For outputs, last_use[v] = n (end of schedule)
    
    # First, get position of each vertex in topological order
    pos_in_topo = [0] * n
    for i, v in enumerate(topo):
        pos_in_topo[v] = i
    
    # Compute last_use
    last_use = [0] * n
    for v in range(n):
        if is_output[v]:
            last_use[v] = n  # Needed until the very end
        else:
            if children[v]:
                last_use[v] = max(pos_in_topo[c] for c in children[v])
            else:
                last_use[v] = pos_in_topo[v]  # Only needed at computation time
    
    # Now, let's simulate the schedule
    # We process vertices in topological order
    # State:
    #   - fast_mem: set of vertices currently in fast memory
    #   - slow_mem: set of vertices currently in slow memory (already computed)
    #   - computed: set of vertices that have been computed at least once
    #   - ops: list of operations
    
    # For each vertex in topo order:
    #   1. Ensure all parents are in fast memory
    #      - If parent is in slow_mem, load it
    #      - If parent is not computed, this shouldn't happen (topo order)
    #   2. Compute the vertex
    #   3. Decide what to do with the result:
    #      - If it's an output, keep in fast memory
    #      - If it will be needed again, decide to keep or store based on memory pressure
    #      - If not needed again, delete it
    
    # But we also need to manage memory pressure.
    # If fast memory is full, we need to store/delete some values.
    
    # Let me implement a more sophisticated approach:
    
    # Key observation: The naive schedule computes each vertex once and keeps all
    # intermediate results in fast memory (or stores them). The number of transfers
    # in the naive schedule is roughly 2 * (number of non-input, non-output vertices 
    # that are stored and reloaded) + similar terms.
    
    # Our goal is to do better than naive.
    
    # Let me try a different approach: dynamic programming on the DAG structure.
    # For each vertex, we can decide:
    #   - When to compute it
    #   - Whether to keep its result in fast memory or store it
    #   - Whether to recompute it later
    
    # This is complex for general DAGs. Let me try a greedy heuristic first.
    
    # Greedy algorithm:
    # 1. Process vertices in topological order
    # 2. Maintain fast memory as a set
    # 3. Before computing v, ensure all parents are in fast memory
    #    - For each parent p of v:
    #      - If p is in fast memory: OK
    #      - If p is in slow memory: load p (add to fast memory)
    #      - If p is not computed: error (shouldn't happen)
    # 4. Compute v
    # 5. After computing v, decide its fate:
    #    - If v is an output: keep in fast memory
    #    - Else if v has children:
    #      - If memory allows: keep in fast memory
    #      - Else: store v to slow memory
    #    - Else: delete v
    # 6. Before computing v, if memory is full, evict some values
    #    - Evict the value with the latest last_use (least urgent)
    #    - Store it to slow memory
    
    # Let me implement this:
    
    ops = []
    fast_mem = set()  # Vertices in fast memory
    slow_mem = set()  # Vertices in slow memory
    computed = set()  # Vertices that have been computed
    
    # Inputs start in fast memory
    for v in inputs:
        fast_mem.add(v)
    
    # Process in topological order
    for v in topo:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in fast_mem:
                if p in slow_mem:
                    # Load p from slow to fast
                    ops.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
                elif p in computed:
                    # Should be in fast or slow, but just in case
                    ops.append(['load', p])
                    fast_mem.add(p)
                else:
                    # Not computed yet - shouldn't happen in topo order
                    pass
        
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
        
        # Decide what to do with v
        if is_output[v]:
            # Keep in fast memory
            pass
        elif children[v]:
            # Will be needed again
            # Check if we have memory pressure
            # For now, keep in fast memory if budget allows
            if len(fast_mem) > fast_memory_budget:
                # Need to evict something
                # Evict the vertex with the latest last_use that is not an output
                # and not currently needed (not a parent of any unprocessed vertex)
                # Actually, we can only evict vertices that are not needed immediately
                # Let's find a vertex to evict
                evicted = False
                # Sort by last_use descending (evict the one needed latest)
                candidates = [x for x in fast_mem if not is_output[x] and x != v]
                candidates.sort(key=lambda x: last_use[x], reverse=True)
                for cand in candidates:
                    # Check if cand is a parent of any unprocessed vertex
                    # Actually, we can evict it if it's not needed right now
                    # But to be safe, let's evict the one with the latest last_use
                    ops.append(['store', cand])
                    fast_mem.discard(cand)
                    slow_mem.add(cand)
                    evicted = True
                    break
                if not evicted:
                    # All are outputs or v itself, just proceed
                    pass
        else:
            # Not needed again, delete
            ops.append(['delete', v])
            fast_mem.discard(v)
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v not in fast_mem:
            if v in slow_mem:
                ops.append(['load', v])
                fast_mem.add(v)
                slow_mem.discard(v)
    
    return ops


def _topo_order(n, parents):
    """Compute topological order of the DAG."""
    in_degree = [0] * n
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
            in_degree[v] += 1
    
    queue = deque([v for v in range(n) if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    return topo

# EVOLVE-BLOCK-END
