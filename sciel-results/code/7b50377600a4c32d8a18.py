"""Communication-minimizing DAG pebbling with layered checkpointing."""
import time
import heapq
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize transfers (loads + stores).
    
    Strategy:
    1. Analyze the DAG structure (parents, children, outputs).
    2. Identify independent computation chains or use a topological approach.
    3. Use a heuristic based on pebbling theory:
       - For layered DAGs, use a 'checkpointing' strategy where we keep
         certain intermediate values in fast memory to avoid recomputation.
       - The optimal number of checkpoints balances memory usage against
         recomputation cost.
    
    Operations:
    - load: bring a value from slow memory to fast (counts as transfer)
    - store: write a value from fast to slow (counts as transfer)
    - compute: compute a vertex value from its parents (no transfer cost)
    - delete: remove a value from fast memory (no transfer cost)
    
    We aim to minimize total transfers (loads + stores).
    """
    start_time = time.time()
    DEADLINE = 85.0  # seconds, leave 5s margin
    
    # Parse the DAG structure
    # dag format: dict with keys:
    #   'num_vertices': int
    #   'parents': list of lists, parents[i] = list of parent vertex indices
    #   'outputs': list of output vertex indices
    #   'inputs': list of input vertex indices (start blue)
    
    num_v = dag.get('num_vertices', 0)
    parents = dag.get('parents', [])
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', list(range(len(parents)) if all(len(p) == 0 for p in parents) else [])))
    
    # If no inputs specified, assume vertices with no parents are inputs
    if not inputs:
        inputs = set(i for i in range(num_v) if not parents[i])
    
    # Build children adjacency
    children = defaultdict(list)
    for i in range(num_v):
        for p in parents[i]:
            children[p].append(i)
    
    # Topological order (parents should already be ordered, but verify)
    in_degree = [len(parents[i]) for i in range(num_v)]
    topo_order = []
    queue = list(range(num_v))
    # Simple topological sort assuming parents come before children
    visited = [False] * num_v
    def dfs(u):
        if visited[u]:
            return
        visited[u] = True
        for c in children[u]:
            dfs(c)
        topo_order.append(u)
    
    for i in range(num_v):
        if not visited[i]:
            dfs(i)
    
    topo_order.reverse()  # Now in topological order (parents first)
    
    # Determine which vertices need to be computed
    # All vertices reachable from inputs that are ancestors of outputs
    needed = set()
    def mark_needed(u):
        if u in needed:
            return
        needed.add(u)
        for c in children[u]:
            mark_needed(c)
    
    for o in outputs:
        mark_needed(o)
    
    # Filter to only needed vertices
    needed_list = [v for v in topo_order if v in needed]
    needed_set = set(needed_list)
    
    # Compute depth of each vertex (longest path from any input)
    depth = [0] * num_v
    for v in needed_list:
        if parents[v]:
            depth[v] = max(depth[p] for p in parents[v] if p in needed_set) + 1
        else:
            depth[v] = 0
    
    max_depth = max(depth) if depth else 0
    
    # Strategy: Layered checkpointing
    # For a DAG with max_depth D and memory budget M:
    # - We can keep M values in fast memory
    # - Optimal strategy: divide computation into phases
    # - In each phase, compute a "frontier" of vertices
    # - Store some intermediate results to avoid recomputation
    
    # Simple approach: Use a greedy pebbling strategy
    # Keep the most "expensive" intermediate values in fast memory
    
    # Cost of recomputing a vertex = number of vertices in its subtree that need to be recomputed
    subtree_size = [1] * num_v
    for v in reversed(needed_list):
        for c in children[v]:
            if c in needed_set:
                subtree_size[v] += subtree_size[c]
    
    # Greedy pebbling:
    # 1. Load all inputs
    # 2. Compute vertices in topological order
    # 3. When memory is full, evict the vertex with smallest subtree_size
    #    (i.e., cheapest to recompute)
    # 4. When a needed value is evicted, recompute it when needed
    
    # Actually, let's use a more principled approach:
    # The naive schedule loads each input once, computes each vertex once,
    # and stores outputs. Transfers = |inputs| + |outputs|.
    
    # For better than naive, we need to exploit the structure.
    # Key insight: if we can keep intermediate values, we avoid reloading inputs
    # or recomputing subtrees.
    
    # Let's implement a simple but effective strategy:
    # Phase-based computation with checkpointing at strategic points.
    
    # Determine the "width" of the DAG at each depth
    width_at_depth = defaultdict(list)
    for v in needed_list:
        width_at_depth[depth[v]].append(v)
    
    # Simple strategy: if memory budget >= max width, just compute in topological order
    max_width = max(len(width_at_depth[d]) for d in width_at_depth) if width_at_depth else 0
    
    if fast_memory_budget >= max_width:
        # Can keep all current frontier in memory
        ops = []
        # Load all inputs
        for i in sorted(inputs & needed_set):
            ops.append(['load', i])
        
        # Compute in topological order
        computed = set(inputs & needed_set)
        in_memory = set(inputs & needed_set)
        
        for v in needed_list:
            if v in computed:
                continue
            # All parents should be in memory or computed
            for p in parents[v]:
                if p in needed_set and p not in in_memory:
                    # Need to reload p
                    ops.append(['load', p])
                    in_memory.add(p)
            ops.append(['compute', v])
            in_memory.add(v)
            computed.add(v)
            # Remove parents that are no longer needed
            for p in parents[v]:
                if p in needed_set:
                    # Check if p is needed by any other uncomputed vertex
                    still_needed = False
                    for c in children[p]:
                        if c in needed_set and c not in computed:
                            still_needed = True
                            break
                    if not still_needed:
                        in_memory.discard(p)
        
        # Store outputs
        for o in sorted(outputs & needed_set):
            if o in in_memory:
                ops.append(['store', o])
            else:
                ops.append(['load', o])
                ops.append(['store', o])
        
        return ops
    
    # For limited memory, use a recomputation strategy
    # The key tradeoff: keep some values in memory vs recompute others
    
    # Use a heuristic: keep values with largest subtree_size in memory
    # This minimizes expected recomputation cost
    
    ops = []
    in_memory = set()
    computed = set()
    
    # Load all inputs initially if they fit
    input_list = sorted(inputs & needed_set)
    if len(input_list) <= fast_memory_budget:
        for i in input_list:
            ops.append(['load', i])
            in_memory.add(i)
            computed.add(i)
    else:
        # Can't keep all inputs; load as needed
        pass
    
    # Process vertices in topological order
    for v in needed_list:
        if v in computed and v in in_memory:
            continue
        
        # Ensure all parents are available
        for p in parents[v]:
            if p not in needed_set:
                continue
            if p not in in_memory:
                # Need to make p available
                if p in computed:
                    # Reload from slow memory
                    ops.append(['load', p])
                    in_memory.add(p)
                else:
                    # Need to compute p first (shouldn't happen in topo order)
                    pass
        
        # Make space if needed
        if len(in_memory) >= fast_memory_budget and v not in in_memory:
            # Evict the value with smallest subtree_size that's not an output
            candidates = [x for x in in_memory if x not in outputs]
            if candidates:
                evict = min(candidates, key=lambda x: subtree_size[x])
                in_memory.discard(evict)
                # Don't need to explicitly delete; it's just gone from fast memory
        
        # Compute v
        if v not in in_memory:
            ops.append(['compute', v])
            in_memory.add(v)
            computed.add(v)
    
    # Store outputs
    for o in sorted(outputs & needed_set):
        if o not in in_memory:
            ops.append(['load', o])
            in_memory.add(o)
        ops.append(['store', o])
    
    return ops

# EVOLVE-BLOCK-END
