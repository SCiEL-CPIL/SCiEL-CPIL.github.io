"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def _topo_order(dag):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return [], {}
    vset = set(vertices)
    indeg = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in vset:
                indeg[v] = len([p2 for p2 in ps if p2 in vset])
                children[p].append(v)
    # Recompute indeg properly
    indeg = {v: 0 for v in vertices}
    for v in vertices:
        ps = [p for p in parents.get(v, []) if p in vset]
        indeg[v] = len(ps)
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(vertices):
        topo = vertices
    return topo, children

def _pebble_layered(dag, budget):
    """
    Pebbling strategy for layered DAGs using recursive divide-and-conquer.
    Key idea: For a chain of layers, use a pebbling strategy that minimizes
    total transfers by recomputing intermediate values when memory is tight.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    
    topo, children = _topo_order(dag)
    
    # Identify layers: vertices with same "depth" (longest path from source)
    depth = {}
    for v in topo:
        ps = [p for p in parents.get(v, []) if p in vset]
        if not ps:
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in ps)
    
    max_depth = max(depth.values()) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in vertices:
        layers[depth[v]].append(v)
    
    # For each layer, determine which vertices are "needed" (have a path to an output)
    needed = set(outputs)
    for v in reversed(topo):
        if v in needed:
            continue
        for c in children.get(v, []):
            if c in needed:
                needed.add(v)
                break
    
    # Pebbling strategy:
    # For layered DAGs, we process layer by layer.
    # Within each layer, we compute all needed vertices.
    # We keep a sliding window of layers in fast memory.
    # When memory is full, we recompute from the nearest retained layer.
    
    # Strategy: Keep layers 0..k in memory (where k is determined by budget),
    # and recompute deeper layers as needed.
    
    # Count vertices per layer to estimate memory needs
    layer_sizes = [len([v for v in layers[d] if v in needed]) for d in range(len(layers))]
    
    # Determine how many layers we can keep in fast memory
    # Each vertex in fast memory uses 1 slot
    # We need to keep at least the current layer's parents available
    
    # Simple strategy: Process in topological order, load parents on demand,
    # compute, and delete parents that are no longer needed.
    # This is the greedy approach but optimized for layered structure.
    
    ops = []
    loaded = set()  # vertices currently in fast memory
    computed = set()  # vertices that have been computed at least once
    
    # Track which vertices are still needed (have unprocessed children in needed set)
    remaining_needed = set(needed)
    
    def get_layer(v):
        return depth.get(v, 0)
    
    # Process vertices in topological order
    for v in topo:
        if v not in needed:
            continue
        
        ps = [p for p in parents.get(v, []) if p in vset and p in needed]
        
        # Load parents that aren't already in fast memory
        for p in ps:
            if p not in loaded:
                ops.append(["load", p])
                loaded.add(p)
        
        # Compute v
        ops.append(["compute", v])
        loaded.add(v)
        computed.add(v)
        
        # If v is an output, store it
        if v in outset:
            ops.append(["store", v])
            loaded.discard(v)
            remaining_needed.discard(v)
        
        # Delete parents that are no longer needed
        for p in ps:
            # Check if p is still needed by any other unprocessed vertex
            still_needed = False
            for c in children.get(p, []):
                if c != v and c in needed and c not in computed:
                    still_needed = True
                    break
            if not still_needed and p not in outset and p not in loaded:
                pass  # already deleted
            elif not still_needed and p not in outset:
                ops.append(["delete", p])
                loaded.discard(p)
    
    return ops[:20000]

def _pebble_optimal_small(dag, budget):
    """
    For small DAGs, use BFS/A* to find optimal pebbling.
    State: frozenset of vertices in fast memory.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 16:  # Too large for exact search
        return None
    
    idx = {v: i for i, v in enumerate(vertices)}
    outset = set(outputs)
    
    # Precompute parent masks
    par_mask = [0] * n
    for v, ps in parents.items():
        i = idx[v]
        m = 0
        for p in ps:
            if p in idx:
                m |= (1 << idx[p])
        par_mask[i] = m
    
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    
    # BFS from empty set to find min transfers
    # State: (loaded_mask, stored_mask)
    # Goal: all outputs stored
    # Transfers: load (add to loaded), store (remove from loaded, add to stored)
    # compute: no transfer, adds to loaded if all parents in loaded
    # delete: no transfer, removes from loaded
    
    # Actually, let's count only load and store as transfers
    # Use BFS to find minimum number of transfers
    
    from collections import deque as dq
    
    start_state = (0, 0)  # (loaded, stored)
    goal_check = lambda s: (s[1] & out_mask) == out_mask
    
    visited = {start_state: 0}
    parent_op = {}
    queue = dq([(start_state, 0)])
    
    while queue:
        state, g = queue.popleft()
        loaded, stored = state
        if goal_check(state):
            # Reconstruct
            seq = []
            s = state
            while s != start_state:
                prev, op, v = parent_op[s]
                seq.append([op, v])
                s = prev
            seq.reverse()
            return seq[:20000]
        
        if g > visited.get(state, float('inf')):
            continue
            
        for v in vertices:
            i = idx[v]
            bit = 1 << i
            
            # Load v if not loaded
            if not (loaded & bit):
                ns = (loaded | bit, stored)
                ng = g + 1
                if ng < visited.get(ns, float('inf')):
                    visited[ns] = ng
                    parent_op[ns] = (state, "load", v)
                    queue.append((ns, ng))
            
            # Store v if loaded and is output
            if (loaded & bit) and (out_mask & bit) and not (stored & bit):
                ns = (loaded & ~bit, stored | bit)
                ng = g + 1
                if ng < visited.get(ns, float('inf')):
                    visited[ns] = ng
                    parent_op[ns] = (state, "store", v)
                    queue.append((ns, ng))
            
            # Delete v if loaded and not output and not stored
            if (loaded & bit) and not (out_mask & bit):
                ns = (loaded & ~bit, stored)
                ng = g  # delete is not a transfer
                if ng < visited.get(ns, float('inf')):
                    visited[ns] = ng
                    parent_op[ns] = (state, "delete", v)
                    queue.append((ns, ng))
            
            # Compute v if all parents loaded
            if (loaded & par_mask[i]) == par_mask[i] and not (loaded & bit):
                ns = (loaded | bit, stored)
                ng = g  # compute is not a transfer
                if ng < visited.get(ns, float('inf')):
                    visited[ns] = ng
                    parent_op[ns] = (state, "compute", v)
                    queue.append((ns, ng))
    
    return None

def _pebble_greedy_optimized(dag, budget):
    """
    Optimized greedy pebbling with better deletion heuristics.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    
    topo, children = _topo_order(dag)
    
    # Compute depth for better ordering
    depth = {}
    for v in topo:
        ps = [p for p in parents.get(v, []) if p in vset]
        if not ps:
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in ps)
    
    # Sort by depth, then by number of children (process high-branching first)
    def sort_key(v):
        return (depth.get(v, 0), -len(children.get(v, [])))
    
    # Use topological order but prioritize vertices with more remaining children
    ops = []
    loaded = set()
    computed = set()
    
    # Track remaining uses of each vertex
    remaining_uses = {}
    for v in vertices:
        remaining_uses[v] = 0
    for v in vertices:
        for p in parents.get(v, []):
            if p in vset:
                remaining_uses[p] = remaining_uses.get(p, 0) + 1
    
    # Process in topological order
    for v in topo:
        ps = [p for p in parents.get(v, []) if p in vset]
        
        # Load parents
        for p in ps:
            if p not in loaded:
                ops.append(["load", p])
                loaded.add(p)
        
        # Compute
        ops.append(["compute", v])
        loaded.add(v)
        computed.add(v)
        
        # Store if output
        if v in outset:
            ops.append(["store", v])
            loaded.discard(v)
        
        # Decrement remaining uses of parents
        for p in ps:
            remaining_uses[p] = remaining_uses.get(p, 0) - 1
            # Delete if no more uses and not output
            if remaining_uses[p] <= 0 and p not in outset and p in loaded:
                ops.append(["delete", p])
                loaded.discard(p)
    
    return ops[:20000]

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50  # Safety margin before 60s limit
    try:
        # Try optimal for small DAGs
        n = len(set(list(dag.get("parents", {}).keys()) + dag.get("outputs", [])))
        if n <= 16:
            result = _pebble_optimal_small(dag, fast_memory_budget)
            if result is not None and len(result) <= 20000:
                return result
        
        # Use optimized greedy for larger DAGs
        result = _pebble_greedy_optimized(dag, fast_memory_budget)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    
    # Fallback
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        pass
    
    # Final fallback: simple greedy
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    
    topo, children = _topo_order(dag)
    
    ops = []
    loaded = set()
    remaining_uses = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in vset:
                remaining_uses[p] = remaining_uses.get(p, 0) + 1
    
    for v in topo:
        ps = [p for p in parents.get(v, []) if p in vset]
        for p in ps:
            if p not in loaded:
                ops.append(["load", p])
                loaded.add(p)
        ops.append(["compute", v])
        loaded.add(v)
        if v in outset:
            ops.append(["store", v])
            loaded.discard(v)
        for p in ps:
            remaining_uses[p] = remaining_uses.get(p, 0) - 1
            if remaining_uses[p] <= 0 and p not in outset and p in loaded:
                ops.append(["delete", p])
                loaded.discard(p)
    
    return ops[:20000]
# EVOLVE-BLOCK-END
