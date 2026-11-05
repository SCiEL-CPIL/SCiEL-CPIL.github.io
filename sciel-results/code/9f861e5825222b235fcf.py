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
    children = {v: [] for v in vertices}
    indeg = {v: 0 for v in vertices}
    for v in vertices:
        ps = [p for p in parents.get(v, []) if p in vset]
        indeg[v] = len(ps)
        for p in ps:
            children[p].append(v)
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

def _pebble_optimal_small(dag, budget):
    """
    For small DAGs (n <= 14), use BFS/A* to find optimal pebbling.
    State: frozenset of vertices in fast memory.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 14:  # Too large for exact search
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

def _pebble_recursive(dag, budget):
    """
    Recursive pebbling strategy for layered DAGs.
    Uses divide-and-conquer to minimize recomputation.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    
    topo, children = _topo_order(dag)
    
    # Compute depth
    depth = {}
    for v in topo:
        ps = [p for p in parents.get(v, []) if p in vset]
        if not ps:
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in ps)
    
    max_depth = max(depth.values()) if depth else 0
    
    # Group vertices by depth
    layers = [[] for _ in range(max_depth + 1)]
    for v in vertices:
        layers[depth[v]].append(v)
    
    # Identify needed vertices (have path to output)
    needed = set(outputs)
    for v in reversed(topo):
        if v in needed:
            continue
        for c in children.get(v, []):
            if c in needed:
                needed.add(v)
                break
    
    # Filter to needed vertices only
    layers = [[v for v in layers[d] if v in needed] for d in range(len(layers))]
    
    # Recursive strategy: process layers in a divide-and-conquer fashion
    # For a chain of layers, we can use a strategy similar to pebbling a chain
    # where we keep some layers in memory and recompute others
    
    ops = []
    
    def process_layer_range(low, high, mem_available):
        """
        Process layers from low to high with mem_available memory slots.
        Returns list of operations.
        """
        if low > high:
            return []
        
        # If we can fit all layers in memory, just process them directly
        total_vertices = sum(len(layers[d]) for d in range(low, high + 1))
        if total_vertices <= mem_available:
            # Process directly
            result_ops = []
            loaded = set()
            for d in range(low, high + 1):
                for v in layers[d]:
                    ps = [p for p in parents.get(v, []) if p in vset and p in needed]
                    for p in ps:
                        if p not in loaded:
                            result_ops.append(["load", p])
                            loaded.add(p)
                    result_ops.append(["compute", v])
                    loaded.add(v)
                    if v in outset:
                        result_ops.append(["store", v])
                        loaded.discard(v)
                    # Delete parents that are no longer needed
                    for p in ps:
                        still_needed = False
                        for c in children.get(p, []):
                            if c != v and c in needed and depth.get(c, 0) >= d:
                                still_needed = True
                                break
                        if not still_needed and p not in outset and p in loaded:
                            result_ops.append(["delete", p])
                            loaded.discard(p)
            return result_ops
        
        # Divide and conquer
        mid = (low + high) // 2
        
        # Process first half
        ops1 = process_layer_range(low, mid, mem_available)
        
        # Process second half - may need to recompute some values
        ops2 = process_layer_range(mid + 1, high, mem_available)
        
        return ops1 + ops2
    
    all_ops = process_layer_range(0, max_depth, budget)
    return all_ops[:20000]

def _pebble_hybrid(dag, budget):
    """
    Hybrid strategy: use optimal for small, recursive for medium, greedy for large.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    
    if n <= 14:
        result = _pebble_optimal_small(dag, budget)
        if result is not None and len(result) <= 20000:
            return result
    
    if n <= 50:
        result = _pebble_recursive(dag, budget)
        if result is not None and len(result) <= 20000:
            return result
    
    return _pebble_greedy_optimized(dag, budget)

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50  # Safety margin before 60s limit
    try:
        result = _pebble_hybrid(dag, fast_memory_budget)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    
    # Fallback: simple greedy
    try:
        result = _pebble_greedy_optimized(dag, fast_memory_budget)
        if result is not None and len(result) <= 20000:
            return result
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
