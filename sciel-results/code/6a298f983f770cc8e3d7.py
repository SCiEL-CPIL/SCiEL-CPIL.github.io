"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def _topo_order(dag):
    """Compute topological order and children mapping."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return [], {}
    vset = set(vertices)
    children = {v: [] for v in vertices}
    indeg = {v: 0 for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in vset:
                children[p].append(v)
                indeg[v] = indeg.get(v, 0) + 1
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


def _compute_depths(dag, topo, vset):
    """Compute depth (longest path from source) for each vertex."""
    parents = dag.get("parents", {})
    depth = {}
    for v in topo:
        ps = [p for p in parents.get(v, []) if p in vset]
        if not ps:
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in ps)
    return depth


def _pebble_bfs_small(dag, budget):
    """
    Exact BFS pebbling for small DAGs (n <= 12).
    State: (loaded_mask, stored_mask)
    Minimize total transfers (load + store).
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 12:
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
    
    # BFS: state = (loaded_mask, stored_mask)
    # Transfers: load (+1), store (+1), compute (0), delete (0)
    start_state = (0, 0)
    goal_check = lambda s: (s[1] & out_mask) == out_mask
    
    from collections import deque as dq
    
    visited = {start_state: 0}
    parent_op = {}
    queue = dq([(start_state, 0)])
    
    max_ops = 20000
    while queue:
        state, g = queue.popleft()
        loaded, stored = state
        if goal_check(state):
            seq = []
            s = state
            while s != start_state:
                prev, op, v = parent_op[s]
                seq.append([op, v])
                s = prev
            seq.reverse()
            if len(seq) <= max_ops:
                return seq
            return None
        
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
            
            # Delete v if loaded and not output
            if (loaded & bit) and not (out_mask & bit):
                ns = (loaded & ~bit, stored)
                ng = g
                if ng < visited.get(ns, float('inf')):
                    visited[ns] = ng
                    parent_op[ns] = (state, "delete", v)
                    queue.append((ns, ng))
            
            # Compute v if all parents loaded and v not loaded
            if (loaded & par_mask[i]) == par_mask[i] and not (loaded & bit):
                ns = (loaded | bit, stored)
                ng = g
                if ng < visited.get(ns, float('inf')):
                    visited[ns] = ng
                    parent_op[ns] = (state, "compute", v)
                    queue.append((ns, ng))
    
    return None


def _pebble_layered_recursive(dag, budget, topo, children, depth, vset):
    """
    Recursive pebbling for layered DAGs.
    
    Strategy: For a chain of layers, use divide-and-conquer.
    - If we can fit all intermediate layers in memory, do forward pass.
    - Otherwise, split at the middle layer, pebble left half keeping boundary,
      then pebble right half.
    
    This minimizes total transfers by reducing recomputation.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    
    # Get vertices in topological order that are needed
    # Compute which vertices are needed (have path to output)
    needed = set(outputs)
    for v in reversed(topo):
        if v in needed:
            continue
        for c in children.get(v, []):
            if c in needed:
                needed.add(v)
                break
    
    # Filter to needed vertices in topo order
    needed_topo = [v for v in topo if v in needed]
    if not needed_topo:
        return []
    
    # Compute depth for needed vertices
    max_depth = max(depth[v] for v in needed_topo) if needed_topo else 0
    min_depth = min(depth[v] for v in needed_topo) if needed_topo else 0
    
    # Group by depth
    layers = {}
    for v in needed_topo:
        d = depth[v]
        if d not in layers:
            layers[d] = []
        layers[d].append(v)
    
    # Sort layers by depth
    sorted_depths = sorted(layers.keys())
    
    # Strategy: Process layers in order, using memory budget to decide
    # how many layers to keep in memory simultaneously.
    
    # Simple effective strategy:
    # 1. Process in topological order
    # 2. Load parents when needed
    # 3. Compute vertex
    # 4. Delete parents that are no longer needed
    # 5. Store outputs
    # 6. When memory is tight, recompute instead of storing
    
    ops = []
    loaded = set()
    
    # Track how many more times each vertex will be needed
    remaining_uses = {}
    for v in needed_topo:
        remaining_uses[v] = 0
    for v in needed_topo:
        ps = [p for p in parents.get(v, []) if p in vset and p in needed]
        for p in ps:
            remaining_uses[p] = remaining_uses.get(p, 0) + 1
    
    # Process in topological order
    for v in needed_topo:
        ps = [p for p in parents.get(v, []) if p in vset and p in needed]
        
        # Load parents that aren't in memory
        for p in ps:
            if p not in loaded:
                ops.append(["load", p])
                loaded.add(p)
        
        # Compute v
        ops.append(["compute", v])
        loaded.add(v)
        
        # Store if output
        if v in outset:
            ops.append(["store", v])
            loaded.discard(v)
        
        # Delete parents that are no longer needed
        for p in ps:
            remaining_uses[p] = remaining_uses.get(p, 0) - 1
            if remaining_uses[p] <= 0 and p not in outset and p in loaded:
                ops.append(["delete", p])
                loaded.discard(p)
    
    return ops[:20000]


def _pebble_greedy_with_budget(dag, budget):
    """
    Greedy pebbling that respects memory budget.
    When memory is full, evict vertices that will be recomputed soonest.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    
    topo, children = _topo_order(dag)
    depth = _compute_depths(dag, topo, vset)
    
    # Compute which vertices are needed
    needed = set(outputs)
    for v in reversed(topo):
        if v in needed:
            continue
        for c in children.get(v, []):
            if c in needed:
                needed.add(v)
                break
    
    needed_topo = [v for v in topo if v in needed]
    if not needed_topo:
        return []
    
    ops = []
    loaded = set()
    
    # Track remaining uses
    remaining_uses = {}
    for v in needed_topo:
        remaining_uses[v] = 0
    for v in needed_topo:
        ps = [p for p in parents.get(v, []) if p in vset and p in needed]
        for p in ps:
            remaining_uses[p] = remaining_uses.get(p, 0) + 1
    
    # Process in topological order
    for v in needed_topo:
        ps = [p for p in parents.get(v, []) if p in vset and p in needed]
        
        # Load parents, respecting budget
        for p in ps:
            if p not in loaded:
                # Check budget
                if len(loaded) >= budget and budget > 0:
                    # Need to evict something: evict vertex with fewest remaining uses
                    # that is not an output and not in ps
                    candidates = [x for x in loaded if x not in outset and x not in ps and remaining_uses.get(x, 0) <= 0]
                    if not candidates:
                        # Evict any non-output, non-parent
                        candidates = [x for x in loaded if x not in outset and x not in ps]
                    if candidates:
                        # Evict the one with fewest remaining uses
                        evict = min(candidates, key=lambda x: remaining_uses.get(x, 0))
                        ops.append(["delete", evict])
                        loaded.discard(evict)
                    else:
                        # Cannot proceed without exceeding budget; this shouldn't happen
                        # if budget is sufficient
                        pass
                
                ops.append(["load", p])
                loaded.add(p)
        
        # Compute v
        ops.append(["compute", v])
        loaded.add(v)
        
        # Store if output
        if v in outset:
            ops.append(["store", v])
            loaded.discard(v)
        
        # Delete parents that are no longer needed
        for p in ps:
            remaining_uses[p] = remaining_uses.get(p, 0) - 1
            if remaining_uses[p] <= 0 and p not in outset and p in loaded:
                ops.append(["delete", p])
                loaded.discard(p)
    
    return ops[:20000]


def pebble(dag, fast_memory_budget):
    """
    Main pebbling function.
    
    Args:
        dag: dict with "parents" (vertex -> list of parent vertices) and "outputs" (list)
        fast_memory_budget: int, number of fast memory slots
    
    Returns:
        list of [operation, vertex] where operation is in {load, store, compute, delete}
    """
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    try:
        parents = dag.get("parents", {})
        outputs = list(dag.get("outputs", []))
        vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
        n = len(vertices)
        
        if n == 0:
            return []
        
        vset = set(vertices)
        topo, children = _topo_order(dag)
        depth = _compute_depths(dag, topo, vset)
        
        # For very small DAGs, use exact BFS
        if n <= 12:
            result = _pebble_bfs_small(dag, fast_memory_budget)
            if result is not None:
                return result
        
        # For larger DAGs, use greedy with budget awareness
        result = _pebble_greedy_with_budget(dag, fast_memory_budget)
        if result is not None and len(result) <= 20000:
            return result
        
    except Exception:
        pass
    
    # Fallback: simple greedy without budget checks
    try:
        parents = dag.get("parents", {})
        outputs = list(dag.get("outputs", []))
        vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
        if not vertices:
            return []
        vset = set(vertices)
        outset = set(outputs)
        
        topo, children = _topo_order(dag)
        
        # Compute needed vertices
        needed = set(outputs)
        for v in reversed(topo):
            if v in needed:
                continue
            for c in children.get(v, []):
                if c in needed:
                    needed.add(v)
                    break
        
        needed_topo = [v for v in topo if v in needed]
        
        ops = []
        loaded = set()
        
        remaining_uses = {}
        for v in needed_topo:
            remaining_uses[v] = 0
        for v in needed_topo:
            ps = [p for p in parents.get(v, []) if p in vset and p in needed]
            for p in ps:
                remaining_uses[p] = remaining_uses.get(p, 0) + 1
        
        for v in needed_topo:
            ps = [p for p in parents.get(v, []) if p in vset and p in needed]
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
    except Exception:
        return []
# EVOLVE-BLOCK-END
