"""Optimized DAG pebbling with improved memory management and recomputation strategies."""
import time
import heapq
from collections import deque

def _topo_order(parents, vertices, vset):
    """Compute topological order and children map."""
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
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
        topo = list(vertices)
    return topo, children

def _compute_recompute_costs(parents, vertices):
    """Compute minimum compute operations to produce each vertex."""
    cost = {}
    def get_cost(v):
        if v in cost:
            return cost[v]
        ps = parents.get(v, [])
        if not ps:
            cost[v] = 1
        else:
            cost[v] = 1 + sum(get_cost(p) for p in ps)
        return cost[v]
    for v in vertices:
        get_cost(v)
    return cost

def _get_ancestors(parents, v):
    """Get all ancestors of vertex v."""
    result = set()
    stack = [v]
    while stack:
        u = stack.pop()
        for p in parents.get(u, []):
            if p not in result:
                result.add(p)
                stack.append(p)
    result.discard(v)
    return result

def _get_descendants(children, v):
    """Get all descendants of vertex v."""
    result = set()
    stack = [v]
    while stack:
        u = stack.pop()
        for c in children.get(u, []):
            if c not in result:
                result.add(c)
                stack.append(c)
    result.discard(v)
    return result

def _validate_ops(dag, ops, budget):
    """Validate operations are correct and within constraints."""
    if not ops:
        return False
    if len(ops) > 20000:
        return False
    
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else len(outputs))
    
    in_mem = set()
    stored = set()
    
    for op, v in ops:
        if op == "load":
            if v in in_mem:
                return False
            if len(in_mem) >= b:
                return False
            in_mem.add(v)
        elif op == "store":
            if v not in in_mem:
                return False
            in_mem.discard(v)
            stored.add(v)
        elif op == "compute":
            ps = parents.get(v, [])
            for p in ps:
                if p not in in_mem and p not in stored:
                    return False
            if v in in_mem:
                return False
            if len(in_mem) >= b:
                return False
            in_mem.add(v)
        elif op == "delete":
            if v not in in_mem:
                return False
            in_mem.discard(v)
        else:
            return False
    
    for o in outset:
        if o not in stored:
            return False
    
    return True

def _pebble_optimal_layered(dag, budget, deadline):
    """
    Optimal pebbling for layered DAGs using dynamic programming.
    
    Key insight: For layered DAGs, we can process layer by layer,
    keeping only the necessary boundary vertices in memory.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    n = len(vertices)
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)
    
    topo, children = _topo_order(parents, vertices, vset)
    
    # Compute layers
    layer_of = {}
    for v in topo:
        ps = parents.get(v, [])
        if not ps:
            layer_of[v] = 0
        else:
            layer_of[v] = 1 + max(layer_of[p] for p in ps)
    
    max_layer = max(layer_of.values()) if layer_of else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in vertices:
        layers[layer_of[v]].append(v)
    
    # For each layer, determine which vertices are "boundary" (needed by future layers)
    # A vertex in layer L is boundary if it has children in layer > L
    layer_boundary = [set() for _ in range(max_layer + 1)]
    for v in vertices:
        for child in children.get(v, []):
            if layer_of[child] > layer_of[v]:
                layer_boundary[layer_of[v]].add(v)
    
    # Also track which vertices in each layer are outputs
    layer_outputs = [set() for _ in range(max_layer + 1)]
    for v in outset:
        layer_outputs[layer_of[v]].add(v)
    
    # Process layers in order
    ops = []
    in_mem = set()
    stored = set()
    
    # Precompute importance: how many future uses a vertex has
    future_uses = {v: 0 for v in vertices}
    for v in vertices:
        for child in children.get(v, []):
            future_uses[v] += 1
            # Also count grandchildren for better estimation
            for gc in children.get(child, []):
                future_uses[v] += 0.5
    
    def get_importance(v):
        """Get importance score for memory management."""
        if v in outset:
            return float('inf')  # Outputs must be kept until stored
        return future_uses[v]
    
    def ensure_space():
        """Ensure at least one slot is available in memory."""
        if len(in_mem) < b:
            return
        # Find least important vertex to delete
        candidates = [v for v in in_mem if v not in outset]
        if candidates:
            to_delete = min(candidates, key=get_importance)
            ops.append(["delete", to_delete])
            in_mem.discard(to_delete)
    
    # Process each layer
    for L in range(max_layer + 1):
        if time.time() > deadline:
            break
        
        layer_vertices = layers[L]
        
        # First, ensure all parents of vertices in this layer are in memory
        for v in layer_vertices:
            ps = parents.get(v, [])
            for p in ps:
                if p not in in_mem and p not in stored:
                    # Need to load or recompute p
                    if not parents.get(p, []):
                        # p is an input
                        ensure_space()
                        ops.append(["load", p])
                        in_mem.add(p)
                    else:
                        # p needs to be recomputed
                        # First ensure p's parents are in memory
                        pps = parents.get(p, [])
                        for pp in pps:
                            if pp not in in_mem and pp not in stored:
                                if not parents.get(pp, []):
                                    ensure_space()
                                    ops.append(["load", pp])
                                    in_mem.add(pp)
                                else:
                                    # Recursively ensure pp's parents
                                    ppps = parents.get(pp, [])
                                    for ppp in ppps:
                                        if ppp not in in_mem and ppp not in stored:
                                            if not parents.get(ppp, []):
                                                ensure_space()
                                                ops.append(["load", ppp])
                                                in_mem.add(ppp)
                                    if all(ppp in in_mem or ppp in stored for ppp in ppps):
                                        ensure_space()
                                        ops.append(["compute", pp])
                                        in_mem.add(pp)
                        if all(pp in in_mem or pp in stored for pp in pps):
                            ensure_space()
                            ops.append(["compute", p])
                            in_mem.add(p)
        
        # Now compute all vertices in this layer
        for v in layer_vertices:
            ps = parents.get(v, [])
            if all(p in in_mem or p in stored for p in ps):
                ensure_space()
                ops.append(["compute", v])
                in_mem.add(v)
        
        # After computing layer L, decide what to keep
        # Keep: boundary vertices (needed by future layers) and outputs in this layer
        # Delete: everything else
        
        # First, store outputs in this layer
        for v in layer_outputs[L]:
            if v in in_mem:
                ops.append(["store", v])
                in_mem.discard(v)
                stored.add(v)
        
        # Delete non-boundary, non-output vertices
        to_delete = set()
        for v in in_mem:
            if layer_of.get(v, -1) < L:  # From earlier layers
                if v not in layer_boundary[layer_of[v]] and v not in outset:
                    to_delete.add(v)
            elif layer_of.get(v, -1) == L:
                if v not in layer_boundary[L] and v not in outset:
                    to_delete.add(v)
        
        for v in to_delete:
            ops.append(["delete", v])
            in_mem.discard(v)
    
    # Final cleanup: store any remaining outputs
    for v in outset:
        if v in in_mem and v not in stored:
            ops.append(["store", v])
            in_mem.discard(v)
            stored.add(v)
    
    return ops[:20000]

def _pebble_hierarchical(dag, budget, deadline):
    """
    Hierarchical pebbling: recursively split the DAG and optimize each part.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    n = len(vertices)
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)
    
    topo, children = _topo_order(parents, vertices, vset)
    
    # Compute layers
    layer_of = {}
    for v in topo:
        ps = parents.get(v, [])
        if not ps:
            layer_of[v] = 0
        else:
            layer_of[v] = 1 + max(layer_of[p] for p in ps)
    
    max_layer = max(layer_of.values()) if layer_of else 0
    
    # Recursive pebbling with optimal split points
    def pebble_range(layer_start, layer_end, mem_budget):
        """Pebble layers from layer_start to layer_end."""
        if layer_start > layer_end:
            return [], set(), set()
        if time.time() > deadline:
            return [], set(), set()
        
        # Find optimal split point
        best_split = (layer_start + layer_end) // 2
        best_cost = float('inf')
        
        for split in range(layer_start, layer_end + 1):
            # Compute boundary size between left and right
            left_boundary = set()
            for v in vertices:
                if layer_start <= layer_of[v] <= split:
                    for child in children.get(v, []):
                        if split < layer_of[child] <= layer_end:
                            left_boundary.add(v)
            
            # Also count outputs in left part that need to be stored
            left_outputs = {v for v in outset if layer_start <= layer_of[v] <= split}
            
            boundary_size = len(left_boundary) + len(left_outputs)
            
            # Cost estimate: boundary_size * 2 (store + load) + recursive costs
            cost = boundary_size * 2
            if cost < best_cost:
                best_cost = cost
                best_split = split
        
        split = best_split
        
        # Pebble left half
        left_ops, left_in, left_out = pebble_range(layer_start, split, mem_budget)
        
        # Store boundary vertices from left that are needed by right
        store_ops = []
        boundary_to_store = set()
        for v in vertices:
            if layer_start <= layer_of[v] <= split:
                for child in children.get(v, []):
                    if split < layer_of[child] <= layer_end:
                        boundary_to_store.add(v)
        
        for v in sorted(boundary_to_store, key=lambda x: (x if isinstance(x, int) else 0)):
            if v in left_out:
                store_ops.append(["store", v])
        
        # Pebble right half
        right_ops, right_in, right_out = pebble_range(split + 1, layer_end, mem_budget)
        
        # Load boundary vertices needed by right
        load_ops = []
        for v in sorted(boundary_to_store, key=lambda x: (x if isinstance(x, int) else 0)):
            load_ops.append(["load", v])
        
        all_ops = left_ops + store_ops + load_ops + right_ops
        return all_ops, left_in, right_out
    
    all_ops, _, _ = pebble_range(0, max_layer, b)
    
    # Ensure all outputs are stored
    stored = set()
    for v in outset:
        if v not in stored:
            all_ops.append(["store", v])
            stored.add(v)
    
    return all_ops[:20000]

def _pebble_greedy_optimal(dag, budget):
    """
    Greedy pebbling with optimal memory management.
    Uses importance-based deletion to minimize recomputations.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    
    # Precompute: for each vertex, which future vertices need it
    needed_by = {v: set() for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            needed_by[p].add(v)
    
    # Compute importance: number of future uses (with depth weighting)
    def compute_importance(v, depth=0):
        if v in outset:
            return float('inf')
        imp = 1  # At least itself
        for w in needed_by[v]:
            imp += compute_importance(w, depth + 1) * (0.9 ** depth)
        return imp
    
    importance = {}
    for v in vertices:
        importance[v] = compute_importance(v)
    
    ops = []
    in_mem = set()
    stored = set()
    
    def ensure_space():
        if len(in_mem) < b:
            return
        candidates = [v for v in in_mem if v not in outset and v not in stored]
        if candidates:
            to_delete = min(candidates, key=lambda v: importance[v])
            ops.append(["delete", to_delete])
            in_mem.discard(to_delete)
    
    # Process vertices in topological order
    for v in topo:
        ps = parents.get(v, [])
        
        # Ensure all parents are in memory
        for p in ps:
            if p not in in_mem and p not in stored:
                # Need to load or recompute p
                if not parents.get(p, []):
                    ensure_space()
                    ops.append(["load", p])
                    in_mem.add(p)
                else:
                    # Recompute p
                    pps = parents.get(p, [])
                    for pp in pps:
                        if pp not in in_mem and pp not in stored:
                            if not parents.get(pp, []):
                                ensure_space()
                                ops.append(["load", pp])
                                in_mem.add(pp)
                            else:
                                ppps = parents.get(pp, [])
                                for ppp in ppps:
                                    if ppp not in in_mem and ppp not in stored:
                                        if not parents.get(ppp, []):
                                            ensure_space()
                                            ops.append(["load", ppp])
                                            in_mem.add(ppp)
                                if all(ppp in in_mem or ppp in stored for ppp in ppps):
                                    ensure_space()
                                    ops.append(["compute", pp])
                                    in_mem.add(pp)
                    if all(pp in in_mem or pp in stored for pp in pps):
                        ensure_space()
                        ops.append(["compute", p])
                        in_mem.add(p)
        
        # Compute v
        if all(p in in_mem or p in stored for p in ps):
            ensure_space()
            ops.append(["compute", v])
            in_mem.add(v)
        
        # Store if output
        if v in outset:
            ops.append(["store", v])
            in_mem.discard(v)
            stored.add(v)
        
        # Delete if not needed by future vertices
        elif not needed_by[v] or all(w in stored for w in needed_by[v]):
            ops.append(["delete", v])
            in_mem.discard(v)
    
    return ops[:20000]

def _pebble_astar_small(dag, budget, deadline):
    """A* search for small DAGs (n <= 15)."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 15:
        return None
    
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    
    par_mask = [0] * n
    is_input = [False] * n
    for i, v in enumerate(vertices):
        ps = parents.get(v, [])
        if not ps:
            is_input[i] = True
        m = 0
        for p in ps:
            m |= (1 << idx[p])
        par_mask[i] = m
    
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)
    
    def popcount(x):
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")
    
    # Heuristic: minimum transfers to complete
    def heuristic(mem_mask, produced_mask):
        missing = out_mask & ~produced_mask
        if missing == 0:
            return 0
        h = 0
        mm = missing
        while mm:
            lb = mm & (-mm)
            i = lb.bit_length() - 1
            # Need to compute this vertex and its missing ancestors
            anc = par_mask[i] & ~mem_mask
            h += popcount(anc) + 1
            mm ^= lb
        return h
    
    start_state = (0, 0)  # (mem_mask, produced_mask)
    counter = 0
    open_heap = [(heuristic(0, 0), 0, counter, start_state)]
    g_score = {start_state: 0}
    parent_op = {}
    best = None
    
    while open_heap:
        if time.time() > deadline:
            break
        
        f, g, _, (mem, produced) = heapq.heappop(open_heap)
        
        if produced == out_mask:
            # Reconstruct path
            seq = []
            state = (mem, produced)
            while state != start_state:
                prev, op, v = parent_op[state]
                seq.append([op, v])
                state = prev
            seq.reverse()
            best = seq
            break
        
        if g > g_score.get(state, float("inf")):
            continue
        
        for i in range(n):
            bit = 1 << i
            v = vertices[i]
            
            # Load input
            if is_input[i] and not (mem & bit):
                if popcount(mem) < b:
                    new_mem = mem | bit
                    new_prod = produced
                    ng = g + 1
                    ns = (new_mem, new_prod)
                    if ng < g_score.get(ns, float("inf")):
                        g_score[ns] = ng
                        parent_op[ns] = ((mem, produced), "load", v)
                        counter += 1
                        heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))
            
            # Compute
            if (mem & par_mask[i]) == par_mask[i] and not (mem & bit):
                if popcount(mem) < b:
                    new_mem = mem | bit
                    new_prod = produced | bit if (out_mask & bit) else produced
                    ng = g
                    ns = (new_mem, new_prod)
                    if ng < g_score.get(ns, float("inf")):
                        g_score[ns] = ng
                        parent_op[ns] = ((mem, produced), "compute", v)
                        counter += 1
                        heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))
            
            # Store output
            if (mem & bit) and (out_mask & bit):
                new_mem = mem & ~bit
                new_prod = produced | bit
                ng = g + 1
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))
            
            # Delete non-output
            if (mem & bit) and not (out_mask & bit):
                new_mem = mem & ~bit
                new_prod = produced
                ng = g
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))
    
    return best

def pebble(dag, fast_memory_budget):
    """
    Main pebbling function: tries multiple strategies and returns the best valid schedule.
    """
    deadline = time.time() + 45
    candidates = []
    
    # Strategy 1: Optimal layered pebbling
    try:
        r = _pebble_optimal_layered(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    # Strategy 2: Hierarchical pebbling
    try:
        r = _pebble_hierarchical(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    # Strategy 3: Greedy with importance-based memory management
    try:
        r = _pebble_greedy_optimal(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    # Strategy 4: A* for small DAGs
    try:
        r = _pebble_astar_small(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        # Fallback: simple postorder with recomputation
        parents = dag.get("parents", {})
        outputs = list(dag.get("outputs", []))
        vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
        vset = set(vertices)
        topo, children = _topo_order(parents, vertices, vset)
        post = list(reversed(topo))
        
        ops = []
        loaded = set()
        stored = set()
        
        def pebble_subtree(v):
            if v in loaded:
                return
            ps = parents.get(v, [])
            for p in ps:
                pebble_subtree(p)
            ops.append(["compute", v])
            loaded.add(v)
        
        for v in post:
            if v in set(outputs):
                pebble_subtree(v)
                ops.append(["store", v])
                stored.add(v)
                loaded.discard(v)
        
        if len(ops) <= 20000 and _validate_ops(dag, ops, fast_memory_budget):
            return ops
        return []
    
    # Return candidate with minimum transfers
    def transfers(ops):
        return sum(1 for op, v in ops if op in ("load", "store"))
    
    best = min(candidates, key=transfers)
    return best
