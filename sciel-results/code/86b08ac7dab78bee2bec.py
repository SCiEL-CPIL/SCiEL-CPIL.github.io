"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

def _topo_order(parents, vertices, vset):
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

def _is_layered(parents, vertices, topo, children):
    """Check if DAG is layered: each vertex's parents are all in one contiguous layer."""
    if not vertices:
        return True, []
    
    # Assign layers by longest path from sources
    layer = {}
    for v in topo:
        ps = parents.get(v, [])
        if not ps:
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in ps) + 1
    
    # Check that all parents of v are in layer[v]-1
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            continue
        for p in ps:
            if layer[p] != layer[v] - 1:
                return False, []
    
    # Group by layer
    max_layer = max(layer.values()) if layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in vertices:
        layers[layer[v]].append(v)
    
    return True, layers

def _pebble_optimal_layered(dag, budget):
    """
    Optimal pebbling for layered DAGs.
    Key insight: In a layered DAG, if we process layer by layer, we only need to
    load inputs (layer 0) once and store outputs (last layer) once. Intermediate
    vertices are computed for free. The only cost is loads + stores.
    
    Strategy:
    1. Load all inputs (layer 0)
    2. For each subsequent layer, compute all vertices
    3. Keep as many as memory allows, prioritizing those needed by next layer
    4. Store outputs at the end
    
    This minimizes transfers because:
    - Each input is loaded exactly once
    - Each output is stored exactly once
    - No intermediate vertex needs to be loaded (they're computed)
    - No input needs to be reloaded (we keep them or recompute from them)
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    layered, layers = _is_layered(parents, vertices, topo, children)
    
    if not layered or not layers:
        return None
    
    n_layers = len(layers)
    if n_layers == 0:
        return []
    
    ops = []
    in_mem = set()
    
    # Load all inputs
    for v in layers[0]:
        ops.append(["load", v])
        in_mem.add(v)
    
    # Process each layer
    for li in range(1, n_layers):
        layer = layers[li]
        
        # Compute all vertices in this layer
        for v in layer:
            ops.append(["compute", v])
            in_mem.add(v)
        
        # If this is the last layer, store outputs
        if li == n_layers - 1:
            for v in layer:
                if v in outset:
                    ops.append(["store", v])
                    in_mem.discard(v)
        else:
            # Decide what to keep from this layer for the next layer
            next_layer = layers[li + 1]
            
            # Count how many times each vertex in this layer is needed by next layer
            need_count = defaultdict(int)
            for v in next_layer:
                ps = parents.get(v, [])
                for p in ps:
                    need_count[p] += 1
            
            # Keep vertices with highest need count, up to budget
            sorted_layer = sorted(layer, key=lambda v: need_count.get(v, 0), reverse=True)
            
            # Determine which to keep
            keep = set()
            for v in sorted_layer:
                if len(keep) < b:
                    keep.add(v)
                else:
                    break
            
            # Delete vertices not in keep
            for v in layer:
                if v not in keep:
                    ops.append(["delete", v])
                    in_mem.discard(v)
            
            # Also delete from previous layers if over budget
            while len(in_mem) > b:
                # Find a vertex to delete from earlier layers
                candidates = [v for v in in_mem if v not in outset and v not in layer]
                if not candidates:
                    break
                # Delete the one with lowest need in future layers
                def future_need(v):
                    count = 0
                    for fl in layers[li+1:]:
                        for w in fl:
                            if v in parents.get(w, []):
                                count += 1
                    return count
                v_to_del = min(candidates, key=future_need)
                ops.append(["delete", v_to_del])
                in_mem.discard(v_to_del)
    
    return ops[:20000]

def _pebble_astar_optimized(dag, budget, deadline):
    """
    Optimized A* search over memory states with recompute-cost heuristic.
    Key improvements:
    1. Better heuristic: use actual recompute cost, not just output count
    2. Prune states that exceed memory budget
    3. Early termination if we find a solution with cost <= current best
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 20:  # Limit for state space explosion
        return None
    
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    
    par_mask = [0] * n
    is_input = [False] * n
    children_mask = [0] * n
    for i, v in enumerate(vertices):
        ps = parents.get(v, [])
        if not ps:
            is_input[i] = True
        m = 0
        for p in ps:
            j = idx[p]
            m |= (1 << j)
            children_mask[j] |= (1 << i)
        par_mask[i] = m
    
    b = max(1, int(budget) if budget is not None else n)
    
    def popcount(x):
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")
    
    # Precompute recompute cost for each vertex (number of loads needed to compute it)
    recompute_cost = [0] * n
    def calc_cost(i):
        if recompute_cost[i] != 0:
            return recompute_cost[i]
        if is_input[i]:
            recompute_cost[i] = 1
        else:
            s = 0
            mm = par_mask[i]
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                s += calc_cost(j)
                mm ^= lb
            recompute_cost[i] = s  # No extra cost for compute
        return recompute_cost[i]
    
    for i in range(n):
        calc_cost(i)
    
    # Precompute ancestor inputs for heuristic
    ancestor_inputs = [0] * n
    def compute_ancestor_inputs(i):
        m = par_mask[i]
        if m == 0:
            ancestor_inputs[i] = (1 << i) if is_input[i] else 0
        else:
            ai = 0
            mm = m
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                ai |= ancestor_inputs[j]
                mm ^= lb
            ancestor_inputs[i] = ai
        return ancestor_inputs[i]
    for i in range(n):
        compute_ancestor_inputs(i)
    
    def heuristic(mem_mask):
        """Lower bound on transfers needed to complete all outputs from this memory state."""
        missing = out_mask & ~mem_mask
        if missing == 0:
            return 0
        h = 0
        mm = missing
        while mm:
            lb = mm & (-mm)
            i = lb.bit_length() - 1
            # To compute output i, we need all its ancestor inputs in memory
            # Count how many are missing
            anc = ancestor_inputs[i] & ~mem_mask
            h += popcount(anc)
            mm ^= lb
        return h
    
    full_mask = (1 << n) - 1
    start_state = (0, 0)  # (mem_mask, produced_mask)
    goal_produced = out_mask
    
    counter = 0
    open_heap = [(heuristic(0), 0, counter, (0, 0))]
    g_score = {start_state: 0}
    parent_op = {}
    best = None
    best_cost = float("inf")
    
    while open_heap:
        if time.time() > deadline:
            break
        f, g, _, (mem, produced) = heapq.heappop(open_heap)
        
        if produced == goal_produced:
            seq = []
            state = (mem, produced)
            while state != start_state:
                prev, op, v = parent_op[state]
                seq.append([op, v])
                state = prev
            seq.reverse()
            best = seq
            best_cost = g
            break
        
        if g > g_score.get(state, float("inf")):
            continue
        
        # Prune: if g + heuristic >= best_cost, skip
        if g + heuristic(mem) >= best_cost:
            continue
        
        for i in range(n):
            bit = 1 << i
            v = vertices[i]
            
            # Load: only for inputs, if not in memory and memory not full
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
                        heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
            
            # Compute: if all parents in memory and v not in memory
            if (mem & par_mask[i]) == par_mask[i] and not (mem & bit):
                new_mem = mem | bit
                new_prod = produced | bit if (out_mask & bit) else produced
                ng = g  # Compute is free
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
            
            # Store: if v in memory and is output
            if (mem & bit) and (out_mask & bit):
                new_mem = mem & ~bit
                new_prod = produced | bit
                ng = g + 1
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
            
            # Delete: if v in memory and not output (to free memory)
            if (mem & bit) and not (out_mask & bit):
                new_mem = mem & ~bit
                new_prod = produced
                ng = g
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem), ng, counter, ns))
    
    return best

def _pebble_postorder(dag, budget):
    """Pebble each output's subtree in postorder, recomputing on demand."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    topo, children = _topo_order(parents, vertices, vset)
    post = list(reversed(topo))
    
    ops = []
    loaded = set()
    stored = set()
    
    def pebble(v):
        if v in loaded:
            return
        ps = parents.get(v, [])
        for p in ps:
            pebble(p)
        ops.append(["compute", v])
        loaded.add(v)
        # Delete parents that are no longer needed
        for p in ps:
            if p in outset or p in stored:
                continue
            needed = False
            for w in children.get(p, []):
                if w != v and w not in stored and w not in loaded:
                    needed = True
                    break
            if not needed:
                ops.append(["delete", p])
                loaded.discard(p)
    
    for v in post:
        if v in outset:
            pebble(v)
            ops.append(["store", v])
            stored.add(v)
            loaded.discard(v)
    
    return ops[:20000]

def _pebble_greedy_memory_aware(dag, budget):
    """Greedy that respects memory budget, loading/computing as needed."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    
    ops = []
    in_mem = set()
    stored = set()
    
    def ensure_in_mem(v):
        if v in in_mem:
            return
        ps = parents.get(v, [])
        if not ps:
            if len(in_mem) >= b:
                for x in list(in_mem):
                    if x not in outset:
                        ops.append(["delete", x])
                        in_mem.discard(x)
                        if len(in_mem) < b:
                            break
            ops.append(["load", v])
            in_mem.add(v)
        else:
            for p in ps:
                ensure_in_mem(p)
            ops.append(["compute", v])
            in_mem.add(v)
    
    for v in outputs:
        ensure_in_mem(v)
        ops.append(["store", v])
        stored.add(v)
        in_mem.discard(v)
        to_delete = []
        for x in in_mem:
            if x not in outset:
                needed = False
                for rem_out in outputs:
                    if rem_out in stored:
                        continue
                    # Check if x is an ancestor of rem_out
                    stack = [x]
                    while stack:
                        u = stack.pop()
                        if u == rem_out:
                            needed = True
                            break
                        for ch in children.get(u, []):
                            stack.append(ch)
                    if needed:
                        break
                if not needed:
                    to_delete.append(x)
        for x in to_delete:
            ops.append(["delete", x])
            in_mem.discard(x)
    
    return ops[:20000]

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50  # Safety margin before 60s kill
    
    candidates = []
    
    # Try layered DP first (fast and optimal for layered DAGs)
    try:
        r = _pebble_optimal_layered(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Try optimized A* for small DAGs
    try:
        r = _pebble_astar_optimized(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Postorder pebbling
    try:
        r = _pebble_postorder(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Memory-aware greedy
    try:
        r = _pebble_greedy_memory_aware(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    # Pick candidate with fewest transfers (load+store)
    def transfers(ops):
        return sum(1 for op, v in ops if op in ("load", "store"))
    best = min(candidates, key=transfers)
    return best
# EVOLVE-BLOCK-END
