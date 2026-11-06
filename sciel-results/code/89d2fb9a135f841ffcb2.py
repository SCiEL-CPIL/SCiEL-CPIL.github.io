"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def _topo_order(parents, vertices):
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
    cost = {}
    def get_cost(v):
        if v in cost:
            return cost[v]
        ps = parents.get(v, [])
        if not ps:
            cost[v] = 1
        else:
            c = 1 + sum(get_cost(p) for p in ps)
            cost[v] = c
        return cost[v]
    for v in vertices:
        get_cost(v)
    return cost

def _get_ancestors(parents, v, memo=None):
    if memo is None:
        memo = {}
    if v in memo:
        return memo[v]
    ps = parents.get(v, [])
    if not ps:
        memo[v] = {v}
        return memo[v]
    anc = set()
    for p in ps:
        anc |= _get_ancestors(parents, p, memo)
    anc.add(v)
    memo[v] = anc
    return anc

def _get_descendants(children, v, memo=None):
    if memo is None:
        memo = {}
    if v in memo:
        return memo[v]
    ch = children.get(v, [])
    if not ch:
        memo[v] = {v}
        return memo[v]
    desc = set()
    for c in ch:
        desc |= _get_descendants(children, c, memo)
    desc.add(v)
    memo[v] = desc
    return memo

def _is_layered(parents, vertices, topo):
    """Check if DAG is layered: all edges go from layer i to layer i+1."""
    depth = {}
    for v in topo:
        ps = parents.get(v, [])
        if not ps:
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth[p] for p in ps)
    # Check all edges span exactly one layer
    for v in vertices:
        ps = parents.get(v, [])
        for p in ps:
            if depth[v] != depth[p] + 1:
                return False
    return True

def _recursive_pebble_layered(parents, children, vertices, outputs, budget, deadline, depth_map=None):
    """
    Recursive pebbling for layered DAGs.
    For a layered DAG with layers L0, L1, ..., Lk:
    - To compute a vertex in layer i, we need its parents in layer i-1.
    - Strategy: process layer by layer, keeping only what's needed.
    - For each layer, compute all vertices, then keep only those needed by next layer.
    """
    if depth_map is None:
        depth_map = {}
        for v in vertices:
            ps = parents.get(v, [])
            if not ps:
                depth_map[v] = 0
            else:
                depth_map[v] = 1 + max(depth_map[p] for p in ps)
    
    max_depth = max(depth_map.values()) if depth_map else 0
    
    # Group vertices by depth
    layers = {}
    for v in vertices:
        d = depth_map[v]
        if d not in layers:
            layers[d] = []
        layers[d].append(v)
    
    # Sort vertices within each layer for determinism
    for d in layers:
        layers[d].sort(key=lambda x: x if isinstance(x, int) else str(x))
    
    outset = set(outputs)
    ops = []
    in_mem = set()
    stored = set()
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    # For each layer, determine which vertices from this layer are needed by the next layer
    # A vertex v in layer d is needed if some vertex in layer d+1 has v as parent
    needed_by_next = {}
    for d in range(max_depth + 1):
        needed_by_next[d] = set()
        if d + 1 <= max_depth:
            for v in layers.get(d + 1, []):
                for p in parents.get(v, []):
                    if depth_map.get(p, -1) == d:
                        needed_by_next[d].add(p)
    
    # Process layers in order
    for d in range(max_depth + 1):
        if time.time() > deadline:
            break
        
        layer = layers.get(d, [])
        if not layer:
            continue
        
        # First, make sure all parents of this layer's vertices are in memory
        # They should be from the previous layer (or inputs)
        
        # Compute all vertices in this layer
        for v in layer:
            if time.time() > deadline:
                break
            
            ps = parents.get(v, [])
            
            # Ensure parents are in memory (they should be from previous layer)
            for p in ps:
                if p not in in_mem:
                    # Need to load or recompute p
                    if p in stored:
                        if len(in_mem) >= b:
                            # Delete something we can
                            for x in list(in_mem):
                                if x not in outset and x not in needed_by_next.get(d - 1, set()):
                                    ops.append(["delete", x])
                                    in_mem.discard(x)
                                    if len(in_mem) < b:
                                        break
                        ops.append(["load", p])
                        in_mem.add(p)
                    else:
                        # p is an input (no parents) - load it
                        pps = parents.get(p, [])
                        if not pps:
                            if len(in_mem) >= b:
                                for x in list(in_mem):
                                    if x not in outset:
                                        ops.append(["delete", x])
                                        in_mem.discard(x)
                                        if len(in_mem) < b:
                                            break
                            ops.append(["load", p])
                            in_mem.add(p)
                        else:
                            # Should not happen in layered DAG, but handle gracefully
                            # Recursively ensure p is available
                            for pp in pps:
                                if pp not in in_mem:
                                    ppps = parents.get(pp, [])
                                    if not ppps:
                                        if len(in_mem) >= b:
                                            for x in list(in_mem):
                                                if x not in outset:
                                                    ops.append(["delete", x])
                                                    in_mem.discard(x)
                                                    if len(in_mem) < b:
                                                        break
                                        ops.append(["load", pp])
                                        in_mem.add(pp)
                                    else:
                                        if len(in_mem) >= b:
                                            for x in list(in_mem):
                                                if x not in outset:
                                                    ops.append(["delete", x])
                                                    in_mem.discard(x)
                                                    if len(in_mem) < b:
                                                        break
                                        ops.append(["load", pp])
                                        in_mem.add(pp)
                            ops.append(["compute", p])
                            in_mem.add(p)
            
            # Compute v
            if len(in_mem) >= b and v not in in_mem:
                # Make space
                for x in list(in_mem):
                    if x not in outset and x not in needed_by_next.get(d, set()):
                        ops.append(["delete", x])
                        in_mem.discard(x)
                        if len(in_mem) < b:
                            break
            
            ops.append(["compute", v])
            in_mem.add(v)
        
        # After computing this layer, decide what to keep
        # Keep: outputs in this layer, vertices needed by next layer
        keep = set()
        for v in layer:
            if v in outset:
                # Store output
                if v in in_mem:
                    ops.append(["store", v])
                    stored.add(v)
                    in_mem.discard(v)
            elif d < max_depth and v in needed_by_next[d]:
                keep.add(v)
        
        # Delete vertices not needed
        for v in layer:
            if v in in_mem and v not in keep and v not in outset:
                ops.append(["delete", v])
                in_mem.discard(v)
    
    return ops[:20000]

def _pebble_astar(dag, budget, deadline):
    """
    A* search over memory bitmasks with improved heuristic.
    For small DAGs (n <= 16), finds near-optimal solutions.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 16:  # Limit to 16 vertices for performance
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
    
    # Compute recompute cost
    rcost = [0] * n
    def get_rcost(i):
        if rcost[i] > 0:
            return rcost[i]
        m = par_mask[i]
        if m == 0:
            rcost[i] = 1
        else:
            s = 1
            mm = m
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                s += get_rcost(j)
                mm ^= lb
            rcost[i] = s
        return rcost[i]
    for i in range(n):
        get_rcost(i)
    
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)
    
    def popcount(x):
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")
    
    # Precompute ancestor masks (all ancestors including self)
    ancestor_set = [0] * n
    for i in range(n):
        m = par_mask[i]
        if m == 0:
            ancestor_set[i] = (1 << i)
        else:
            ai = (1 << i)
            mm = m
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                ai |= ancestor_set[j]
                mm ^= lb
            ancestor_set[i] = ai
    
    # Precompute which inputs are needed for each output
    input_need = [0] * n
    for i in range(n):
        if out_mask & (1 << i):
            input_need[i] = 0
            mm = ancestor_set[i]
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                if is_input[j]:
                    input_need[i] |= lb
                mm ^= lb
    
    # Improved heuristic: estimate minimum transfers
    def heuristic(mem_mask, produced):
        missing = out_mask & ~produced
        if missing == 0:
            return 0
        
        h = 0
        mm = missing
        while mm:
            lb = mm & (-mm)
            i = lb.bit_length() - 1
            h += 1  # store
            # Need to load inputs not in memory
            anc_inputs = input_need[i] & ~mem_mask
            h += popcount(anc_inputs)
            mm ^= lb
        
        # Memory pressure: if we're at capacity, we'll need to delete and reload
        if popcount(mem_mask) >= b:
            h += 1
        
        return h
    
    start_state = (0, 0)
    goal_produced = out_mask
    
    counter = 0
    open_heap = [(heuristic(0, 0), 0, counter, (0, 0))]
    g_score = {start_state: 0}
    parent_op = {}
    best = None
    max_expansions = 50000
    
    expansions = 0
    while open_heap:
        if time.time() > deadline or expansions > max_expansions:
            break
        f, g, _, (mem, produced) = heapq.heappop(open_heap)
        expansions += 1
        
        if produced == goal_produced:
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
            if (mem & bit) and (out_mask & bit) and not (produced & bit):
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

def _verify_ops(ops, dag, budget):
    """Verify that the operations sequence is valid."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    in_mem = set()
    stored = set()
    computed = set()
    b = max(1, int(budget) if budget is not None else 10**9)
    
    for op, v in ops:
        if op == "load":
            if v in in_mem:
                return False
            if len(in_mem) >= b:
                return False
            # v must be an input or previously computed/stored
            if v not in computed and v not in stored:
                ps = parents.get(v, [])
                if ps:  # not an input
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
                if p not in in_mem:
                    return False
            if len(in_mem) >= b and v not in in_mem:
                return False
            in_mem.add(v)
            computed.add(v)
        elif op == "delete":
            if v not in in_mem:
                return False
            in_mem.discard(v)
        else:
            return False
    
    # Check all outputs are stored
    for o in outset:
        if o not in stored:
            return False
    
    return True

def _local_search_polish(ops, dag, budget, deadline):
    """Try to reduce transfers by local search: swap adjacent operations where possible."""
    if not ops or len(ops) > 500:
        return ops
    
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else 10**9)
    
    def transfers(ops_list):
        return sum(1 for op, v in ops_list if op in ("load", "store"))
    
    best_ops = ops
    best_score = transfers(ops)
    
    # Try removing redundant loads/stores
    # For each pair of consecutive operations, check if we can remove one
    for i in range(len(ops) - 1):
        if time.time() > deadline:
            break
        for j in range(i + 1, min(i + 5, len(ops))):
            if ops[i][0] == ops[j][0] and ops[i][1] == ops[j][1]:
                # Same operation on same vertex - try removing one
                trial = ops[:i] + ops[i+1:j] + ops[j+1:]
                if _verify_ops(trial, dag, budget) and transfers(trial) < best_score:
                    best_ops = trial
                    best_score = transfers(trial)
    
    return best_ops

def _pebble_postorder(dag, budget):
    """Pebble each output's subtree in postorder, recomputing on demand."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    topo, children = _topo_order(parents, vertices)
    post = list(reversed(topo))
    
    rcost = _compute_recompute_costs(parents, vertices)
    
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
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices)
    
    ops = []
    in_mem = set()
    stored = set()
    
    def ensure_in_mem2(v):
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
                ensure_in_mem2(p)
            if len(in_mem) >= b and v not in in_mem:
                for x in list(in_mem):
                    if x not in outset:
                        ops.append(["delete", x])
                        in_mem.discard(x)
                        if len(in_mem) < b:
                            break
            ops.append(["compute", v])
            in_mem.add(v)
    
    for v in outputs:
        ensure_in_mem2(v)
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
                    # Check if x is ancestor of rem_out
                    anc = _get_ancestors(parents, rem_out)
                    if x in anc:
                        needed = True
                        break
                if not needed:
                    to_delete.append(x)
        for x in to_delete:
            ops.append(["delete", x])
            in_mem.discard(x)
    
    return ops[:20000]

def _hierarchical_pebble(dag, budget, deadline):
    """
    Hierarchical pebbling: partition DAG into blocks based on depth layers,
    recursively pebble each block.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices)
    n = len(vertices)
    
    # Compute depth
    depth = {v: 0 for v in vertices}
    for v in topo:
        ps = parents.get(v, [])
        if ps:
            depth[v] = 1 + max(depth[p] for p in ps)
    
    max_depth = max(depth.values()) if depth else 0
    
    # For layered DAGs, use the recursive layered pebbling
    if _is_layered(parents, vertices, topo):
        return _recursive_pebble_layered(parents, children, vertices, outputs, budget, deadline, depth)
    
    # For general DAGs, use block-based approach
    if max_depth <= 20:
        blocks_by_depth = {}
        for v in topo:
            d = depth[v]
            if d not in blocks_by_depth:
                blocks_by_depth[d] = []
            blocks_by_depth[d].append(v)
        blocks = [blocks_by_depth[d] for d in sorted(blocks_by_depth.keys())]
    else:
        num_blocks = max(2, min(int(n ** 0.5), 10))
        block_size = (n + num_blocks - 1) // num_blocks
        blocks = [topo[i:i+block_size] for i in range(0, n, block_size)]
    
    ops = []
    in_mem = set()
    stored = set()
    
    def can_delete(v):
        if v in outset or v in stored:
            return False
        for w in children.get(v, []):
            if w not in stored and w not in in_mem:
                return False
        return True
    
    def make_space(needed):
        freed = 0
        for v in reversed(topo):
            if v in in_mem and can_delete(v):
                ops.append(["delete", v])
                in_mem.discard(v)
                freed += 1
                if freed >= needed:
                    break
    
    for block in blocks:
        for v in block:
            if time.time() > deadline:
                break
            
            ps = parents.get(v, [])
            
            for p in ps:
                if p not in in_mem:
                    if p in stored:
                        if len(in_mem) >= b:
                            make_space(1)
                        ops.append(["load", p])
                        in_mem.add(p)
                    else:
                        pps = parents.get(p, [])
                        if not pps:
                            if len(in_mem) >= b:
                                make_space(1)
                            ops.append(["load", p])
                            in_mem.add(p)
                        else:
                            for pp in pps:
                                if pp not in in_mem:
                                    ppps = parents.get(pp, [])
                                    if not ppps:
                                        if len(in_mem) >= b:
                                            make_space(1)
                                        ops.append(["load", pp])
                                        in_mem.add(pp)
                                    else:
                                        if len(in_mem) >= b:
                                            make_space(1)
                                        ops.append(["load", pp])
                                        in_mem.add(pp)
                            if len(in_mem) >= b and p not in in_mem:
                                make_space(1)
                            ops.append(["compute", p])
                            in_mem.add(p)
            
            if len(in_mem) >= b and v not in in_mem:
                make_space(1)
            ops.append(["compute", v])
            in_mem.add(v)
            
            if v in outset:
                ops.append(["store", v])
                stored.add(v)
                in_mem.discard(v)
            else:
                future_needed = any(
                    v in parents.get(w, []) 
                    for w in vertices 
                    if w not in stored and w not in in_mem and depth.get(w, 0) > depth.get(v, 0)
                )
                if not future_needed:
                    ops.append(["delete", v])
                    in_mem.discard(v)
                else:
                    if len(in_mem) > b:
                        for x in reversed(list(in_mem)):
                            if x != v and x not in outset and can_delete(x):
                                ops.append(["delete", x])
                                in_mem.discard(x)
                                break
    
    return ops[:20000]

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 40  # Safety margin from 60s limit
    candidates = []
    
    # Try A* first for small DAGs (can find optimal)
    try:
        r = _pebble_astar(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            if _verify_ops(r, dag, fast_memory_budget):
                candidates.append(r)
    except Exception:
        pass
    
    # Try hierarchical pebbling (includes layered pebbling)
    try:
        r = _hierarchical_pebble(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            if _verify_ops(r, dag, fast_memory_budget):
                candidates.append(r)
    except Exception:
        pass
    
    # Postorder pebbling (recomputation-based)
    try:
        r = _pebble_postorder(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            if _verify_ops(r, dag, fast_memory_budget):
                candidates.append(r)
    except Exception:
        pass
    
    # Memory-aware greedy
    try:
        r = _pebble_greedy_memory_aware(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            if _verify_ops(r, dag, fast_memory_budget):
                candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        # Fallback: simple topological order with recomputation
        try:
            parents = dag.get("parents", {})
            outputs = list(dag.get("outputs", []))
            vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
            if vertices:
                topo, _ = _topo_order(parents, vertices)
                ops = []
                for v in topo:
                    ps = parents.get(v, [])
                    for p in ps:
                        if p not in ops:
                            ops.append(["load", p])
                    ops.append(["compute", v])
                    if v in set(outputs):
                        ops.append(["store", v])
                if len(ops) <= 20000:
                    return ops
        except Exception:
            pass
        return []
    
    # Local search polish on the best candidate
    def transfers(ops):
        return sum(1 for op, v in ops if op in ("load", "store"))
    
    best = min(candidates, key=transfers)
    
    # Try local search polish if we have time
    if time.time() < deadline - 5:
        try:
            polished = _local_search_polish(best, dag, fast_memory_budget, deadline)
            if polished and transfers(polished) < transfers(best):
                best = polished
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
