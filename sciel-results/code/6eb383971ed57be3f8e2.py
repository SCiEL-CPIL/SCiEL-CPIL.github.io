"""DAG pebbling: communication-minimizing schedule with bounded fast memory."""
import time
import heapq
from collections import deque

def _topo_order(parents, vertices):
    """Return topological order and children map."""
    vset = set(vertices)
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in vset:
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

def _compute_layers(parents, topo, vertices):
    """Compute layer (longest path from input) for each vertex."""
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
    return layer_of, max_layer, layers

def _validate_ops(dag, ops, budget):
    """Validate operations: memory budget respected, all outputs stored, no illegal ops."""
    if not ops:
        return False
    if len(ops) > 20000:
        return False
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(outputs))
    
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

def _count_transfers(ops):
    return sum(1 for op, v in ops if op in ("load", "store"))

def _pebble_layered_optimal(dag, budget):
    """
    Layered pebbling with optimal memory management.
    
    For layered DAGs, process layer by layer. At each layer:
    - Ensure parents (from previous layers) are available
    - Compute vertices in this layer
    - Decide which to keep in memory vs. store to slow memory
    
    Uses a greedy strategy that minimizes expected recomputation cost.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices)
    layer_of, max_layer, layers = _compute_layers(parents, topo, vertices)
    
    # Precompute: for each vertex, which future layers need it
    # and the total "use count" (how many times it's needed by descendants)
    future_uses = {v: 0 for v in vertices}
    for v in vertices:
        for child in children.get(v, []):
            future_uses[v] += 1
    
    # Compute recompute cost: min number of compute ops to produce v from inputs
    recompute_cost = {}
    def get_rcost(v):
        if v in recompute_cost:
            return recompute_cost[v]
        ps = parents.get(v, [])
        if not ps:
            recompute_cost[v] = 1  # load
        else:
            c = 1 + sum(get_rcost(p) for p in ps)
            recompute_cost[v] = c
        return c
    for v in vertices:
        get_rcost(v)
    
    ops = []
    in_mem = set()
    stored = set()
    
    def ensure_available(v):
        """Ensure v is in in_mem or stored. If in stored, load it."""
        if v in in_mem or v in stored:
            return
        # Need to compute v: ensure parents available
        ps = parents.get(v, [])
        for p in ps:
            ensure_available(p)
        if v not in in_mem:
            ops.append(["compute", v])
            in_mem.add(v)
    
    def make_space():
        """Free up one slot in in_mem by deleting least valuable vertex."""
        if len(in_mem) < b:
            return
        # Find vertex to delete: minimize (future_uses * recompute_cost)
        best_v = None
        best_score = float('inf')
        for v in in_mem:
            if v in outset:
                continue  # Don't delete outputs
            score = future_uses[v] * recompute_cost[v]
            if score < best_score:
                best_score = score
                best_v = v
        if best_v is not None:
            ops.append(["delete", best_v])
            in_mem.discard(best_v)
        else:
            # All are outputs, delete any non-output if possible
            for v in in_mem:
                if v not in outset:
                    ops.append(["delete", v])
                    in_mem.discard(v)
                    return
            # As last resort, delete first output
            if in_mem:
                v = next(iter(in_mem))
                ops.append(["delete", v])
                in_mem.discard(v)
    
    for layer_idx in range(max_layer + 1):
        layer_vertices = layers[layer_idx]
        for v in layer_vertices:
            ps = parents.get(v, [])
            # Ensure all parents are available
            for p in ps:
                if p not in in_mem and p not in stored:
                    ensure_available(p)
                elif p in stored and p not in in_mem:
                    # Load from stored
                    make_space()
                    ops.append(["load", p])
                    in_mem.add(p)
            
            # Compute v
            if v not in in_mem:
                make_space()
                ops.append(["compute", v])
                in_mem.add(v)
            
            # Decide: store or delete or keep
            if v in outset:
                # Must store
                make_space() if len(in_mem) >= b else None
                ops.append(["store", v])
                in_mem.discard(v)
                stored.add(v)
            else:
                # Check if needed by future layers
                needed_future = False
                for child in children.get(v, []):
                    if layer_of[child] > layer_idx:
                        needed_future = True
                        break
                
                if not needed_future:
                    # Not needed later, delete
                    ops.append(["delete", v])
                    in_mem.discard(v)
                elif len(in_mem) >= b:
                    # Keep if valuable, else delete
                    # Compare value of keeping v vs. what we'd delete
                    v_score = future_uses[v] * recompute_cost[v]
                    other_v = None
                    other_score = float('inf')
                    for u in in_mem:
                        if u != v and u not in outset:
                            u_score = future_uses[u] * recompute_cost[u]
                            if u_score < other_score:
                                other_score = u_score
                                other_v = u
                    if other_v is not None and other_score < v_score:
                        ops.append(["delete", other_v])
                        in_mem.discard(other_v)
                    else:
                        # Delete v, will recompute if needed
                        ops.append(["delete", v])
                        in_mem.discard(v)
    
    # Ensure all outputs are stored
    for o in outset:
        if o not in stored:
            if o in in_mem:
                ops.append(["store", o])
                in_mem.discard(o)
                stored.add(o)
            else:
                ensure_available(o)
                make_space()
                ops.append(["store", o])
                in_mem.discard(o)
                stored.add(o)
    
    return ops[:20000]

def _pebble_astar_small(dag, budget, deadline):
    """A* search for small DAGs (n <= 15) to find optimal schedule."""
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
    
    b = max(1, int(budget) if budget is not None else n)
    
    def popcount(x):
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")
    
    # Precompute recompute costs for heuristic
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
    
    def heuristic(mem_mask, stored_mask):
        """Lower bound on remaining transfers."""
        missing_outputs = out_mask & ~stored_mask
        h = 0
        mm = missing_outputs
        while mm:
            lb = mm & (-mm)
            i = lb.bit_length() - 1
            # Need to store this output: at least 1 transfer
            h += 1
            # If not in memory, need to compute it (0 transfers) but its inputs may need loading
            if not (mem_mask & (1 << i)):
                # Count input ancestors not in memory
                anc_inputs = 0
                stack = [i]
                visited = set()
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    if is_input[node] and not (mem_mask & (1 << node)):
                        anc_inputs |= (1 << node)
                    m = par_mask[node]
                    while m:
                        lb2 = m & (-m)
                        j = lb2.bit_length() - 1
                        stack.append(j)
                        m ^= lb2
                h += popcount(anc_inputs)
            mm ^= lb
        return h
    
    start_state = (0, 0)  # (mem_mask, stored_mask)
    counter = 0
    open_heap = [(heuristic(0, 0), 0, counter, (0, 0))]
    g_score = {start_state: 0}
    parent_op = {}
    best = None
    start_t = time.time()
    
    while open_heap:
        if time.time() > deadline:
            break
        f, g, _, (mem, stored) = heapq.heappop(open_heap)
        
        if stored == out_mask:
            seq = []
            state = (mem, stored)
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
                    new_stored = stored
                    ng = g + 1
                    ns = (new_mem, new_stored)
                    if ng < g_score.get(ns, float("inf")):
                        g_score[ns] = ng
                        parent_op[ns] = ((mem, stored), "load", v)
                        counter += 1
                        heapq.heappush(open_heap, (ng + heuristic(new_mem, new_stored), ng, counter, ns))
            
            # Compute (parents must be in memory or stored)
            if (mem & par_mask[i]) == par_mask[i] and not (mem & bit):
                # Check parents are available (in mem or stored)
                parents_ok = True
                pm = par_mask[i]
                while pm:
                    lb = pm & (-pm)
                    j = lb.bit_length() - 1
                    if not (mem & (1 << j)) and not (stored & (1 << j)):
                        parents_ok = False
                        break
                    pm ^= lb
                if parents_ok:
                    new_mem = mem | bit
                    new_stored = stored | (bit if (out_mask & bit) else 0)
                    ng = g
                    ns = (new_mem, new_stored)
                    if ng < g_score.get(ns, float("inf")):
                        g_score[ns] = ng
                        parent_op[ns] = ((mem, stored), "compute", v)
                        counter += 1
                        heapq.heappush(open_heap, (ng + heuristic(new_mem, new_stored), ng, counter, ns))
            
            # Store output
            if (mem & bit) and (out_mask & bit):
                new_mem = mem & ~bit
                new_stored = stored | bit
                ng = g + 1
                ns = (new_mem, new_stored)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, stored), "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_stored), ng, counter, ns))
            
            # Delete non-output
            if (mem & bit) and not (out_mask & bit):
                new_mem = mem & ~bit
                new_stored = stored
                ng = g
                ns = (new_mem, new_stored)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, stored), "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_stored), ng, counter, ns))
    
    return best

def _pebble_naive(dag, budget):
    """Naive topological pebbling: load inputs, compute, store outputs."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices)
    
    ops = []
    in_mem = set()
    stored = set()
    
    for v in topo:
        ps = parents.get(v, [])
        for p in ps:
            if p not in in_mem and p not in stored:
                # Load input
                if len(in_mem) >= b:
                    # Delete something
                    for x in list(in_mem):
                        if x not in outset:
                            ops.append(["delete", x])
                            in_mem.discard(x)
                            if len(in_mem) < b:
                                break
                if not parents.get(p, []):
                    ops.append(["load", p])
                    in_mem.add(p)
                else:
                    # Recompute
                    pps = parents.get(p, [])
                    for pp in pps:
                        if pp not in in_mem and pp not in stored:
                            if not parents.get(pp, []):
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
                                # Deep recompute - just load if input
                                if not parents.get(pp, []):
                                    ops.append(["load", pp])
                                    in_mem.add(pp)
                    if all(p2 in in_mem or p2 in stored for p2 in pps):
                        if len(in_mem) >= b:
                            for x in list(in_mem):
                                if x not in outset:
                                    ops.append(["delete", x])
                                    in_mem.discard(x)
                                    if len(in_mem) < b:
                                        break
                        ops.append(["compute", p])
                        in_mem.add(p)
        
        if v not in in_mem:
            if len(in_mem) >= b:
                for x in list(in_mem):
                    if x not in outset:
                        ops.append(["delete", x])
                        in_mem.discard(x)
                        if len(in_mem) < b:
                            break
            if not ps:
                ops.append(["load", v])
            else:
                ops.append(["compute", v])
            in_mem.add(v)
        
        if v in outset:
            ops.append(["store", v])
            in_mem.discard(v)
            stored.add(v)
        else:
            # Delete if not needed by future vertices
            needed = False
            for child in children.get(v, []):
                if child not in stored:
                    needed = True
                    break
            if not needed:
                ops.append(["delete", v])
                in_mem.discard(v)
    
    return ops[:20000]

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50  # Safety margin for 90s limit
    candidates = []
    
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    
    # Try A* for small DAGs (most likely to be optimal)
    if n <= 15:
        try:
            r = _pebble_astar_small(dag, fast_memory_budget, deadline)
            if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
                candidates.append(r)
        except Exception:
            pass
    
    # Try layered optimal pebbling
    try:
        r = _pebble_layered_optimal(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    # Naive fallback
    try:
        r = _pebble_naive(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000 and _validate_ops(dag, r, fast_memory_budget):
            candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        # Last resort: minimal valid schedule
        ops = []
        in_mem = set()
        stored = set()
        b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)
        
        for v in vertices:
            ps = parents.get(v, [])
            for p in ps:
                if p not in in_mem and p not in stored:
                    if not parents.get(p, []):
                        while len(in_mem) >= b:
                            for x in list(in_mem):
                                if x not in outset:
                                    ops.append(["delete", x])
                                    in_mem.discard(x)
                                    break
                        ops.append(["load", p])
                        in_mem.add(p)
            if v not in in_mem:
                while len(in_mem) >= b:
                    for x in list(in_mem):
                        if x not in outset:
                            ops.append(["delete", x])
                            in_mem.discard(x)
                            break
                if not ps:
                    ops.append(["load", v])
                else:
                    ops.append(["compute", v])
                in_mem.add(v)
            if v in outset:
                ops.append(["store", v])
                in_mem.discard(v)
                stored.add(v)
        
        if _validate_ops(dag, ops, fast_memory_budget):
            return ops[:20000]
        return []
    
    best = min(candidates, key=_count_transfers)
    return best
