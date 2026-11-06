"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def _pebble_fallback(dag, budget):
    # Greedy baseline: load parents, compute, store outputs, delete when possible.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    ops = []
    def cost(v):
        ps = parents.get(v, [])
        if not ps:
            return 1
        return 1 + sum(cost(p) for p in ps)
    cost_cache = {}
    def ccost(v):
        if v not in cost_cache:
            ps = parents.get(v, [])
            cost_cache[v] = 1 + (sum(ccost(p) for p in ps) if ps else 0)
        return cost_cache[v]
    # Process vertices in topological order, load all needed, compute, store outputs, delete non-outputs.
    # Determine topological order.
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
        topo = vertices  # cycle fallback
    outset = set(outputs)
    for v in topo:
        # load parents if not loaded
        for p in parents.get(v, []):
            ops.append(["load", p])
        ops.append(["compute", v])
        if v in outset:
            ops.append(["store", v])
        # delete parents that are no longer needed
        for p in parents.get(v, []):
            still = any(p in parents.get(w, []) for w in topo[topo.index(v)+1:] if w in vset)
            if not still and p not in outset:
                ops.append(["delete", p])
    return ops[:20000]

def _count_transfers(ops):
    return sum(1 for op in ops if op[0] in ("load", "store"))

def _naive_schedule(dag):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return 0
    vset = set(vertices)
    # Topological order
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
        topo = vertices
    outset = set(outputs)
    ops = []
    for v in topo:
        for p in parents.get(v, []):
            ops.append(["load", p])
        ops.append(["compute", v])
        if v in outset:
            ops.append(["store", v])
        for p in parents.get(v, []):
            still = any(p in parents.get(w, []) for w in topo[topo.index(v)+1:] if w in vset)
            if not still and p not in outset:
                ops.append(["delete", p])
    return _count_transfers(ops)

def _pebble_astar(dag, budget, deadline):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    # Precompute parents bitmasks and children sets
    par_mask = [0] * n
    children = [set() for _ in range(n)]
    for v, ps in parents.items():
        i = idx[v]
        m = 0
        for p in ps:
            j = idx[p]
            m |= (1 << j)
            children[j].add(i)
        par_mask[i] = m
    # Cost (recompute count) per vertex
    cost = [0] * n
    def ccost(i):
        if cost[i] != 0:
            return cost[i]
        m = par_mask[i]
        if m == 0:
            cost[i] = 1
        else:
            s = 0
            mm = m
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                s += ccost(j)
                mm ^= lb
            cost[i] = 1 + s
        return cost[i]
    for i in range(n):
        ccost(i)

    def popcount(x):
        return bin(x).count("1")

    # Admissible heuristic: min recomputations needed to reach goal from loaded set L
    # We approximate by: for each output not in L, we need to recompute it (cost) but some parents may be free.
    # Lower bound: number of outputs not yet in L that are "unavailable" — use count of outputs missing.
    def heuristic(loaded):
        missing = (out_mask & ~loaded)
        # each missing output requires at least 1 op (compute) -> transfers lower bound ~ missing count
        return popcount(missing)

    start = 0
    goal = out_mask
    # A*
    counter = 0
    open_heap = [(heuristic(start), 0, counter, start)]
    g_score = {start: 0}
    parent_op = {}
    best = None
    start_t = time.time()
    # Reconstruct ops from path
    def reconstruct(path_loaded, start_mask):
        # We stored parent_op[state] = (prev_state, op, vertex)
        # Build reverse list
        seq = []
        state = goal
        while state != start_mask:
            prev, op, v = parent_op[state]
            seq.append([op, v])
            state = prev
        seq.reverse()
        return seq

    while open_heap:
        if time.time() > deadline:
            break
        f, g, _, state = heapq.heappop(open_heap)
        if state == goal:
            best = reconstruct(state, start)
            break
        if g > g_score.get(state, float("inf")):
            continue
        # Generate successors: for each vertex v, possible ops
        for v in vertices:
            i = idx[v]
            bit = 1 << i
            # load v if not loaded
            if not (state & bit):
                ns = state | bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "load", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            # store v if loaded and is output
            if (state & bit) and (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            # delete v if loaded and not output
            if (state & bit) and not (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            # compute v if all parents loaded
            if (state & par_mask[i]) == par_mask[i]:
                ns = state | bit
                ng = g + 1  # compute is not a transfer, but we count ops; transfers = load+store only
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
    return best

def _pebble_iterative(dag, budget, deadline):
    """Iterative deepening A* for small DAGs to find optimal pebbling."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 12:
        return None  # Too large for IDA*
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    par_mask = [0] * n
    for v, ps in parents.items():
        i = idx[v]
        m = 0
        for p in ps:
            j = idx[p]
            m |= (1 << j)
        par_mask[i] = m
    
    def popcount(x):
        return bin(x).count("1")
    
    def heuristic(loaded):
        missing = (out_mask & ~loaded)
        return popcount(missing)
    
    start = 0
    goal = out_mask
    
    # IDA*
    h0 = heuristic(start)
    for threshold in range(h0, 200):
        if time.time() > deadline:
            break
        
        def dfs(state, g, parent_op, path):
            if time.time() > deadline:
                return None, threshold
            if state == goal:
                return path, threshold
            f = g + heuristic(state)
            if f > threshold:
                return None, f
            best_bound = float('inf')
            # Generate successors in a good order
            # Priority: compute if possible, then load, then delete/store
            successors = []
            for v in vertices:
                i = idx[v]
                bit = 1 << i
                # compute v if all parents loaded
                if (state & par_mask[i]) == par_mask[i] and not (state & bit):
                    ns = state | bit
                    successors.append((ns, "compute", v, 0))
            for v in vertices:
                i = idx[v]
                bit = 1 << i
                # load v if not loaded
                if not (state & bit):
                    ns = state | bit
                    successors.append((ns, "load", v, 1))
            for v in vertices:
                i = idx[v]
                bit = 1 << i
                # delete v if loaded and not output
                if (state & bit) and not (out_mask & bit):
                    ns = state & ~bit
                    successors.append((ns, "delete", v, 2))
                # store v if loaded and is output
                if (state & bit) and (out_mask & bit):
                    ns = state & ~bit
                    successors.append((ns, "store", v, 3))
            
            successors.sort(key=lambda x: x[3])
            for ns, op, v, _ in successors:
                new_path = path + [[op, v]]
                result, bound = dfs(ns, g + 1, parent_op, new_path)
                if result is not None:
                    return result, bound
                best_bound = min(best_bound, bound)
            return None, best_bound
        
        result, bound = dfs(start, 0, {}, [])
        if result is not None:
            return result
        if bound > 200:
            break
    return None

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50  # safety margin before 60s (and 90s kill)
    
    # Try IDA* first for small DAGs
    try:
        result = _pebble_iterative(dag, fast_memory_budget, deadline)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    
    # Try A*
    try:
        result = _pebble_astar(dag, fast_memory_budget, deadline)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    
    # Fallback to greedy baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        return _pebble_fallback(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
