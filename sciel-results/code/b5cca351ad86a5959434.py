"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
import math
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
        topo = vertices
    return topo, children

def _pebble_fallback(dag, budget):
    # Greedy baseline: load parents, compute, store outputs, delete when possible.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    topo, children = _topo_order(parents, vertices)
    # last-use index for each vertex
    last_use = {v: -1 for v in vertices}
    for i, v in enumerate(topo):
        for p in parents.get(v, []):
            last_use[p] = i
    ops = []
    loaded = set()
    for i, v in enumerate(topo):
        for p in parents.get(v, []):
            if p not in loaded:
                ops.append(["load", p])
                loaded.add(p)
        ops.append(["compute", v])
        if v in outset:
            ops.append(["store", v])
            loaded.discard(v)
        else:
            loaded.add(v)
        # delete parents no longer needed (last use passed, not outputs)
        for p in list(loaded):
            if p not in outset and last_use.get(p, -1) < i:
                ops.append(["delete", p])
                loaded.discard(p)
    return ops[:20000]

def _pebble_astar(dag, budget, deadline):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 18:
        return None
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    par_mask = [0] * n
    for v, ps in parents.items():
        i = idx[v]
        m = 0
        for p in ps:
            m |= (1 << idx[p])
        par_mask[i] = m
    b = max(1, int(budget) if budget is not None else n)

    def popcount(x):
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")

    # Precompute for each vertex: which outputs depend on it
    output_deps = [0] * n  # bitmask of outputs that depend on vertex i
    for o in outputs:
        oi = idx[o]
        # BFS to find all ancestors of o
        stack = [oi]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            output_deps[node] |= (1 << oi)
            for p in parents.get(vertices[node], []):
                if p in idx:
                    stack.append(idx[p])

    # Precompute recompute cost (number of compute ops needed to regenerate v from inputs)
    recompute_cost = [0] * n
    def calc_cost(i, seen=None):
        if seen is None:
            seen = set()
        if i in seen:
            return 0
        seen.add(i)
        m = par_mask[i]
        if m == 0:
            recompute_cost[i] = 1
        else:
            s = 1
            mm = m
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                s += calc_cost(j, seen)
                mm ^= lb
            recompute_cost[i] = s
        return recompute_cost[i]
    for i in range(n):
        calc_cost(i)

    def heuristic(loaded):
        # Improved heuristic: for each missing output, estimate minimum transfers needed
        h = 0
        for o in outputs:
            oi = idx[o]
            if loaded & (1 << oi):
                continue
            # This output is not loaded
            # At minimum, we need to compute it (1 op) and store it (1 op)
            # Plus we need all its parents loaded if not already
            par = par_mask[oi]
            missing_parents = par & ~loaded
            h += 2 + popcount(missing_parents)  # compute + store + loads
        return h

    start = 0
    goal = out_mask
    counter = 0
    open_heap = [(heuristic(start), 0, counter, start)]
    g_score = {start: 0}
    parent_op = {}
    best = None
    while open_heap:
        if time.time() > deadline:
            break
        f, g, _, state = heapq.heappop(open_heap)
        if state == goal:
            seq = []
            s = goal
            while s != start:
                prev, op, v = parent_op[s]
                seq.append([op, v])
                s = prev
            seq.reverse()
            best = seq
            break
        if g > g_score.get(state, float("inf")):
            continue
        for v in vertices:
            i = idx[v]
            bit = 1 << i
            if not (state & bit):
                if popcount(state) < b:  # respect memory budget
                    ns = state | bit
                    ng = g + 1
                    if ng < g_score.get(ns, float("inf")):
                        g_score[ns] = ng
                        parent_op[ns] = (state, "load", v)
                        counter += 1
                        heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            if (state & bit) and (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            if (state & bit) and not (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            if (state & par_mask[i]) == par_mask[i]:
                ns = state | bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
    return best

def _pebble_beam(dag, budget, deadline):
    """Beam search with recompute-aware heuristic for medium-sized DAGs."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 30:
        return None
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    par_mask = [0] * n
    for v, ps in parents.items():
        i = idx[v]
        m = 0
        for p in ps:
            m |= (1 << idx[p])
        par_mask[i] = m
    b = max(1, int(budget) if budget is not None else n)
    
    def popcount(x):
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")
    
    # Precompute recompute cost
    recompute_cost = [0] * n
    def calc_cost(i, seen=None):
        if seen is None:
            seen = set()
        if i in seen:
            return 0
        seen.add(i)
        m = par_mask[i]
        if m == 0:
            recompute_cost[i] = 1
        else:
            s = 1
            mm = m
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                s += calc_cost(j, seen)
                mm ^= lb
            recompute_cost[i] = s
        return recompute_cost[i]
    for i in range(n):
        calc_cost(i)
    
    def heuristic(loaded):
        h = 0
        for o in outputs:
            oi = idx[o]
            if loaded & (1 << oi):
                continue
            par = par_mask[oi]
            missing_parents = par & ~loaded
            h += 2 + popcount(missing_parents)
        return h
    
    beam_width = 50
    max_steps = 500
    beam = [(heuristic(0), 0, 0, 0)]  # (f, g, step, state)
    parent_map = {0: None}  # state -> (prev_state, op, vertex)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        if not beam:
            break
        new_beam = []
        seen_states = set()
        for f, g, _, state in beam:
            if state == out_mask:
                # Reconstruct path
                seq = []
                s = state
                while parent_map[s] is not None:
                    prev, op, v = parent_map[s]
                    seq.append([op, v])
                    s = prev
                seq.reverse()
                return seq
            for v in vertices:
                i = idx[v]
                bit = 1 << i
                # Generate successors
                ns = None
                op = None
                if not (state & bit):
                    if popcount(state) < b:
                        ns = state | bit
                        op = "load"
                elif (state & bit) and (out_mask & bit):
                    ns = state & ~bit
                    op = "store"
                elif (state & bit) and not (out_mask & bit):
                    ns = state & ~bit
                    op = "delete"
                elif (state & par_mask[i]) == par_mask[i]:
                    ns = state | bit
                    op = "compute"
                if ns is not None and ns not in seen_states:
                    ng = g + 1
                    nf = ng + heuristic(ns)
                    seen_states.add(ns)
                    parent_map[ns] = (state, op, v)
                    new_beam.append((nf, ng, step + 1, ns))
        if not new_beam:
            break
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    return None

def _pebble_local_search(dag, budget, initial_ops, deadline):
    """Local search to improve an initial schedule by swapping/reordering operations."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    vset = set(vertices)
    outset = set(outputs)
    
    if not initial_ops:
        return initial_ops
    
    def count_transfers(ops):
        return sum(1 for op, v in ops if op in ("load", "store"))
    
    def validate(ops):
        """Check if a schedule is valid."""
        loaded = set()
        for op, v in ops:
            if op == "load":
                if v in loaded:
                    return False
                loaded.add(v)
            elif op == "store":
                if v not in loaded:
                    return False
                loaded.discard(v)
            elif op == "delete":
                if v not in loaded:
                    return False
                loaded.discard(v)
            elif op == "compute":
                ps = parents.get(v, [])
                for p in ps:
                    if p not in loaded:
                        return False
                if v not in loaded:
                    loaded.add(v)
        # All outputs must be stored (i.e., they were loaded at some point and then stored)
        # This is implicitly checked by the fact that store requires load first
        return True
    
    best_ops = list(initial_ops)
    best_cost = count_transfers(best_ops)
    
    # Try simple local moves: swap adjacent operations if valid
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_ops) - 1):
            if time.time() > deadline:
                break
            # Try swapping ops[i] and ops[i+1]
            trial = best_ops.copy()
            trial[i], trial[i+1] = trial[i+1], trial[i]
            if validate(trial):
                c = count_transfers(trial)
                if c < best_cost:
                    best_ops = trial
                    best_cost = c
                    improved = True
                    break
    return best_ops

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 40  # safety margin before 60s / 90s kill
    candidates = []
    
    # Try A* for small DAGs (up to 18 vertices)
    try:
        r = _pebble_astar(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Try beam search for medium DAGs (up to 30 vertices)
    try:
        r = _pebble_beam(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Greedy fallback
    try:
        r = _pebble_fallback(dag, fast_memory_budget)
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
    
    # Try local search to improve
    try:
        improved = _pebble_local_search(dag, fast_memory_budget, best, deadline)
        if improved is not None and len(improved) <= 20000:
            best = improved
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
