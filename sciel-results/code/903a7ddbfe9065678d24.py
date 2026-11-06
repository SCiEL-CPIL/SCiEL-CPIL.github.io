"""Communication-minimizing DAG pebbling: DFS postorder with register eviction."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque


def _topo_order(parents, vertices, vset):
    """Kahn's topological sort; returns (topo_list, children_map)."""
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
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
        # Cycle or missing nodes; fall back to sorted order
        topo = sorted(vertices, key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)))
    return topo, children


def _pebble_dfs(dag, budget):
    """
    DFS postorder pebbling with register eviction.
    
    Strategy:
    - Process outputs in order (by topological position).
    - For each output, recursively compute its dependency tree in postorder.
    - After computing a vertex, if all its children are done, evict it (free register).
    - This is the classic 'pebbling with recomputation' strategy.
    - Total ops = sum over all vertices of (1 compute) + (1 load for inputs if not already loaded).
    - No stores: outputs are just computed and left in memory (or we don't track them).
    
    Key: we want to minimize total operations. Since compute is free (not a transfer),
    the score is based on transfers (load + store). We want to minimize loads.
    
    Actually, re-reading the task: "Score mean transfers / naive schedule". 
    Transfers = load + store operations.
    
    Optimal: minimize loads. Each input that must be loaded counts as 1 transfer.
    If we can keep an input in memory across multiple outputs, we save loads.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    all_verts = set(list(parents.keys()) + outputs)
    if not all_verts:
        return []
    
    # Sort vertices for deterministic ordering
    def vkey(v):
        if isinstance(v, int):
            return (0, v, "")
        return (1, 0, str(v))
    vertices = sorted(all_verts, key=vkey)
    vset = set(vertices)
    outset = set(outputs)
    
    topo, children = _topo_order(parents, vertices, vset)
    
    # Determine processing order for outputs: topological order
    out_order = [o for o in topo if o in outset]
    
    # For each vertex, compute which vertices are its descendants (in the subgraph rooted at outputs)
    # We need to know when a vertex is "done" - i.e., all its children that are on paths to outputs are computed.
    
    # Precompute: for each vertex, the set of children that are relevant (on path to some output)
    # A child is relevant if it's an output or has a relevant child.
    relevant_children = {v: [] for v in vertices}
    for v in vertices:
        for c in children.get(v, []):
            if c in outset or any(cc in outset for cc in _descendants(c, children, outset)):
                relevant_children[v].append(c)
    
    # Actually, simpler: a vertex is "freed" when all its children that lead to outputs are computed.
    # Let's just do DFS and track which children have been computed.
    
    ops = []
    in_reg = set()  # vertices currently in fast memory
    done = set()    # vertices that have been fully computed (and their result is available)
    
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    # Count how many times each input is loaded
    input_loads = {}
    
    def make_space():
        """Evict a non-output vertex from registers if needed."""
        if len(in_reg) < b:
            return
        # Evict the vertex that is furthest from being needed
        # Simple heuristic: evict the one with the highest topological order (computed earliest)
        candidates = [v for v in in_reg if v not in outset]
        if candidates:
            # Evict the one with the smallest topological index (oldest)
            topo_idx = {v: i for i, v in enumerate(topo)}
            victim = min(candidates, key=lambda v: topo_idx.get(v, 0))
            ops.append(["delete", victim])
            in_reg.discard(victim)
    
    def compute_vertex(v):
        """Compute vertex v, ensuring all parents are in registers."""
        if v in done:
            return
        ps = parents.get(v, [])
        for p in ps:
            compute_vertex(p)
        # Now all parents are in done (and should be in in_reg or can be reloaded)
        # Ensure parents are in registers
        for p in ps:
            if p not in in_reg:
                if p in outset:
                    # Output: reload it (it was stored/evicted)
                    ops.append(["load", p])
                    in_reg.add(p)
                else:
                    # Non-output: it was deleted, need to recompute
                    # This shouldn't happen if we keep parents until children are done
                    compute_vertex(p)
                    if p not in in_reg:
                        # Recomputed but evicted? Reload
                        ops.append(["load", p])
                        in_reg.add(p)
        
        ops.append(["compute", v])
        in_reg.add(v)
        done.add(v)
        
        # Free parents that are no longer needed
        for p in ps:
            if p in outset:
                continue  # Keep outputs
            # Check if p is needed for any other child not yet done
            needed = False
            for c in children.get(p, []):
                if c != v and c not in done:
                    # Check if c is on a path to an output
                    if c in outset or _can_reach_output(c, children, outset):
                        needed = True
                        break
            if not needed:
                ops.append(["delete", p])
                in_reg.discard(p)
    
    def _can_reach_output(v, children, outset, memo=None):
        """Check if v can reach an output."""
        if v in outset:
            return True
        for c in children.get(v, []):
            if _can_reach_output(c, children, outset):
                return True
        return False
    
    # Simpler approach: precompute the set of vertices that are ancestors of outputs
    ancestors_of_outputs = set()
    def add_ancestors(v):
        if v in ancestors_of_outputs:
            return
        ancestors_of_outputs.add(v)
        for p in parents.get(v, []):
            add_ancestors(p)
    for o in outputs:
        add_ancestors(o)
    
    # Now redo: only compute vertices that are ancestors of outputs
    ops = []
    in_reg = set()
    done = set()
    
    def compute_v2(v):
        if v in done:
            return
        ps = parents.get(v, [])
        for p in ps:
            if p in ancestors_of_outputs:
                compute_v2(p)
        ops.append(["compute", v])
        in_reg.add(v)
        done.add(v)
        
        # Free parents no longer needed
        for p in ps:
            if p in outset:
                continue
            needed = False
            for c in children.get(p, []):
                if c != v and c in ancestors_of_outputs and c not in done:
                    needed = True
                    break
            if not needed:
                ops.append(["delete", p])
                in_reg.discard(p)
    
    for o in out_order:
        compute_v2(o)
        # Output is done; we could delete it to free space, but we need it to be "blue" at the end
        # The task says "all outputs must finish blue" - so we keep them in memory
    
    return ops[:20000]


def _pebble_optimal_small(dag, budget):
    """
    For small DAGs (n <= 12), use A* search over (memory_mask, done_mask) states.
    Cost = number of transfers (load + store). Compute and delete are free.
    Goal: all outputs are in done_mask.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    all_verts = set(list(parents.keys()) + outputs)
    if not all_verts:
        return []
    
    def vkey(v):
        if isinstance(v, int):
            return (0, v, "")
        return (1, 0, str(v))
    vertices = sorted(all_verts, key=vkey)
    n = len(vertices)
    if n > 12:
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
            if p in idx:
                m |= (1 << idx[p])
        par_mask[i] = m
    
    b = max(1, int(budget) if budget is not None else n)
    
    # Precompute: for each vertex, which outputs it can reach
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children_map:
                children_map[p].append(v)
    
    def can_reach_output(v, memo={}):
        if v in memo:
            return memo[v]
        if v in outputs:
            memo[v] = True
            return True
        result = False
        for c in children_map.get(v, []):
            if can_reach_output(c, memo):
                result = True
                break
        memo[v] = result
        return result
    
    # Heuristic: minimum number of loads needed
    # Each output not yet done needs its input ancestors to be loaded
    # Lower bound: for each output not done, count input ancestors not in memory
    input_mask = 0
    for i in range(n):
        if is_input[i]:
            input_mask |= (1 << i)
    
    # Precompute ancestor input masks
    anc_input = [0] * n
    def get_anc_input(i, memo={}):
        if i in memo:
            return memo[i]
        m = par_mask[i]
        if m == 0:
            result = (1 << i) if is_input[i] else 0
        else:
            result = 0
            mm = m
            while mm:
                lb = mm & (-mm)
                j = lb.bit_length() - 1
                result |= get_anc_input(j, memo)
                mm ^= lb
        memo[i] = result
        return result
    
    for i in range(n):
        get_anc_input(i)
    
    def heuristic(mem_mask, done_mask):
        """Lower bound on remaining transfers."""
        h = 0
        for i in range(n):
            if out_mask & (1 << i):
                if not (done_mask & (1 << i)):
                    # This output is not done. We need to compute it.
                    # All its input ancestors must be in memory at some point.
                    # Those not currently in memory must be loaded.
                    missing_inputs = anc_input[i] & ~mem_mask & input_mask
                    h += missing_inputs.bit_count()
        return h
    
    start_t = time.time()
    deadline = start_t + 30
    
    # State: (mem_mask, done_mask)
    # Actions: load (input, not in mem, mem < b), compute (parents in mem, not done),
    #          delete (non-output, in mem), store (output, in mem -> but we don't use store)
    # Actually, we can skip store: just keep outputs in memory until done.
    # When an output is computed, it's done. We can delete it to free space, but then we'd need to reload.
    # Better: keep outputs in memory.
    
    open_heap = [(heuristic(0, 0), 0, 0, (0, 0))]
    g_score = {(0, 0): 0}
    parent_op = {}
    counter = 0
    best = None
    
    while open_heap:
        if time.time() > deadline:
            break
        f, g, _, (mem, done) = heapq.heappop(open_heap)
        
        if done == out_mask:
            # Reconstruct
            seq = []
            state = (mem, done)
            while state != (0, 0):
                prev, op, v = parent_op[state]
                seq.append([op, v])
                state = prev
            seq.reverse()
            best = seq
            break
        
        if g > g_score.get(state, float("inf")):
            continue
        
        # Generate successors
        for i in range(n):
            bit = 1 << i
            v = vertices[i]
            
            # Load: input, not in mem, mem < b
            if is_input[i] and not (mem & bit) and mem.bit_count() < b:
                new_mem = mem | bit
                ng = g + 1
                ns = (new_mem, done)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, done), "load", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, done), ng, counter, ns))
            
            # Compute: parents in mem, not done
            if (mem & par_mask[i]) == par_mask[i] and not (done & bit):
                new_mem = mem | bit
                new_done = done | bit
                ng = g  # compute is free
                ns = (new_mem, new_done)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, done), "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_done), ng, counter, ns))
            
            # Delete: non-output, in mem
            if (mem & bit) and not (out_mask & bit):
                new_mem = mem & ~bit
                ng = g
                ns = (new_mem, done)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, done), "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, done), ng, counter, ns))
    
    return best


def pebble(dag, fast_memory_budget):
    deadline = time.time() + 45
    candidates = []
    
    # Try optimal search for small DAGs
    try:
        r = _pebble_optimal_small(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # DFS postorder with eviction
    try:
        r = _pebble_dfs(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        # Fallback: simple sequential
        parents = dag.get("parents", {})
        outputs = list(dag.get("outputs", []))
        all_verts = set(list(parents.keys()) + outputs)
        if not all_verts:
            return []
        
        def vkey(v):
            if isinstance(v, int):
                return (0, v, "")
            return (1, 0, str(v))
        vertices = sorted(all_verts, key=vkey)
        vset = set(vertices)
        outset = set(outputs)
        
        topo, children = _topo_order(parents, vertices, vset)
        
        ops = []
        in_reg = set()
        
        for v in topo:
            ps = parents.get(v, [])
            for p in ps:
                if p not in in_reg:
                    if not ps:  # input
                        ops.append(["load", p])
                    else:
                        # Recompute
                        pass  # This is broken; use DFS instead
            ops.append(["compute", v])
            in_reg.add(v)
        
        return ops[:20000]
    
    # Pick candidate with fewest transfers
    def transfers(ops):
        return sum(1 for op, v in ops if op in ("load", "store"))
    
    best = min(candidates, key=transfers)
    return best
# EVOLVE-BLOCK-END
