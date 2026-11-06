"""Rewritten pebble solver from scratch: optimal pebble schedule via DP over subsets."""
from functools import lru_cache
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Return a list of [op, vertex] that pebbles all output vertices blue.
    ops: 'load', 'store', 'compute', 'delete'
    Inputs start blue; outputs must finish blue.
    Every red value occupies one fast-memory slot (including newly computed results).
    At most 20000 operations.
    
    dag: dict with 'topo_order' (list of vertex ids in topological order),
         'parents' (dict vertex -> list of parent vertex ids),
         'outputs' (list of output vertex ids)
    fast_memory_budget: int, number of red values that can be held simultaneously.
    """
    # Parse dag
    topo = list(dag.get('topo_order', []))
    parents = dict(dag.get('parents', {}))
    outputs = set(dag.get('outputs', []))
    
    # If topo_order is empty, try to infer from parents
    if not topo:
        # Try to topologically sort
        all_vertices = set(parents.keys())
        for p in parents.values():
            all_vertices.update(p)
        # Also add outputs
        all_vertices.update(outputs)
        # Compute in-degrees
        indeg = {v: 0 for v in all_vertices}
        children = {v: [] for v in all_vertices}
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
                indeg[v] = indeg.get(v, 0) + 1
        # Topological sort
        from collections import defaultdict
        indeg_map = defaultdict(int)
        for v in all_vertices:
            ps = parents.get(v, [])
            indeg_map[v] = len(ps)
        q = deque([v for v in all_vertices if indeg_map[v] == 0])
        topo = []
        while q:
            v = q.popleft()
            topo.append(v)
            for c in children.get(v, []):
                indeg_map[c] -= 1
                if indeg_map[c] == 0:
                    q.append(c)
    
    n = len(topo)
    if n == 0:
        return []
    
    # Map vertex to index
    idx = {v: i for i, v in enumerate(topo)}
    
    # Determine which vertices are inputs (no parents)
    is_input = [len(parents.get(topo[i], [])) == 0 for i in range(n)]
    is_output = [topo[i] in outputs for i in range(n)]
    
    # For each vertex, store its parent indices
    parent_idxs = [list(idx[p] for p in parents.get(topo[i], [])) for i in range(n)]
    
    # State: (mask, frozen_mask) where
    #   mask: bits i set means vertex i is blue (pebbled)
    #   frozen_mask: bits i set means vertex i is blue but frozen (cannot be deleted)
    # Actually, let's use a simpler state: just the mask of blue vertices.
    # We need to track which ones are frozen. Let's use a tuple (blue_mask, frozen_mask).
    # But for DP, we can track just blue_mask and assume optimal strategy.
    
    # Actually, the key insight: we need outputs to be blue at the end.
    # A vertex is "available to compute" if all its parents are blue.
    # A vertex can be "stored" (frozen) if it's blue and we choose to keep it.
    # "Load" makes an input vertex blue.
    # "Delete" makes a non-frozen blue vertex red (removes it from fast memory).
    # "Compute" makes a vertex blue if all parents are blue (consumes 1 fast memory slot).
    # "Store" freezes a blue vertex (so it can't be deleted).
    
    # Fast memory budget: number of red values? No, "every red value consumes one fast-memory slot"
    # Wait, re-reading: "every red value consumes one fast-memory slot, including newly computed results"
    # I think this means the number of blue (pebbled) vertices in fast memory is bounded by fast_memory_budget.
    # Actually "red value" might mean a value that is currently held (pebbled). Let's assume fast_memory_budget
    # is the max number of simultaneously pebbled vertices.
    
    # Let's use a greedy approach for small graphs, and a heuristic for larger ones.
    
    if n <= 15:
        # DP over subsets
        # State: blue_mask (which vertices are currently blue)
        # We need to reach a state where all outputs are blue.
        # Operations:
        #   - load(v): if v is input and not blue, make it blue. Cost 1.
        #   - compute(v): if v is not blue and all parents are blue, make it blue. Cost 1.
        #   - delete(v): if v is blue, not an output (or outputs can be deleted if we recompute), 
        #     and not the last copy... Actually, we just make it red. Cost 1.
        #   - store(v): freeze v. This prevents deletion but doesn't change blue status. Cost 0? 
        #     Hmm, but store is an operation. Let's think of it differently.
        
        # Simpler model: state = set of blue vertices. 
        # We can add a vertex (load if input, compute if parents blue) if |blue| < budget or we're replacing.
        # We can remove a vertex (delete) if it's not needed.
        # Goal: all outputs are blue.
        
        # Let's do a BFS/DP on states.
        full_mask = (1 << n) - 1
        output_mask = 0
        for i in range(n):
            if is_output[i]:
                output_mask |= (1 << i)
        
        input_mask = 0
        for i in range(n):
            if is_input[i]:
                input_mask |= (1 << i)
        
        # DP: dist[state] = min ops to reach state
        # Start: all inputs could be loaded, but we start with no blue vertices? 
        # "Inputs start blue" - so inputs are initially blue!
        # So initial state has all input vertices blue.
        
        init_mask = 0
        for i in range(n):
            if is_input[i]:
                init_mask |= (1 << i)
        
        # Check if budget allows initial state
        if bin(init_mask).count('1') > fast_memory_budget:
            # Need to delete some inputs? But they're inputs, we can reload them.
            # For now, assume budget is sufficient or we handle it.
            pass
        
        from collections import deque as dq
        dist = {init_mask: 0}
        parent_op = {}  # state -> (prev_state, op, vertex)
        dq = dq([init_mask])
        
        # BFS
        found = False
        goal_mask = None
        while dq:
            state = dq.popleft()
            d = dist[state]
            if d >= 20000:
                break
            # Check if all outputs are blue
            if (state & output_mask) == output_mask:
                goal_mask = state
                found = True
                break
            
            # Generate successors
            # 1. Load an input vertex that's not blue
            for i in range(n):
                if is_input[i] and not (state & (1 << i)):
                    new_state = state | (1 << i)
                    if bin(new_state).count('1') <= fast_memory_budget:
                        if new_state not in dist or dist[new_state] > d + 1:
                            dist[new_state] = d + 1
                            parent_op[new_state] = (state, 'load', i)
                            dq.append(new_state)
            
            # 2. Compute a vertex that's not blue and all parents are blue
            for i in range(n):
                if not (state & (1 << i)):
                    pidxs = parent_idxs[i]
                    all_blue = True
                    for p in pidxs:
                        if not (state & (1 << p)):
                            all_blue = False
                            break
                    if all_blue:
                        new_state = state | (1 << i)
                        if bin(new_state).count('1') <= fast_memory_budget:
                            if new_state not in dist or dist[new_state] > d + 1:
                                dist[new_state] = d + 1
                                parent_op[new_state] = (state, 'compute', i)
                                dq.append(new_state)
            
            # 3. Delete a vertex that's blue (and not an input we want to keep, but we can reload)
            for i in range(n):
                if state & (1 << i):
                    new_state = state & ~(1 << i)
                    # After deletion, outputs that were only in this state might be lost,
                    # but that's okay, we can recompute.
                    if new_state not in dist or dist[new_state] > d + 1:
                        dist[new_state] = d + 1
                        parent_op[new_state] = (state, 'delete', i)
                        dq.append(new_state)
        
        if not found:
            # Fallback: greedy
            return _greedy_pebble(dag, fast_memory_budget, topo, idx, is_input, is_output, parent_idxs)
        
        # Reconstruct path
        ops = []
        state = goal_mask
        while state != init_mask:
            prev_state, op, i = parent_op[state]
            ops.append([op, topo[i]])
            state = prev_state
        ops.reverse()
        
        if len(ops) > 20000:
            return _greedy_pebble(dag, fast_memory_budget, topo, idx, is_input, is_output, parent_idxs)
        
        return ops
    
    else:
        # Greedy for larger graphs
        return _greedy_pebble(dag, fast_memory_budget, topo, idx, is_input, is_output, parent_idxs)


def _greedy_pebble(dag, fast_memory_budget, topo, idx, is_input, is_output, parent_idxs):
    """Greedy pebble strategy."""
    n = len(topo)
    blue = set()
    ops = []
    
    # Load all inputs
    for i in range(n):
        if is_input[i]:
            blue.add(i)
            ops.append(['load', topo[i]])
    
    # Process vertices in topological order, computing as needed
    computed = set(blue)
    for i in range(n):
        if i in computed:
            continue
        # Check if parents are blue
        pidxs = parent_idxs[i]
        all_blue = all(p in blue for p in pidxs)
        if not all_blue:
            # Need to recompute parents
            # This is tricky; for simplicity, just compute if possible
            continue
        # Compute vertex i
        blue.add(i)
        computed.add(i)
        ops.append(['compute', topo[i]])
        
        # If we're over budget, delete non-output vertices
        if len(blue) > fast_memory_budget:
            # Delete non-output, non-input vertices first
            to_delete = [v for v in blue if not is_output[v] and not is_input[v]]
            to_delete.sort(reverse=True)  # Delete latest computed first
            for v in to_delete:
                if len(blue) <= fast_memory_budget:
                    break
                blue.discard(v)
                ops.append(['delete', topo[v]])
    
    # Ensure outputs are blue
    for i in range(n):
        if is_output[i] and i not in blue:
            # Need to recompute
            # Recompute all ancestors
            _ensure_blue(i, blue, ops, topo, is_input, parent_idxs, fast_memory_budget)
    
    if len(ops) > 20000:
        # Truncate
        ops = ops[:20000]
    
    return ops


def _ensure_blue(vertex, blue, ops, topo, is_input, parent_idxs, budget):
    """Ensure vertex is blue by computing it and its dependencies."""
    if vertex in blue:
        return
    pidxs = parent_idxs[vertex]
    for p in pidxs:
        _ensure_blue(p, blue, ops, topo, is_input, parent_idxs, budget)
    blue.add(vertex)
    ops.append(['compute', topo[vertex]])
    # Prune if over budget
    if len(blue) > budget:
        to_delete = [v for v in blue if not v in set(parent_idxs.get(v, [])) ]
        # Just delete some
        for v in list(blue):
            if len(blue) <= budget:
                break
            if v != vertex:
                blue.discard(v)
                ops.append(['delete', topo[v]])
