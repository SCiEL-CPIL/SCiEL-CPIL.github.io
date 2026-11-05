# EVOLVE-BLOCK-START
import time
import heapq
import sys

def pebble(dag, fast_memory_budget):
    """
    Minimize total transfers (load+store) for DAG pebbling.
    Returns: list of [operation, vertex]
    """
    deadline = time.time() + 85  # 5s safety margin

    # Parse DAG structure - handle multiple possible formats
    if isinstance(dag, dict):
        n = dag.get('n', dag.get('num_vertices', len(dag.get('parents', []))))
        parents = dag.get('parents', [])
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        if not parents:
            # Try other keys
            parents = dag.get('edges', [])
    else:
        # Assume tuple/list format: (n, parents, inputs, outputs)
        n = dag[0]
        parents = dag[1]
        inputs = dag[2]
        outputs = dag[3]

    n = int(n)
    parents = list(parents)
    inputs = list(inputs)
    outputs = list(outputs)

    # Ensure parents is list of lists
    if len(parents) != n:
        parents = [parents[i] if i < len(parents) else [] for i in range(n)]

    # Build children
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            if p < n:
                children[p].append(v)

    # Determine inputs/outputs if not specified
    if not inputs:
        inputs = [v for v in range(n) if not parents[v]]
    if not outputs:
        outputs = [v for v in range(n) if not children[v]]

    inputs = [int(x) for x in inputs]
    outputs = [int(x) for x in outputs]

    # Edge case: empty DAG
    if n == 0:
        return []

    # If budget >= n, we can hold everything - just load all inputs, compute all, store outputs
    if fast_memory_budget >= n:
        return _simple_schedule(n, parents, children, inputs, outputs)

    # A* search
    result = _astar_pebble(n, parents, children, inputs, outputs, fast_memory_budget, deadline)
    if result is not None:
        return result

    # Fallback: greedy schedule
    return _greedy_schedule(n, parents, children, inputs, outputs, fast_memory_budget)


def _simple_schedule(n, parents, children, inputs, outputs):
    """Simple topological schedule when budget is large."""
    ops = []
    input_set = set(inputs)
    output_set = set(outputs)
    for v in range(n):
        if v in input_set:
            ops.append(['load', v])
        ops.append(['compute', v])
        if v in output_set:
            ops.append(['store', v])
    return ops


def _greedy_schedule(n, parents, children, inputs, outputs, budget):
    """Greedy schedule: process in topological order, manage memory greedily."""
    ops = []
    red = set()  # vertices currently in fast memory
    input_set = set(inputs)
    output_set = set(outputs)

    # Precompute: for each vertex, which outputs depend on it
    def depends_on_output(v):
        # BFS/DFS from v to see if we reach any output
        visited = set()
        stack = [v]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            if u in output_set:
                return True
            for c in children[u]:
                if c not in visited:
                    stack.append(c)
        return False

    # Process in topological order
    for v in range(n):
        # Load if input
        if v in input_set:
            ops.append(['load', v])
            red.add(v)

        # Compute
        ops.append(['compute', v])
        red.add(v)

        # Check if we need to evict to stay within budget
        while len(red) > budget:
            # Evict the vertex that is least likely to be needed again
            # Heuristic: evict non-output vertices first, then those with fewest remaining dependents
            candidates = [x for x in red if x not in output_set]
            if not candidates:
                # Must evict an output? That shouldn't happen if we store outputs
                candidates = list(red)

            # Score each candidate: how many remaining vertices depend on it
            def eviction_score(x):
                # Count how many future vertices (in topological order after x) depend on x
                count = 0
                for u in range(x + 1, n):
                    # Check if x is an ancestor of u
                    # Simple check: BFS from u backwards
                    # This is expensive; use a heuristic
                    pass
                # Just use: is it an output? (don't evict outputs)
                # For now, evict the smallest index (earliest in topo order)
                return x

            # Evict the one with lowest "need" - use topological index as proxy
            evict = min(candidates, key=lambda x: (x not in output_set, x))
            if evict in output_set and evict not in outputs:
                # Shouldn't happen
                pass
            ops.append(['delete', evict])
            red.discard(evict)

        # Store if output
        if v in output_set:
            ops.append(['store', v])
            red.discard(v)
        elif v not in output_set and not depends_on_output(v):
            # Delete if not needed for any output
            ops.append(['delete', v])
            red.discard(v)

    return ops


def _astar_pebble(n, parents, children, inputs, outputs, budget, deadline):
    """A* search over memory states."""
    # Precompute
    parents_bitmask = []
    for v in range(n):
        pm = 0
        for p in parents[v]:
            if p < n:
                pm |= (1 << p)
        parents_bitmask.append(pm)

    input_set = set(inputs)
    output_set = set(outputs)
    output_bitmask = 0
    for o in outputs:
        if o < n:
            output_bitmask |= (1 << o)

    # Precompute ancestors
    ancestors = [0] * n
    for v in range(n):
        anc = 0
        for p in parents[v]:
            if p < n:
                anc |= (1 << p) | ancestors[p]
        ancestors[v] = anc

    # For each output, which inputs are ancestors
    output_input_anc = {}
    for o in outputs:
        if o >= n:
            continue
        anc = ancestors[o]
        ia = 0
        for i in inputs:
            if i < n and (anc & (1 << i)):
                ia |= (1 << i)
        output_input_anc[o] = ia

    # Heuristic: lower bound on remaining transfers
    def heuristic(red_mask, done_mask):
        stores = 0
        union_inputs = 0
        for o in outputs:
            if o >= n:
                continue
            if not (done_mask & (1 << o)):
                stores += 1
                union_inputs |= output_input_anc.get(o, 0) & ~red_mask
        loads = bin(union_inputs).count('1')
        return stores + loads

    # Initial state
    init_red = 0
    init_done = 0
    init_state = (init_red, init_done)

    best_g = {init_state: 0}
    counter = 0
    open_list = [(0, 0, 0, init_state, [])]

    best_solution = None
    best_cost = float('inf')
    expansions = 0
    max_expansions = 200000

    while open_list:
        if time.time() > deadline:
            break
        if expansions >= max_expansions:
            break

        f, g, _, state, path = heapq.heappop(open_list)
        red_mask, done_mask = state

        if done_mask == output_bitmask:
            if g < best_cost:
                best_cost = g
                best_solution = path[:]
            break

        if best_g.get(state, float('inf')) < g:
            continue

        expansions += 1

        # Generate moves
        # 1. Load inputs
        for i in inputs:
            if i >= n:
                continue
            bit = 1 << i
            if not (red_mask & bit):
                new_red = red_mask | bit
                if bin(new_red).count('1') <= budget:
                    ns = (new_red, done_mask)
                    ng = g + 1
                    if ng < best_g.get(ns, float('inf')):
                        best_g[ns] = ng
                        h = heuristic(new_red, done_mask)
                        counter += 1
                        heapq.heappush(open_list, (ng + h, ng, counter, ns, path + [['load', i]]))

        # 2. Compute
        for v in range(n):
            bit = 1 << v
            if red_mask & bit:
                continue
            pm = parents_bitmask[v]
            if pm & ~red_mask == 0:
                new_red = red_mask | bit
                if bin(new_red).count('1') <= budget:
                    ns = (new_red, done_mask)
                    ng = g
                    if ng < best_g.get(ns, float('inf')):
                        best_g[ns] = ng
                        h = heuristic(new_red, done_mask)
                        counter += 1
                        heapq.heappush(open_list, (ng + h, ng, counter, ns, path + [['compute', v]]))

        # 3. Store outputs
        for o in outputs:
            if o >= n:
                continue
            bit = 1 << o
            if (red_mask & bit) and not (done_mask & bit):
                new_red = red_mask & ~bit
                new_done = done_mask | bit
                ns = (new_red, new_done)
                ng = g + 1
                if ng < best_g.get(ns, float('inf')):
                    best_g[ns] = ng
                    h = heuristic(new_red, new_done)
                    counter += 1
                    heapq.heappush(open_list, (ng + h, ng, counter, ns, path + [['store', o]]))

        # 4. Delete (only non-output or done outputs, and only if not needed)
        for v in range(n):
            bit = 1 << v
            if not (red_mask & bit):
                continue
            if v in output_set and not (done_mask & bit):
                continue
            # Check if needed for any not-done output
            needed = False
            for o in outputs:
                if o >= n:
                    continue
                if not (done_mask & (1 << o)):
                    if ancestors[o] & bit:
                        needed = True
                        break
            if needed:
                continue
            new_red = red_mask & ~bit
            ns = (new_red, done_mask)
            ng = g
            if ng < best_g.get(ns, float('inf')):
                best_g[ns] = ng
                h = heuristic(new_red, done_mask)
                counter += 1
                heapq.heappush(open_list, (ng + h, ng, counter, ns, path + [['delete', v]]))

    return best_solution
# EVOLVE-BLOCK-END
