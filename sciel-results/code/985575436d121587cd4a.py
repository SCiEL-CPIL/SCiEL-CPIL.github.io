"""A*/Dijkstra state-space search for communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    # ---- Parse the DAG structure ----
    # dag: dict or object. We support common formats:
    #   dag['vertices'] or dag['n']: number of vertices
    #   dag['parents'][v]: list of parents (topologically ordered)
    #   dag['inputs']: set/list of input vertices (start blue)
    #   dag['outputs']: set/list of output vertices (must finish blue)
    #   dag['topo']: topological order (list of vertices)
    # We try to be robust to different representations.

    def _get(dag, key, default=None):
        if isinstance(dag, dict):
            return dag.get(key, default)
        return getattr(dag, key, default)

    n = _get(dag, 'n', None)
    if n is None:
        n = _get(dag, 'num_vertices', None)
    if n is None:
        verts = _get(dag, 'vertices', None)
        if verts is not None:
            n = len(verts)
    if n is None:
        parents = _get(dag, 'parents', {})
        if isinstance(parents, dict):
            n = len(parents)
        else:
            n = len(parents)
    if n is None:
        n = 0

    parents = _get(dag, 'parents', {})
    if isinstance(parents, dict):
        parent_list = [list(parents.get(v, [])) for v in range(n)]
    else:
        parent_list = [list(p) for p in parents]
    # ensure length n
    while len(parent_list) < n:
        parent_list.append([])

    topo = _get(dag, 'topo', None)
    if topo is None:
        topo = _get(dag, 'topological_order', None)
    if topo is None:
        topo = list(range(n))

    inputs = _get(dag, 'inputs', None)
    if inputs is None:
        inputs = _get(dag, 'input_vertices', None)
    if inputs is None:
        inputs = set()
    if isinstance(inputs, (list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    outputs = _get(dag, 'outputs', None)
    if outputs is None:
        outputs = _get(dag, 'output_vertices', None)
    if outputs is None:
        outputs = set()
    if isinstance(outputs, (list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build adjacency: children of each vertex
    children = defaultdict(list)
    for v in range(n):
        for p in parent_list[v]:
            children[p].append(v)

    # A vertex is "computable" if it has no parents (input) OR all parents are blue.
    # We track blue (in fast memory) as a bitmask.

    # ---- Precompute for heuristic: for each vertex, the set of vertices that
    #      depend on it (transitively), to estimate necessary recomputations. ----
    # For the heuristic we use: number of outputs not yet blue that are
    # descendants of currently-stored values that we might lose. A simpler
    # admissible heuristic: 0 is admissible but weak. We use a better one:
    # the minimum number of loads needed = number of input vertices not blue
    # that are needed. We'll use a lightweight heuristic.

    # Precompute descendant count to outputs for heuristic weighting
    # (used only for tie-breaking / heuristic, must remain admissible).

    # ---- A* over states ----
    # State: (blue_mask, ) where blue_mask is a bitmask of vertices in fast memory.
    # We also need to know which computed values are "available" (in fast memory).
    # Actually the blue_mask fully captures fast-memory contents.
    # A vertex is blue iff its bit is set.
    # Initially: inputs are blue.
    # Goal: all outputs are blue.

    # Operations and their cost (transfers):
    #   load(v):  move v from slow to fast. Cost 1. Requires v computable and not blue.
    #   store(v): move v from fast to slow. Cost 1. Requires v blue and not output-needed... 
    #             Actually store just removes from fast memory. Cost 1.
    #   compute(v): compute v from its parents. Cost 0 (no transfer) but produces a blue
    #             value occupying a slot. Requires all parents blue and v not blue.
    #             Wait - compute produces a NEW blue value. If v was previously computed
    #             and stored, we recompute it. Cost 0 transfers but uses a slot.
    #   delete(v): free a slot without moving to slow. Cost 0. Requires v blue.
    #
    # Constraint: number of blue vertices <= fast_memory_budget.
    #
    # The goal is to minimize total transfers = loads + stores.
    #
    # To minimize transfers, we prefer compute (cost 0) over load (cost 1) whenever
    # possible, and prefer delete (cost 0) over store (cost 1) when freeing space.
    # But compute requires parents to be blue.

    # Initial state: inputs are blue (they start blue, i.e., available in fast memory? 
    # or in slow memory?). The problem says "Inputs start blue". Blue = in fast memory.
    # So inputs start in fast memory, occupying slots.
    init_mask = 0
    for v in inputs:
        if v < n:
            init_mask |= (1 << v)

    # If init_mask exceeds budget, we may need to store some inputs immediately.
    # But let's handle that in the search.

    # Goal check: all outputs blue.
    goal_mask = 0
    for v in outputs:
        if v < n:
            goal_mask |= (1 << v)

    def is_goal(mask):
        return (mask & goal_mask) == goal_mask

    # Heuristic: admissible lower bound on remaining transfers.
    # A vertex that is an output and not blue must eventually be blue.
    # To make it blue, either it's loaded (cost >= 1) or computed (cost 0 but
    # requires its parents blue, which may require transfers).
    # A safe admissible heuristic: 0. We use a slightly better one:
    # count outputs not blue that have no path of blue ancestors -> need at least 1 load each.
    # But computing that is expensive. We use 0 for safety (admissible) but add
    # a tie-breaker via secondary key.

    def heuristic(mask):
        # Admissible: number of output vertices not blue that are "isolated"
        # (none of their ancestors are blue and not an input). For those, at least
        # one transfer is needed. This is a lower bound.
        # To keep it cheap and admissible, return 0. We rely on Dijkstra/A* with
        # g-cost to find optimal, and use h for ordering only (must be admissible).
        # We'll use a very cheap admissible h: 0.
        return 0

    # We'll run A* (Dijkstra since h=0) with a secondary tie-breaker to prefer
    # states with more blue vertices (closer to goal) to speed up convergence.

    # To keep the state space manageable, we limit the search and use a good
    # ordering. We also cap operations at 20000.

    # State representation: (blue_mask)
    # We store: g (transfers so far), path (list of operations)
    # To avoid storing full paths in the heap (memory), we store parent pointers.

    # Given potential state-space explosion, we use a bounded search:
    # We'll use A* with a max number of expansions, and fall back to a greedy
    # topological schedule if the search doesn't finish in time.

    MAX_EXPANSIONS = 200000  # generous but bounded
    MAX_OPS = 20000

    # For efficient "can compute v" check, we precompute for each vertex the
    # bitmask of its parents.
    parent_mask = [0] * n
    for v in range(n):
        m = 0
        for p in parent_list[v]:
            if p < n:
                m |= (1 << p)
        parent_mask[v] = m

    # A vertex v is computable (all parents blue) iff (mask & parent_mask[v]) == parent_mask[v]
    # Note: inputs have parent_mask 0, so always computable.

    # We'll do A*. To keep memory bounded, we store best g per state.
    best_g = {}
    parent_info = {}  # state -> (prev_state, operation, vertex)

    # Priority queue: (f, g, tiebreak, mask)
    # tiebreak = -popcount(mask) to prefer more-blue states
    def popcount(x):
        return bin(x).count('1')

    start = init_mask
    # If start already exceeds budget, we must store some. We'll let the search
    # handle it, but to be safe, if popcount(start) > budget, we need to reduce.
    # We'll add a "forced store" step. Actually, let's just start and let the
    # search find a valid path; but the initial state must be valid (<= budget).
    # If not valid, we store inputs until valid.
    ops_forced = []
    cur = start
    while popcount(cur) > fast_memory_budget and fast_memory_budget >= 0:
        # store some non-output input
        stored = False
        for v in inputs:
            if (cur & (1 << v)) and v not in outputs:
                cur &= ~(1 << v)
                ops_forced.append(['store', v])
                stored = True
                break
        if not stored:
            # must store an output or we're stuck; store any
            for v in range(n):
                if cur & (1 << v):
                    cur &= ~(1 << v)
                    ops_forced.append(['store', v])
                    break
            else:
                break
    start = cur

    if is_goal(start):
        return ops_forced

    h0 = heuristic(start)
    heap = [(h0, 0, -popcount(start), start)]
    best_g[start] = 0
    parent_info[start] = None
    expansions = 0

    # Also track operations count along the path to enforce MAX_OPS.
    op_count = {start: len(ops_forced)}

    found = None
    while heap and expansions < MAX_EXPANSIONS:
        f, g, tie, mask = heapq.heappop(heap)
        if g > best_g.get(mask, float('inf')):
            continue
        expansions += 1

        if is_goal(mask):
            found = mask
            break

        # If operations so far + forced >= MAX_OPS, skip expanding (can't add more)
        if op_count.get(mask, 0) + len(ops_forced) >= MAX_OPS:
            continue

        # Generate neighbors
        # 1) compute(v): cost 0, if computable and not blue and popcount(mask)+1 <= budget
        # 2) load(v): cost 1, if computable (or input available) and not blue and popcount+1<=budget
        # 3) store(v): cost 1, if blue and not needed... (any blue)
        # 4) delete(v): cost 0, if blue

        # For efficiency, iterate over vertices.
        # Compute candidates: vertices not blue whose parents are all blue.
        # Load candidates: inputs not blue (load from slow), or computed-not-blue? 
        #   Actually load means the value exists in slow memory. Inputs are always
        #   available to load. Previously computed values that were stored can be loaded.
        #   For simplicity, allow load on any vertex that is "available in slow":
        #   inputs, or any vertex that was computed at least once. To keep it simple
        #   and admissible, allow load on inputs and on any vertex whose parents are
        #   all blue (i.e., it could have been computed and stored). This is a
        #   relaxation; the search will find optimal.
        #
        # To keep the search tractable, we restrict:
        #   - compute: any v not blue, parents blue, room available
        #   - load: any input v not blue, room available  (inputs always loadable)
        #   - store/delete: any v blue
        #
        # This is a valid relaxation that still allows optimal solutions because
        # any value can be produced by compute (cost 0) if its parents are blue,
        # so load is only needed for inputs whose parents aren't blue (none) — 
        # actually inputs have no parents, so compute would also work for inputs?
        # No: compute requires parents blue; inputs have no parents so compute is
        # always possible for inputs too. So load is redundant for inputs if
        # compute is free. But compute for an input that's already "available" 
        # is fine. So we can drop load entirely and use compute for everything!
        #
        # Insight: since compute has cost 0 and is always possible when parents
        # are blue, and inputs have no parents, we can always "compute" any vertex
        # whose parents are blue. Load (cost 1) is never needed if compute is free
        # and available. So optimal transfers come only from store operations
        # (to free space) — but store costs 1, delete costs 0. So we should
        # always prefer delete over store to free space!
        #
        # Wait, but delete loses the value (can't reload without recompute). Store
        # keeps it in slow memory (can reload). But reload = load = cost 1, and
        # recompute = cost 0 if parents still blue. So delete is always at least
        # as good as store, UNLESS the parents won't be blue later (then recompute
        # would need parents loaded). 
        #
        # This is getting complex. Let's allow all four operations and let A* find
        # the optimum. We restrict load to inputs and to vertices that are "slow-available"
        # (we track which have been computed/stored). To keep it simple, allow load
        # on any vertex (relaxation) — the cost model handles it.

        # We'll generate:
        # - compute(v) for v not blue, parents blue, room
        # - load(v) for v not blue, room  (relaxation: any v)
        # - store(v) for v blue
        # - delete(v) for v blue
        #
        # To reduce branching, we only consider load(v) for inputs and for v that
        # are "needed" (descendants of outputs). We'll allow load for inputs always,
        # and for others only if they have no blue parent (so compute wouldn't work).

        current_pop = popcount(mask)

        # compute(v)
        for v in range(n):
            if mask & (1 << v):
                continue
            if (mask & parent_mask[v]) != parent_mask[v]:
                continue
            if current_pop + 1 > fast_memory_budget:
                continue
            new_mask = mask | (1 << v)
            new_g = g  # cost 0
            if new_g < best_g.get(new_mask, float('inf')):
                best_g[new_mask] = new_g
                parent_info[new_mask] = (mask, ('compute', v))
                op_count[new_mask] = op_count.get(mask, 0)
                h = heuristic(new_mask)
                heapq.heappush(heap, (new_g + h, new_g, -popcount(new_mask), new_mask))

        # load(v): only for inputs (always available) and for vertices whose
        # parents are NOT all blue (so compute isn't an option) — relaxation.
        for v in inputs:
            if mask & (1 << v):
                continue
            if current_pop + 1 > fast_memory_budget:
                continue
            new_mask = mask | (1 << v)
            new_g = g + 1
            if new_g < best_g.get(new_mask, float('inf')):
                best_g[new_mask] = new_g
                parent_info[new_mask] = (mask, ('load', v))
                op_count[new_mask] = op_count.get(mask, 0) + 1
                h = heuristic(new_mask)
                heapq.heappush(heap, (new_g + h, new_g, -popcount(new_mask), new_mask))

        # store(v) and delete(v)
        for v in range(n):
            if not (mask & (1 << v)):
                continue
            # store
            new_mask = mask & ~(1 << v)
            new_g = g + 1
            if new_g < best_g.get(new_mask, float('inf')):
                best_g[new_mask] = new_g
                parent_info[new_mask] = (mask, ('store', v))
                op_count[new_mask] = op_count.get(mask, 0) + 1
                h = heuristic(new_mask)
                heapq.heappush(heap, (new_g + h, new_g, -popcount(new_mask), new_mask))
            # delete
            new_mask = mask & ~(1 << v)
            new_g = g  # cost 0
            if new_g < best_g.get(new_mask, float('inf')):
                best_g[new_mask] = new_g
                parent_info[new_mask] = (mask, ('delete', v))
                op_count[new_mask] = op_count.get(mask, 0)
                h = heuristic(new_mask)
                heapq.heappush(heap, (new_g + h, new_g, -popcount(new_mask), new_mask))

    if found is None:
        # Fallback: greedy topological schedule
        return _greedy_fallback(dag, fast_memory_budget, n, parent_list, inputs, outputs, ops_forced)

    # Reconstruct path
    path = []
    state = found
    while parent_info.get(state) is not None:
        prev, (op, v) = parent_info[state]
        path.append([op, v])
        state = prev
    path.reverse()
    result = ops_forced + path
    # Enforce max ops
    if len(result) > MAX_OPS:
        result = result[:MAX_OPS]
    return result


def _greedy_fallback(dag, fast_memory_budget, n, parent_list, inputs, outputs, ops_forced):
    """Simple topological schedule: load inputs, compute in topo order, store non-outputs."""
    ops = list(ops_forced)
    # Track blue
    blue = set(inputs)
    # Compute in topological order
    for v in range(n):
        if v in outputs:
            continue
        # ensure parents blue
        for p in parent_list[v]:
            if p not in blue:
                # load p
                ops.append(['load', p])
                blue.add(p)
        # compute v
        ops.append(['compute', v])
        blue.add(v)
        # if over budget, store some
        while len(blue) > fast_memory_budget:
            # store a non-output
            stored = False
            for x in list(blue):
                if x not in outputs:
                    ops.append(['store', x])
                    blue.discard(x)
                    stored = True
                    break
            if not stored:
                break
    # ensure outputs blue
    for v in outputs:
        if v not in blue:
            for p in parent_list[v]:
                if p not in blue:
                    ops.append(['load', p])
                    blue.add(p)
            ops.append(['compute', v])
            blue.add(v)
    return ops
# EVOLVE-BLOCK-END
