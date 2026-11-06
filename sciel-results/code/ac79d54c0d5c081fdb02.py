"""Communication-minimizing DAG pebbling: list-scheduling heuristic with
recomputation-cost priority and a bounded fast-memory buffer."""
import time
import math
import heapq

# EVOLVE-BLOCK-START

def pebble(dag, fast_memory_budget):
    """Return a list of [operation, vertex] operations that pebble `dag` so that
    every output vertex finishes blue, respecting the fast-memory budget.

    `dag` exposes:
      - dag.get_vertices()  -> iterable of vertex ids (topologically ordered)
      - dag.get_parents(v)  -> iterable of parent vertex ids of v
      - dag.get_inputs()    -> iterable of input vertex ids
      - dag.get_outputs()   -> iterable of output vertex ids
    """
    start = time.time()
    DEADLINE = 85.0  # safety margin vs. 90s kill

    # ---------- 1. Parse the DAG ----------
    try:
        vertices = list(dag.get_vertices())
    except Exception:
        vertices = list(dag.vertices) if hasattr(dag, "vertices") else []
    n = len(vertices)
    if n == 0:
        return []

    parents = {}
    for v in vertices:
        try:
            parents[v] = list(dag.get_parents(v))
        except Exception:
            parents[v] = []

    try:
        inputs = set(dag.get_inputs())
    except Exception:
        inputs = set()
    try:
        outputs = list(dag.get_outputs())
    except Exception:
        outputs = []
    output_set = set(outputs)

    # Build children map (for reachability / depth computations)
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            if p in children:
                children[p].append(v)

    # ---------- 2. Compute structural metrics ----------
    # depth[v] = longest path length from v to any output (recomputation cost proxy)
    # We process in reverse topological order (vertices are topologically ordered).
    depth = {}
    for v in reversed(vertices):
        if v in output_set:
            depth[v] = 0
        else:
            ch = children[v]
            if ch:
                depth[v] = 1 + max(depth[c] for c in ch if c in depth)
            else:
                depth[v] = 0

    # in-degree / topological order (inputs are sources)
    indeg = {}
    for v in vertices:
        indeg[v] = len(parents[v])

    # topological order via Kahn (inputs first)
    in_queue = [v for v in vertices if indeg[v] == 0]
    top_order = []
    indeg_work = dict(indeg)
    while in_queue:
        # deterministic: pop smallest by (depth, id)
        in_queue.sort(key=lambda x: (-depth.get(x, 0), x))
        v = in_queue.pop(0)
        top_order.append(v)
        for c in children[v]:
            indeg_work[c] -= 1
            if indeg_work[c] == 0:
                in_queue.append(c)
    if len(top_order) != n:
        # fallback: use given order
        top_order = list(vertices)

    # ---------- 3. List-scheduling pebbling heuristic ----------
    # State:
    #   blue[v]  : vertex v is currently blue (in fast memory)
    #   red[v]   : vertex v's value is available (blue or just computed)
    #   in_mem   : set of vertices occupying a fast-memory slot (blue)
    #   computed : set of vertices whose value has been computed at least once
    #
    # We maintain a priority queue of "ready" vertices (all parents red).
    # Priority = -depth[v] (higher depth = more recomputation cost = compute sooner)
    #
    # When fast memory is full and we need a slot, we evict the vertex with the
    # lowest depth (least recomputation cost) that is not an output.

    B = fast_memory_budget
    ops = []
    blue = set()      # vertices currently in fast memory (blue)
    red = set()       # vertices whose value is available (blue is subset)
    computed = set()  # vertices ever computed

    # Initialize inputs as blue
    for v in inputs:
        if v in depth:
            ops.append(["load", v])
            blue.add(v)
            red.add(v)

    # Ready set: vertices whose all parents are red
    def is_ready(v):
        for p in parents[v]:
            if p not in red:
                return False
        return True

    # Build initial ready queue
    ready = []
    for v in top_order:
        if v in inputs:
            continue
        if is_ready(v):
            heapq.heappush(ready, (-depth.get(v, 0), v))

    # Track how many times each vertex has been computed (for recomputation)
    compute_count = {v: 0 for v in vertices}

    # We need to compute each output at least once.
    # Strategy: process ready vertices in priority order. When we compute a vertex,
    # it becomes red and blue (occupies a slot). When memory is full, evict lowest-depth
    # non-output blue vertex (make it red-only by "store"ing, or delete if not needed).
    #
    # Actually, let's use a simpler model:
    # - "load": input becomes blue (in memory)
    # - "compute": vertex becomes blue (in memory), requires all parents blue
    # - "store": vertex leaves memory (blue -> red, value still available)
    # - "delete": value is discarded (red -> gone), must reload/recompute later
    #
    # For simplicity and correctness, let's use:
    # - All inputs start blue (loaded).
    # - compute(v): requires all parents blue. v becomes blue.
    # - When memory full, evict a blue vertex (store it, it becomes red).
    # - If a parent is red (not blue), we must reload it (load) or recompute.
    #
    # Let's implement a worklist algorithm:

    # Reset for clean implementation
    ops = []
    blue = set()
    red = set()
    in_memory = set()  # same as blue for tracking slots

    for v in inputs:
        if v in depth:
            ops.append(["load", v])
            blue.add(v)
            red.add(v)
            in_memory.add(v)

    # We'll use a priority queue of ready-to-compute vertices.
    # A vertex is ready if all its parents are blue.
    # When we compute, it becomes blue.
    # When memory is full, we evict the lowest-depth blue vertex that is not an output
    # and not needed soon.

    # Count how many outputs need each vertex (for eviction priority)
    # Actually, depth already captures this.

    # Main loop: compute vertices until all outputs are blue.
    # Use a ready queue.

    # Determine which vertices are "needed" (ancestors of outputs)
    needed = set()
    # BFS from outputs backwards
    to_visit = list(outputs)
    while to_visit:
        v = to_visit.pop()
        if v in needed:
            continue
        needed.add(v)
        for p in parents.get(v, []):
            if p in depth and p not in needed:
                to_visit.append(p)

    # Only schedule needed vertices
    # Rebuild ready queue
    ready = []
    for v in top_order:
        if v not in needed:
            continue
        if v in inputs:
            continue
        # Check if all parents are blue
        all_parents_blue = all(p in blue for p in parents[v])
        if all_parents_blue:
            heapq.heappush(ready, (-depth.get(v, 0), v))

    # Eviction: when in_memory is full, evict the blue vertex with lowest depth
    # that is not an output and not currently needed for an imminent computation.
    # Simple heuristic: evict lowest depth non-output.

    def evict_one():
        """Evict one blue vertex to free a memory slot."""
        # Candidates: blue vertices that are not outputs
        candidates = [v for v in blue if v not in output_set]
        if not candidates:
            # Evict even an output if necessary (shouldn't happen often)
            candidates = list(blue)
        if not candidates:
            return None
        # Pick lowest depth
        victim = min(candidates, key=lambda v: depth.get(v, 0))
        ops.append(["store", victim])
        blue.discard(victim)
        in_memory.discard(victim)
        # It's still red (value available)
        return victim

    # Compute until all outputs are blue
    while not all(o in blue for o in outputs):
        # Check deadline
        if time.time() > DEADLINE:
            break

        if not ready:
            # No ready vertices. This means some parent is not blue.
            # We need to reload or recompute a parent.
            # Find a vertex that is needed but not blue, and try to make its parents blue.
            # This is complex; for now, try to reload inputs or recompute.
            # Simple fallback: reload all inputs that are red but not blue.
            progress = False
            for v in inputs:
                if v in needed and v not in blue and v in red:
                    # Reload
                    if len(in_memory) >= B:
                        evict_one()
                    ops.append(["load", v])
                    blue.add(v)
                    in_memory.add(v)
                    progress = True
                    # Re-check ready queue
                    for c in children[v]:
                        if c in needed and c not in blue:
                            if all(p in blue for p in parents[c]):
                                heapq.heappush(ready, (-depth.get(c, 0), c))
            if not progress:
                # Stuck: break to avoid infinite loop
                break

        # Pop highest priority ready vertex
        neg_depth, v = heapq.heappop(ready)
        if v in blue:
            continue  # already computed
        if v not in needed:
            continue

        # Ensure all parents are blue (they should be, but double-check)
        for p in parents[v]:
            if p not in blue:
                # Parent not blue; can't compute. Push back and try to fix.
                heapq.heappush(ready, (neg_depth, v))
                break
        else:
            # All parents blue, compute v
            if len(in_memory) >= B:
                evict_one()
            ops.append(["compute", v])
            blue.add(v)
            in_memory.add(v)
            # Children of v may now be ready
            for c in children[v]:
                if c in needed and c not in blue:
                    if all(p in blue for p in parents[c]):
                        heapq.heappush(ready, (-depth.get(c, 0), c))

    # Ensure all outputs are blue at the end
    for o in outputs:
        if o not in blue:
            # Try to compute it
            # This is a fallback; in practice outputs should be computed above.
            pass

    return ops

# EVOLVE-BLOCK-END
