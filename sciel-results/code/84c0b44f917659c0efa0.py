"""DAG pebbling: communication-minimizing schedule via greedy lookahead cost model."""
import time

# EVOLVE-BLOCK-START

def _build_graph(dag):
    """
    Parse the DAG into adjacency structures.
    Expected dag format (from task): has 'vertices' or similar with topologically
    ordered parents and output vertices.
    """
    # Try to extract structure from various possible formats
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', list(range(dag.get('num_vertices', dag.get('n', 0))))))
        if 'parents' in dag:
            parents = dag['parents']
        elif 'preds' in dag:
            parents = dag['preds']
        elif 'edges' in dag:
            # Build parents from edges
            edges = dag['edges']
            parents = {v: [] for v in vertices}
            for u, v in edges:
                if v not in parents:
                    parents[v] = []
                parents[v].append(u)
        else:
            parents = {v: [] for v in vertices}
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_vertices', [])))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_vertices', [])))
        num_v = len(vertices)
        # Ensure all vertices in parents dict
        for v in vertices:
            if v not in parents:
                parents[v] = []
    elif isinstance(dag, (list, tuple)):
        # Assume list of vertices, parents derived from index (vertex i depends on 0..i-1?)
        # More likely: dag is list of lists where dag[i] = parents of vertex i
        parents = {}
        for i, p in enumerate(dag):
            if isinstance(p, (list, tuple)):
                parents[i] = list(p)
            else:
                parents[i] = []
        vertices = list(range(len(dag)))
        outputs = []
        inputs = []
        num_v = len(dag)
    else:
        # Fallback: assume dag is an int (number of vertices) or has attributes
        try:
            num_v = int(dag)
        except:
            num_v = getattr(dag, 'num_vertices', getattr(dag, 'n', 0))
        vertices = list(range(num_v))
        parents = {v: [] for v in vertices}
        outputs = []
        inputs = []

    # Determine inputs and outputs if not provided
    if not inputs:
        inputs = [v for v in vertices if len(parents.get(v, [])) == 0]
    if not outputs:
        # Outputs are vertices with no children
        children = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                children[p].append(v)
        outputs = [v for v in vertices if len(children[v]) == 0]

    return vertices, parents, children, inputs, outputs


def _naive_schedule_cost(dag, fast_memory_budget):
    """
    Estimate the cost of a naive sequential schedule:
    Process vertices in topological order. For each vertex:
    - Load all parent values that aren't in fast memory
    - Compute
    - Store result (if it's an output) or keep in fast memory
    - Delete parent values that won't be needed again
    """
    vertices, parents, children, inputs, outputs = _build_graph(dag)
    num_v = len(vertices)

    # Compute for each vertex how many times its value is needed (fanout)
    fanout = {v: len(children.get(v, [])) for v in vertices}

    # Topological order is just 0..num_v-1 (already topologically sorted per task)
    # Naive: load all parents, compute, store if output, delete parents not needed again
    cost = 0
    in_fast = set()

    for v in vertices:
        # Load parents not in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                cost += 1  # load
                in_fast.add(p)
        # Compute
        # Store if output
        if v in outputs:
            cost += 1  # store
        else:
            # Keep in fast memory (no cost for compute itself, but it occupies a slot)
            pass
        in_fast.add(v)

        # Delete parents that won't be needed again
        for p in parents.get(v, []):
            remaining = 0
            for c in children.get(p, []):
                if c > v:  # c not yet processed
                    remaining += 1
            if remaining == 0 and p not in outputs:
                in_fast.discard(p)

    # Store any outputs still in fast memory (shouldn't happen if we stored on compute)
    # But ensure all outputs are stored
    for v in outputs:
        if v in in_fast:
            cost += 1
            in_fast.discard(v)

    return cost


def pebble(dag, fast_memory_budget):
    """
    Generate a pebbling schedule for the DAG that minimizes memory transfers.
    Returns a list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'.
    """
    deadline = time.time() + 55  # Safety margin before 60s limit

    vertices, parents, children, inputs, outputs = _build_graph(dag)
    num_v = len(vertices)
    budget = int(fast_memory_budget) if fast_memory_budget is not None else num_v

    # If budget is large enough to hold everything, do a simple topological pass
    if budget >= num_v:
        ops = []
        for v in vertices:
            for p in parents.get(v, []):
                ops.append(['load', p])
            ops.append(['compute', v])
            if v in outputs:
                ops.append(['store', v])
            else:
                ops.append(['delete', v])
        return ops

    # Compute fanout (how many children each vertex has)
    fanout = {v: len(children.get(v, [])) for v in vertices}

    # For each vertex, compute its "importance" - how much reusing it saves
    # A vertex with high fanout is worth keeping in fast memory
    # Also consider depth: vertices used by many downstream vertices are important

    # Compute depth (longest path from inputs)
    depth = {}
    for v in vertices:
        if not parents.get(v):
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in parents[v]) + 1

    # Compute remaining uses: for each vertex, how many times it's needed after current position
    # This is just fanout for the simple case, but for layered DAGs it could be more complex

    # Greedy strategy:
    # Process vertices in topological order.
    # At each step, decide what to load, compute, and what to evict.
    # Use a cost model: prefer to keep vertices with high "value" in fast memory.
    # Value of keeping vertex v = number of future uses * (cost of reloading)
    # Cost of reloading = 1 (one transfer)
    # So value = remaining_uses[v]

    # We'll use a priority-based eviction: when fast memory is full, evict the vertex
    # with the lowest remaining uses.

    ops = []
    in_fast = set()
    computed = set()  # vertices that have been computed
    stored = set()  # outputs that have been stored

    # Precompute for each vertex the list of children (in topological order)
    # and for each vertex, the set of vertices that still need it at each point

    # For efficiency, track remaining uses dynamically
    remaining_uses = {v: fanout[v] for v in vertices}

    for v in vertices:
        # Check deadline
        if time.time() > deadline:
            # Emergency: just do a simple schedule for remaining
            for v2 in vertices[v:]:
                for p in parents.get(v2, []):
                    if p not in in_fast:
                        ops.append(['load', p])
                    in_fast.add(p)
                ops.append(['compute', v2])
                if v2 in outputs:
                    ops.append(['store', v2])
                    stored.add(v2)
                else:
                    ops.append(['delete', v2])
                    in_fast.discard(v2)
                for p in parents.get(v2, []):
                    remaining_uses[p] = max(0, remaining_uses[p] - 1)
                    if remaining_uses[p] == 0 and p not in outputs:
                        in_fast.discard(p)
            break

        # Load all parents not in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                # Need to load p
                if len(in_fast) >= budget:
                    # Evict the vertex with lowest remaining uses
                    # Don't evict v's other parents that are still needed
                    evict_candidates = [x for x in in_fast if x not in parents.get(v, []) and x != v]
                    if evict_candidates:
                        # Evict the one with lowest remaining uses
                        evict_v = min(evict_candidates, key=lambda x: remaining_uses.get(x, 0))
                        ops.append(['delete', evict_v])
                        in_fast.discard(evict_v)
                    else:
                        # Must evict one of the parents (will reload later)
                        # Evict the one with lowest remaining uses among parents
                        evict_v = min(parents.get(v, []), key=lambda x: remaining_uses.get(x, 0))
                        ops.append(['delete', evict_v])
                        in_fast.discard(evict_v)
                ops.append(['load', p])
                in_fast.add(p)

        # Compute v
        ops.append(['compute', v])
        computed.add(v)
        in_fast.add(v)

        # If v is an output, store it
        if v in outputs:
            ops.append(['store', v])
            stored.add(v)
            # After storing, we can delete it from fast memory if budget is tight
            # But keep it for now; we'll delete at the end if needed

        # Update remaining uses for parents
        for p in parents.get(v, []):
            remaining_uses[p] = max(0, remaining_uses[p] - 1)

        # If v is not an output and we have too many in fast memory, consider deleting
        # Actually, we'll let the eviction handle this at load time

    # Final cleanup: store any outputs not yet stored, delete non-outputs
    for v in outputs:
        if v not in stored:
            if v in in_fast:
                ops.append(['store', v])
            else:
                # Need to load it first (shouldn't happen if we computed it)
                ops.append(['load', v])
                ops.append(['store', v])
            stored.add(v)
            in_fast.discard(v)

    # Delete any remaining non-output vertices in fast memory
    for v in list(in_fast):
        if v not in outputs:
            ops.append(['delete', v])
            in_fast.discard(v)

    # Ensure all outputs are stored (safety)
    for v in outputs:
        if v not in stored:
            ops.append(['store', v])
            stored.add(v)

    # Truncate if too many operations
    if len(ops) > 20000:
        # This shouldn't happen, but truncate as a safety
        ops = ops[:20000]

    return ops

# EVOLVE-BLOCK-END
