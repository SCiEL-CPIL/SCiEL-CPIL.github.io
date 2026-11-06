"""DAG pebbling optimizer: minimize transfer count (loads+stores)."""
import time
import heapq
from itertools import product

_START_TIME = None

def _now():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.monotonic()
    return time.monotonic()

def _deadline_exceeded(margin=2.0):
    return (_now() - _START_TIME) > (90.0 - margin)

def pebble(dag, fast_memory_budget):
    """
    Returns list of [operation, vertex] with operations load/store/compute/delete.
    
    Strategy: Use recursive pebbling with space-time tradeoff.
    For a layered DAG, we exploit the tree structure of parents.
    """
    # Parse the DAG structure
    # dag has: vertices, parents (topologically ordered), outputs
    vertices = list(dag.get('vertices', range(dag.get('num_vertices', 0))))
    parents = dag.get('parents', [])
    outputs = set(dag.get('outputs', []))
    num_vertices = len(vertices)
    
    # Build parent/children maps
    parent_map = {}  # v -> list of parent indices
    child_map = {v: [] for v in vertices}
    for v in vertices:
        pv = parents[v] if v < len(parents) else []
        parent_map[v] = list(pv)
        for p in parent_map[v]:
            child_map[p].append(v)
    
    # Identify input vertices (no parents)
    input_vertices = [v for v in vertices if len(parent_map[v]) == 0]
    
    # Build topological order (assume already given in vertex order)
    topo_order = list(vertices)
    
    # For each vertex, compute which other vertices depend on it (transitively)
    # and the "pebbling cost" to recompute it
    # Simple approach: recursive evaluation with memoization and space management
    
    B = fast_memory_budget
    
    # Determine which vertices are "heavy" (expensive to recompute)
    # For a layered DAG, vertices deeper have more ancestors
    
    # Strategy: 
    # 1. Load all inputs that are needed, compute bottom-up, store intermediates that are reused
    # 2. Key insight: a vertex needs to be stored if it has multiple children that need it
    #    after it's no longer in fast memory
    
    # Compute "reusability" - number of distinct paths from v to outputs
    # A vertex should be stored if reusability > 1 and recompute cost is high
    
    # Let's use a cleaner recursive approach:
    # recursive_pebble(sub_dag, available_space) -> (ops, ops_count)
    # For a single vertex v with parents P:
    #   If v is input: load(v)
    #   Else: for each parent p in P:
    #     if p is reused later (by sibling computations or ancestor needs):
    #       compute p, if not enough space, store p, recompute later or keep
    #     else: compute p, delete p after use
    #   compute v
    #   if v is reused later or v is output: keep/store v
    #   else: delete v
    
    # Simpler baseline improvement: identify critical path vertices vs branch vertices
    
    # Compute for each vertex:
    # 1. depth (distance from inputs)
    # 2. fan-out (number of children)
    # 3. recompute_cost (total vertices needed to recompute, recursively)
    # 4. Is it on a shared path?
    
    # Build the full computation DAG analysis
    import sys
    sys.setrecursionlimit(100000)
    
    # Memoize recompute cost
    recompute_cost = {}
    def get_recompute_cost(v, _memo=recompute_cost):
        if v in _memo:
            return _memo[v]
        if v in parent_map and len(parent_map[v]) == 0:
            _memo[v] = 1  # just a load
            return 1
        cost = 1  # compute itself
        seen_parents = set()
        for p in parent_map[v]:
            c = get_recompute_cost(p)
            if p not in seen_parents:
                cost += c
                seen_parents.add(p)
            # Note: shared parents among siblings are only computed once
        _memo[v] = cost
        return cost
    
    # For each vertex, compute how many times it's "needed" after first computation
    # i.e., the number of distinct children (or further uses) that require it
    # In a DAG, a vertex v is used by all its children. If a child c1 is computed,
    # then later c2 is computed and also needs v, v must still be available.
    
    # Key question: in a topological evaluation order, which vertices can be deleted?
    # A vertex v can be deleted after all its children have been computed.
    
    # For a single output vertex (or multiple outputs evaluated sequentially):
    # We need a topological order of computation.
    # At each step, we load a parent if not in memory, compute, and optionally delete.
    
    # Optimal strategy for bounded memory:
    # This is essentially the graph pebbling problem, which is PSPACE-complete in general.
    # For layered/tree-structured DAGs, we can use dynamic programming or heuristics.
    
    # Let's try a practical heuristic:
    # 1. Determine an evaluation order (topological)
    # 2. For each vertex in order:
    #    - If it's an input: load it
    #    - If a parent is not in memory: load or recompute it
    #    - Compute the vertex
    #    - If memory is full and we need more space: evict (store) a vertex with highest recompute cost
    #    - If a vertex's children are all done: it can be deleted
    #    - Outputs must remain blue at end
    
    # First, determine which vertices MUST be stored (have multiple children or are outputs)
    must_store = set()
    for v in vertices:
        if len(child_map[v]) > 1:
            must_store.add(v)
        if v in outputs:
            must_store.add(v)
    
    # Also, vertices whose children appear non-consecutively in topological order
    # need to be stored (since we'll need them again after intervening computations)
    
    # Let's try a simpler but effective strategy:
    # Group vertices by "levels" if the DAG is layered
    # Process level by level, storing only what's needed for the next level
    
    # Detect if DAG is layered (all parents of a vertex are in the previous layer)
    # Assign layers by longest path from inputs
    layer = {}
    for v in topo_order:
        if len(parent_map[v]) == 0:
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in parent_map[v]) + 1
    
    max_layer = max(layer.values()) if layer else 0
    
    # Count vertices per layer
    layers = {}
    for v in vertices:
        layers.setdefault(layer[v], []).append(v)
    
    # For layered DAGs, we can process layer by layer:
    # - Keep all vertices from current and previous layers in memory
    # - Store vertices that will be needed by multiple future vertices
    
    # But memory might be too small for that. Let's use a streaming approach:
    
    # Strategy: Use a stack-based recursive evaluation with explicit store/load
    # For each output, recursively evaluate it, keeping only what's needed.
    
    # Let's implement a cleaner recursive pebbling:
    
    ops = []
    
    def emit(op, v):
        ops.append([op, v])
    
    # Track what's in fast memory
    in_memory = set()
    
    # Determine if we can fit all vertices of a level in memory
    # If B >= max layer size, do simple topological pass with no recomputation
    max_layer_size = max((len(vs) for vs in layers.values()), default=0)
    
    if B >= max_layer_size and B >= len(input_vertices):
        # Simple topological evaluation - load inputs, compute, delete when done
        for v in topo_order:
            if len(parent_map[v]) == 0:
                emit('load', v)
                in_memory.add(v)
            else:
                # Parents should be in memory or need to be loaded
                for p in parent_map[v]:
                    if p not in in_memory:
                        emit('load', p)
                        in_memory.add(p)
                emit('compute', v)
                in_memory.add(v)
                # Delete parents that have no more children to compute
                for p in parent_map[v]:
                    remaining = [c for c in child_map[p] if c not in in_memory or layer[c] > layer[p] + 1]
                    # Actually, check if p is needed by any vertex not yet computed
                    p_children_remaining = [c for c in child_map[p] if c != v and True]
                    # In topo order, all children of p come after p
                    # So p can be deleted after its last child in topo_order is computed
                    last_child_idx = max((topo_order.index(c) for c in child_map[p]), default=-1)
                    current_idx = topo_order.index(v)
                    if current_idx >= last_child_idx and p not in outputs:
                        emit('delete', p)
                        in_memory.discard(p)
                # Don't delete v if it's needed later or is an output
                last_child_idx_v = max((topo_order.index(c) for c in child_map[v]), default=topo_order.index(v))
                if last_child_idx_v == topo_order.index(v) and v not in outputs:
                    # v has no children or all children are before (impossible in topo) - means leaf non-output
                    pass  # Can delete but keep it simple
        return ops
    
    # For constrained memory, use recursive pebbling with memoization
    # This is the key algorithm for the general case
    
    # For now, let's implement a practical approach:
    # Recursive evaluation of each output, with shared sub-computations stored
    
    # Clear ops for the constrained approach
    ops = []
    in_memory = set()
    
    # For each output, compute it via recursive pebbling
    # With limited memory, we store intermediate results that are reused
    
    # Count how many times each vertex is used (number of distinct children paths)
    use_count = {v: max(1, len(child_map[v])) for v in vertices}
    for v in outputs:
        use_count[v] = max(use_count.get(v, 0), 2)  # outputs must be kept
    
    # Determine a pebbling strategy:
    # For vertices with high use_count and high recompute_cost, store them
    # For vertices with low use_count, recompute as needed
    
    # Budget allocation: assign memory slots to stored vertices
    # Sort by "value" = use_count * recompute_cost / (depth + 1)
    
    # Get all non-input vertices sorted by value
    candidates = []
    for v in vertices:
        if len(parent_map[v]) == 0:
            continue
        rc = get_recompute_cost(v)
        uc = use_count[v]
        val = uc * rc
        candidates.append((val, v, rc, uc))
    
    candidates.sort(reverse=True)
    
    # Decide which vertices to "keep" (store) vs "recompute"
    # We need B slots total for inputs + computed results
    # Inputs that are reused multiple times also need slots
    
    # Count how many inputs are reused
    input_use = {}
    for v in vertices:
        for p in parent_map[v]:
            if len(parent_map[p]) == 0:  # p is input
                input_use[p] = input_use.get(p, 0) + 1
    
    # Inputs needed in memory: those with use_count > 1
    input_slots = sum(1 for p, c in input_use.items() if c > 1) + len(outputs)
    
    # Remaining budget for computed intermediates
    remaining_budget = B - min(input_slots, B)
    
    # Select top-value vertices to keep
    keep_set = set()
    for val, v, rc, uc in candidates:
        if len(keep_set) >= remaining_budget:
            break
        if v in outputs:
            continue  # outputs are always kept
        keep_set.add(v)
    
    # Also always keep outputs
    for o in outputs:
        keep_set.add(o)
    
    # Now generate the schedule:
    # Process in topological order.
    # For each vertex v:
    #   For each parent p:
    #     if p is an input:
    #       if p not in memory and (p in keep_set or input_use.get(p,1)==1):
    #         load p
    #     else:
    #       if p not in memory:
    #         if p in keep_set:
    #           # p should have been stored; but if evicted, recompute
    #           # For now, load from slow (simulate as recompute via loads)
    #           # Actually in this model, load = fetch from slow memory
    #           emit('load', p)
    #         else:
    #           # Recompute p recursively
    #           recompute_vertex(p, in_memory, keep_set, emit, parent_map, child_map, outputs, layer, topo_order)
    #   compute v
    #   if v in keep_set or v in outputs:
    #     # keep in memory (no store op needed - it's just not deleted)
    #     pass
    #   else:
    #     # Check if v is still needed
    #     # v's children all come after v in topo order
    #     # v can be deleted after its last child
    #     if all children of v computed:
    #       emit('delete', v)
    
    # Hmm, the recursive recomputation is complex. Let me simplify:
    
    # Alternative simpler approach for bounded memory:
    # Process in topological order.
    # When we need a parent that's not in memory:
    #   If it's an input: load it
    #   If it's a computed vertex: 
    #     - If it's in keep_set: load it (from where it was stored)
    #     - Else: we need to recompute it, which means recomputing its parents, etc.
    #       For simplicity, we just re-load the entire subtree (expensive but correct)
    
    # Even simpler: just load everything we need. The "recompute" savings come from
    # not re-loading things we've already loaded and kept in memory.
    
    # Actually, in this model:
    # - load: fetch a vertex value from slow memory (for inputs or previously computed/stored values)
    # - store: save a computed value to slow memory (so it can be loaded later)
    # - compute: compute a vertex from its parents (parents must be in fast memory)
    # - delete: free a fast memory slot
    
    # So the key optimization is:
    # 1. Minimize loads (each input loaded once if kept in memory)
    # 2. Avoid unnecessary stores/loads for reused vertices
    # 3. Delete vertices when they're no longer needed to free space
    
    # Let me redo this with a cleaner model:
    
    ops = []
    in_memory = set()
    computed_and_stored = set()  # vertices that have been computed and are available in slow memory
    
    def emit2(op, v):
        ops.append([op, v])
    
    def ensure_in_memory(v):
        """Ensure vertex v is in fast memory."""
        if v in in_memory:
            return
        if v in computed_and_stored:
            emit2('load', v)
            in_memory.add(v)
        elif len(parent_map[v]) == 0:
            # Input vertex
            emit2('load', v)
            in_memory.add(v)
        else:
            # Need to compute v
            # First ensure all parents are in memory
            for p in parent_map[v]:
                ensure_in_memory(p)
                # Free space if needed
            emit2('compute', v)
            in_memory.add(v)
            # Delete parents that are no longer needed
            cleanup()
            # If v is reusable, mark as stored
            if v in keep_set or v in outputs:
                emit2('store', v)
                computed_and_stored.add(v)
    
    def cleanup():
        """Delete vertices in memory that are no longer needed."""
        if len(in_memory) <= B:
            return
        # Find deletable vertices (not in keep_set, not output, all children computed)
        # In topo order processing, a vertex's children come after it
        # A vertex can be deleted if all its children are in computed_and_stored or in_memory
        for v in sorted(in_memory, key=lambda x: layer[x]):
            if len(in_memory) <= B:
                break
            if v in outputs or v in keep_set:
                continue
            if len(parent_map[v]) == 0:
                # Input - check if used more
                if input_use.get(v, 1) <= 1 and all(
                c in computed_and_stored or c in in_memory for c in child_map[v]
                ):
                    emit2('delete', v)
                    in_memory.discard(v)
            else:
                # All children must be done
                if all(c in computed_and_stored or c in in_memory for c in child_map[v]):
                    emit2('delete', v)
                    in_memory.discard(v)
    
    # Process outputs one by one (or all in topo order)
    for v in topo_order:
        ensure_in_memory(v)
        # If v is an output or in keep_set, ensure it's stored
        if v in outputs or (v in keep_set and v not in computed_and_stored):
            if v not in in_memory:
                pass  # shouldn't happen
            if v not in computed_and_stored:
                emit2('store', v)
                computed_and_stored.add(v)
        # Clean up space
        if len(in_memory) > B:
            cleanup()
        if len(in_memory) > B:
            # Emergency: force delete non-critical vertices
            for v in sorted(in_memory, key=lambda x: layer[x]):
                if len(in_memory) <= B:
                    break
                if v in outputs:
                    continue
                if v in keep_set and layer[v] < max_layer:
                    continue
                emit2('delete', v)
                in_memory.discard(v)
                computed_and_stored.discard(v)
    
    return ops
