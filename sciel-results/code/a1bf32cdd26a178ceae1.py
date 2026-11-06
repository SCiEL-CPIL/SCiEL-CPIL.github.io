"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Parse DAG structure
    def get_vertices(d):
        if hasattr(d, 'vertices'):
            return list(d.vertices)
        elif isinstance(d, dict) and 'vertices' in d:
            return list(d['vertices'])
        elif hasattr(d, 'nodes'):
            return list(d.nodes)
        elif isinstance(d, dict) and 'nodes' in d:
            return list(d['nodes'])
        else:
            if hasattr(d, 'parents'):
                parents = d.parents
            elif isinstance(d, dict) and 'parents' in d:
                parents = d['parents']
            else:
                raise ValueError("Cannot determine vertices from dag")
            verts = set()
            for v, ps in parents.items():
                verts.add(v)
                verts.update(ps)
            return list(verts)

    def get_parents(d):
        if hasattr(d, 'parents'):
            return dict(d.parents)
        elif isinstance(d, dict) and 'parents' in d:
            return dict(d['parents'])
        elif hasattr(d, 'edges'):
            parents = {}
            for e in d.edges:
                if len(e) == 2:
                    child, parent = e
                    parents.setdefault(child, []).append(parent)
            return parents
        else:
            raise ValueError("Cannot determine parents from dag")

    def get_inputs(d):
        if hasattr(d, 'inputs'):
            return list(d.inputs)
        elif isinstance(d, dict) and 'inputs' in d:
            return list(d['inputs'])
        else:
            parents = get_parents(d)
            return [v for v in get_vertices(d) if not parents.get(v, [])]

    def get_outputs(d):
        if hasattr(d, 'outputs'):
            return list(d.outputs)
        elif isinstance(d, dict) and 'outputs' in d:
            return list(d['outputs'])
        else:
            parents = get_parents(d)
            vertices = get_vertices(d)
            children = {}
            for v, ps in parents.items():
                for p in ps:
                    children.setdefault(p, []).append(v)
            return [v for v in vertices if not children.get(v, [])]

    try:
        vertices = get_vertices(dag)
        parents_map = get_parents(dag)
        inputs = get_inputs(dag)
        outputs = get_outputs(dag)
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # Build children map
    children_map = defaultdict(list)
    for v, ps in parents_map.items():
        for p in ps:
            children_map[p].append(v)

    # Topological sort
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if len(topo_order) != len(vertices):
        topo_order = list(vertices)

    input_set = set(inputs)
    output_set = set(outputs)

    MAX_OPS = 20000

    # Compute next_uses for lookahead
    # For each vertex v, next_uses[v] is a sorted list of topo indices where v is used as a parent
    next_uses = {v: [] for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents_map.get(v, []):
            next_uses[p].append(i)

    use_index = {v: 0 for v in vertices}

    def get_next_use_time(v):
        idx = use_index[v]
        uses = next_uses[v]
        if idx < len(uses):
            return uses[idx]
        return float('inf')

    def advance_use(v):
        use_index[v] += 1

    # Compute layer for each vertex (longest path from any input)
    layer = {v: 0 for v in vertices}
    for v in topo_order:
        ps = parents_map.get(v, [])
        if ps:
            layer[v] = max(layer[p] for p in ps) + 1

    # Group vertices by layer
    layers = defaultdict(list)
    for v in vertices:
        layers[layer[v]].append(v)

    # Count how many times each vertex is used as a parent
    use_count = {v: len(next_uses[v]) for v in vertices}

    # Determine if we can use a simpler strategy for large budget
    n = len(vertices)
    if fast_memory_budget >= n:
        # Simple topological pass with aggressive cleanup
        fast_mem = set()
        operations = []
        op_count = 0

        # Load all inputs first
        for v in topo_order:
            if v in input_set:
                fast_mem.add(v)
                operations.append(['load', v])
                op_count += 1

        # Process in topological order
        for v in topo_order:
            if time.time() > deadline or op_count >= MAX_OPS:
                break

            if v in input_set:
                continue

            # Ensure all parents are in fast memory (they should be, but check)
            for p in parents_map.get(v, []):
                if p not in fast_mem:
                    fast_mem.add(p)
                    operations.append(['load', p])
                    op_count += 1

            # Compute v
            operations.append(['compute', v])
            op_count += 1
            fast_mem.add(v)

            # Advance use indices for parents
            for p in parents_map.get(v, []):
                advance_use(p)

            # Delete parents that are no longer needed
            for p in parents_map.get(v, []):
                if get_next_use_time(p) == float('inf') and p not in output_set:
                    if p in fast_mem:
                        fast_mem.discard(p)
                        operations.append(['store', p])
                        op_count += 1

        # Store outputs
        for v in outputs:
            if time.time() > deadline or op_count >= MAX_OPS:
                break
            if v in fast_mem:
                operations.append(['store', v])
                op_count += 1
                fast_mem.discard(v)

        if op_count > MAX_OPS:
            operations = operations[:MAX_OPS]
        return operations

    # Memory-constrained case: use layer-based approach with lookahead eviction
    fast_mem = set()
    operations = []
    op_count = 0
    transfer_count = 0

    def evict_one(exclude=None):
        """Evict one vertex from fast memory. Returns the evicted vertex."""
        nonlocal op_count, transfer_count
        candidates = [v for v in fast_mem if v != exclude]
        if not candidates:
            return None
        # Evict the vertex with the farthest next use (or no future use)
        # If tied, evict the one with lower use_count (less frequent)
        worst_v = None
        worst_time = -1
        worst_use_count = float('inf')
        for v in candidates:
            t = get_next_use_time(v)
            # If t is inf, it's a strong candidate for eviction
            # Among finite times, evict the one with the largest time
            # If times are equal, evict the one with lower use count
            if t > worst_time or (t == worst_time and use_count[v] < worst_use_count):
                worst_time = t
                worst_use_count = use_count[v]
                worst_v = v

        if worst_v is not None:
            fast_mem.discard(worst_v)
            operations.append(['store', worst_v])
            op_count += 1
            transfer_count += 1
        return worst_v

    def ensure_in_memory(v):
        """Ensure vertex v is in fast memory, evicting if necessary."""
        nonlocal op_count, transfer_count
        if v in fast_mem:
            return
        if len(fast_mem) < fast_memory_budget:
            fast_mem.add(v)
            operations.append(['load', v])
            op_count += 1
            transfer_count += 1
        else:
            evict_one(exclude=v)
            if len(fast_mem) < fast_memory_budget:
                fast_mem.add(v)
                operations.append(['load', v])
                op_count += 1
                transfer_count += 1
            else:
                # Should not happen if evict_one works, but just in case
                # Force evict something
                for cand in list(fast_mem):
                    if cand != v:
                        fast_mem.discard(cand)
                        operations.append(['store', cand])
                        op_count += 1
                        transfer_count += 1
                        break
                fast_mem.add(v)
                operations.append(['load', v])
                op_count += 1
                transfer_count += 1

    # Process each vertex in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        if op_count >= MAX_OPS:
            break

        if v in input_set:
            # Input vertex: we don't need to load it now; it will be loaded when needed as a parent
            continue

        # Non-input vertex: compute it
        # First, ensure all parents are in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                ensure_in_memory(p)
            advance_use(p)

        # Before compute, if fast_mem is full, we need space for v's result
        # The compute operation will place v into fast memory
        # So we need to evict if full
        if len(fast_mem) >= fast_memory_budget:
            evict_one(exclude=None)

        # Compute v - this places v into fast memory
        operations.append(['compute', v])
        op_count += 1
        fast_mem.add(v)

        # After computing v, check if any parents are no longer needed
        # This is a proactive cleanup to keep memory free
        for p in parents_map.get(v, []):
            if get_next_use_time(p) == float('inf') and p not in output_set:
                if p in fast_mem:
                    fast_mem.discard(p)
                    operations.append(['store', p])
                    op_count += 1
                    transfer_count += 1

    # At the end, ensure all outputs are blue (stored to disk)
    for v in outputs:
        if time.time() > deadline:
            break
        if op_count >= MAX_OPS:
            break

        if v in fast_mem:
            operations.append(['store', v])
            op_count += 1
            transfer_count += 1
            fast_mem.discard(v)
        else:
            # Output not in fast memory - this is a problem
            # We need to load it (but it's not on disk either, it was never stored?)
            # Actually, if it was computed, it must have been stored if evicted.
            # So we load it back? No, we just need it to be blue.
            # The problem says "all outputs must finish blue".
            # If it's not in fast memory, it must be on disk (blue).
            # So we don't need to do anything?
            # Wait, "blue" means on disk. "Red" means in fast memory.
            # So if it's not in fast_mem, it's on disk, so it's blue.
            # So we don't need to do anything.
            pass

    # Validate: check we don't exceed max ops
    if op_count > MAX_OPS:
        operations = operations[:MAX_OPS]

    return operations

# EVOLVE-BLOCK-END
