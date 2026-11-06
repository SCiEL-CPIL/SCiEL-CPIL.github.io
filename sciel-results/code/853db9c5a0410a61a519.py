"""Communication-minimizing DAG pebbling with improved eviction policy."""
# EVOLVE-BLOCK-START
import time
from collections import deque, defaultdict


def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

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
    except Exception:
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

    # Precompute next uses for each vertex
    next_uses = {v: [] for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents_map.get(v, []):
            next_uses[p].append(i)

    # Track current use index for each vertex
    use_index = {v: 0 for v in vertices}

    def get_next_use_time(v):
        idx = use_index[v]
        uses = next_uses[v]
        if idx < len(uses):
            return uses[idx]
        return float('inf')

    def advance_use(v):
        use_index[v] += 1

    # Fast memory state
    fast_mem = set()  # set of vertices in fast memory
    operations = []
    op_count = 0
    MAX_OPS = 20000

    # Precompute total uses for each vertex
    total_uses = {v: len(next_uses[v]) for v in vertices}

    def remaining_uses(v):
        """Number of remaining uses of vertex v."""
        return total_uses[v] - use_index[v]

    def evict_optimal(exclude=None):
        """Evict the vertex with the least remaining uses, or one with no more uses.
        
        Args:
            exclude: Optional set of vertices to avoid evicting (e.g., parents about to be loaded)
        """
        nonlocal op_count
        if not fast_mem:
            return None
        
        candidates = [v for v in fast_mem if v not in (exclude or set())]
        if not candidates:
            candidates = list(fast_mem)
        
        # First priority: evict vertices with no more uses
        no_use_v = None
        for v in candidates:
            if remaining_uses(v) == 0:
                no_use_v = v
                break
        if no_use_v is not None:
            fast_mem.discard(no_use_v)
            operations.append(['store', no_use_v])
            op_count += 1
            return no_use_v
        
        # Second priority: evict vertex with fewest remaining uses
        # Tie-break: prefer evicting vertices whose next use is farther in the future
        best_v = None
        best_uses = float('inf')
        best_next_time = float('-inf')
        for v in candidates:
            rem = remaining_uses(v)
            if rem < best_uses or (rem == best_uses and get_next_use_time(v) > best_next_time):
                best_uses = rem
                best_next_time = get_next_use_time(v)
                best_v = v
        if best_v is not None:
            fast_mem.discard(best_v)
            operations.append(['store', best_v])
            op_count += 1
        return best_v

    def ensure_space(count=1, exclude=None):
        """Ensure fast memory has at least `count` free slots.
        
        Args:
            count: Number of free slots needed
            exclude: Optional set of vertices to avoid evicting
        """
        while len(fast_mem) >= fast_memory_budget - (count - 1):
            evict_optimal(exclude)

    def load_vertex(v, exclude=None):
        """Load vertex v into fast memory if not already there.
        
        Args:
            exclude: Optional set of vertices to avoid evicting during space management
        """
        nonlocal op_count
        if v in fast_mem:
            return
        ensure_space(1, exclude)
        fast_mem.add(v)
        operations.append(['load', v])
        op_count += 1

    # Process vertices in topological order
    for v in topo_order:
        if time.time() > deadline or op_count >= MAX_OPS:
            break

        if v in input_set:
            continue

        # Non-input vertex: compute it
        parents_v = parents_map.get(v, [])
        
        # Pre-identify which parents are already in fast memory
        parents_in_mem = [p for p in parents_v if p in fast_mem]
        parents_to_load = [p for p in parents_v if p not in fast_mem]
        
        # Load parents that need to be loaded, being careful about eviction
        # We want to keep parents in memory if they'll be used again soon
        for p in parents_to_load:
            # Try to load this parent; exclude other parents that we still need
            remaining_parents = set(parents_to_load)
            remaining_parents.discard(p)
            load_vertex(p, exclude=remaining_parents)
            advance_use(p)
        
        # Also advance use for parents already in memory
        for p in parents_in_mem:
            advance_use(p)

        # Compute v
        operations.append(['compute', v])
        op_count += 1

        # Add v to fast memory, evicting if necessary
        if len(fast_mem) >= fast_memory_budget:
            evict_optimal(exclude={v})
        fast_mem.add(v)
        
        # If still over budget (shouldn't happen, but safety check)
        if len(fast_mem) > fast_memory_budget:
            fast_mem.discard(v)
            ensure_space(1)
            fast_mem.add(v)

    # At the end, ensure all outputs are blue (stored to disk)
    for v in outputs:
        if time.time() > deadline or op_count >= MAX_OPS:
            break

        if v in fast_mem:
            operations.append(['store', v])
            op_count += 1
            fast_mem.discard(v)
        else:
            # Load and store the output
            if len(fast_mem) >= fast_memory_budget:
                evict_optimal(exclude={v})
            operations.append(['load', v])
            op_count += 1
            fast_mem.add(v)
            operations.append(['store', v])
            op_count += 1
            fast_mem.discard(v)

    return operations[:MAX_OPS]

# EVOLVE-BLOCK-END
