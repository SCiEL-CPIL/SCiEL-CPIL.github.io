"""DAG pebbling with divide-and-conquer to minimize communication."""
import time
import sys
from collections import defaultdict, deque

_DEADLINE = 50.0  # Safety margin before 60s limit


def _get_parents(dag, vertex):
    """Get parent vertices of a given vertex."""
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        if isinstance(parents, dict):
            return parents.get(vertex, [])
        else:
            return list(parents.get(vertex, []))
    elif hasattr(dag, 'parents'):
        p = dag.parents
        if isinstance(p, dict):
            return p.get(vertex, [])
        else:
            return list(p[vertex])
    else:
        return []


def _get_outputs(dag):
    """Get output vertices of the DAG."""
    if isinstance(dag, dict):
        outputs = dag.get('outputs', [])
        return list(outputs)
    elif hasattr(dag, 'outputs'):
        return list(dag.outputs)
    else:
        return []


def _get_all_vertices(dag):
    """Get all vertices in the DAG."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        if vertices is not None:
            return list(vertices)
        parents = dag.get('parents', {})
        all_v = set()
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        return list(all_v)
    elif hasattr(dag, 'vertices'):
        return list(dag.vertices)
    else:
        return []


def _get_inputs(dag):
    """Get input vertices (those with no parents)."""
    all_v = _get_all_vertices(dag)
    inputs = []
    for v in all_v:
        if not _get_parents(dag, v):
            inputs.append(v)
    return inputs


def _topological_order(dag):
    """Get topological order of vertices."""
    if isinstance(dag, dict) and 'topo_order' in dag:
        return list(dag['topo_order'])

    all_v = _get_all_vertices(dag)
    if not all_v:
        return []

    parents = {}
    children = defaultdict(list)

    for v in all_v:
        ps = _get_parents(dag, v)
        parents[v] = ps
        for p in ps:
            children[p].append(v)

    # Kahn's algorithm
    in_degree = {v: len(parents[v]) for v in all_v}
    queue = deque([v for v in all_v if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if len(topo) != len(all_v):
        return all_v

    return topo


def _validate_operations(operations, dag, fast_memory_budget):
    """Validate that operations are legal."""
    in_fast = set()
    for op, v in operations:
        if op == 'load':
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        elif op == 'compute':
            parents = _get_parents(dag, v)
            for p in parents:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        else:
            return False
    return True


def _count_transfers(operations):
    """Count load and store operations."""
    return sum(1 for op, v in operations if op in ('load', 'store'))


def _naive_schedule_transfers(dag, fast_memory_budget):
    """Estimate naive schedule transfers."""
    all_v = _get_all_vertices(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    # Naive: load each input once, store each output once
    transfers = len(inputs) + len(outputs)
    return transfers


def _pebble_subdag(dag, subdag_vertices, fast_memory_budget, deadline):
    """
    Pebble a sub-DAG consisting of vertices in subdag_vertices.
    Uses a simple topological order strategy with memory management.
    Returns list of operations.
    """
    start_time = time.time()

    subdag_set = set(subdag_vertices)
    topo = _topological_order(dag)
    # Filter to subdag vertices in topo order
    subdag_topo = [v for v in topo if v in subdag_set]

    if not subdag_topo:
        return []

    # Determine which vertices are "internal" (have parents in subdag) vs "inputs" (no parents in subdag)
    subdag_inputs = []
    subdag_computes = []
    for v in subdag_topo:
        parents = _get_parents(dag, v)
        parents_in_subdag = [p for p in parents if p in subdag_set]
        if not parents_in_subdag:
            subdag_inputs.append(v)
        else:
            subdag_computes.append(v)

    # Count uses within subdag
    use_count = defaultdict(int)
    for v in subdag_topo:
        for p in _get_parents(dag, v):
            if p in subdag_set:
                use_count[p] += 1

    # Strategy: process in topological order
    # Keep vertices in fast memory if they will be used again
    operations = []
    in_fast = set()

    for v in subdag_topo:
        if time.time() - deadline[0] > _DEADLINE:
            break

        parents = _get_parents(dag, v)
        parents_in_subdag = [p for p in parents if p in subdag_set]

        if not parents_in_subdag:
            # This is an input to the subdag - it should already be in fast memory
            # (loaded by caller or it's a global input)
            if v not in in_fast:
                # Should not happen if caller ensures inputs are loaded
                # But handle gracefully
                if len(in_fast) < fast_memory_budget:
                    operations.append(['load', v])
                    in_fast.add(v)
        else:
            # Compute v - ensure all parents are in fast memory
            for p in parents_in_subdag:
                if p not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        # Need to make space - delete something
                        # Find a vertex in in_fast that won't be needed again
                        to_delete = None
                        for candidate in in_fast:
                            if candidate != v and use_count.get(candidate, 0) <= 1:
                                to_delete = candidate
                                break
                        if to_delete is None:
                            # Delete any non-parent
                            for candidate in in_fast:
                                if candidate not in parents_in_subdag:
                                    to_delete = candidate
                                    break
                        if to_delete is not None:
                            operations.append(['delete', to_delete])
                            in_fast.remove(to_delete)
                    operations.append(['load', p])
                    in_fast.add(p)

            operations.append(['compute', v])
            in_fast.add(v)

            # If v won't be used again in this subdag, delete it immediately
            if use_count.get(v, 0) <= 1 and v not in subdag_set:
                pass
            elif use_count.get(v, 0) <= 1:
                # Check if v is used by any remaining vertex in subdag_topo
                remaining = subdag_topo[subdag_topo.index(v) + 1:] if v in subdag_topo else []
                used_by_remaining = False
                for r in remaining:
                    if v in _get_parents(dag, r):
                        used_by_remaining = True
                        break
                if not used_by_remaining:
                    operations.append(['delete', v])
                    in_fast.remove(v)

    return operations


def _pebble_lru(dag, fast_memory_budget, deadline):
    """
    Simple LRU-based pebbling strategy.
    Process in topological order, load parents as needed, compute, 
    and use LRU eviction when memory is full.
    """
    start_time = time.time()

    all_v = _get_all_vertices(dag)
    if not all_v:
        return []

    topo = _topological_order(dag)
    outputs = _get_outputs(dag)
    output_set = set(outputs)

    # Count how many times each vertex is used (by children)
    use_count = defaultdict(int)
    for v in all_v:
        for p in _get_parents(dag, v):
            use_count[p] += 1
    # Outputs are also "used" (need to be stored)
    for o in outputs:
        use_count[o] += 1

    operations = []
    in_fast = []  # List for LRU ordering
    in_fast_set = set()

    def load_vertex(v):
        """Load vertex into fast memory, evicting LRU if needed."""
        if v in in_fast_set:
            return
        while len(in_fast) >= fast_memory_budget:
            # Evict LRU (first in list)
            evict = in_fast.pop(0)
            in_fast_set.remove(evict)
            operations.append(['delete', evict])
        operations.append(['load', v])
        in_fast.append(v)
        in_fast_set.add(v)

    def touch_vertex(v):
        """Move vertex to most recently used position."""
        if v in in_fast_set:
            in_fast.remove(v)
            in_fast.append(v)

    for v in topo:
        if time.time() - deadline[0] > _DEADLINE:
            break

        parents = _get_parents(dag, v)
        if not parents:
            # Input vertex - don't load unless needed
            continue

        # Load all parents
        for p in parents:
            if p not in in_fast_set:
                load_vertex(p)
            else:
                touch_vertex(p)

        # Compute v
        operations.append(['compute', v])
        in_fast.append(v)
        in_fast_set.add(v)

        # If v is an output, we'll store it later
        # If v won't be used again, delete it
        remaining_uses = use_count.get(v, 0) - 1  # Subtract current use
        if remaining_uses <= 0 and v not in output_set:
            # Delete immediately
            if v in in_fast_set:
                in_fast.remove(v)
                in_fast_set.remove(v)
                operations.append(['delete', v])
        elif remaining_uses <= 0 and v in output_set:
            # It's an output, keep it for now, store later
            pass
        else:
            # Will be used again, keep in memory
            pass

    # Store all outputs
    for o in outputs:
        if o in in_fast_set:
            operations.append(['store', o])
            in_fast.remove(o)
            in_fast_set.remove(o)
        elif o not in in_fast_set:
            # Need to load it
            load_vertex(o)
            operations.append(['store', o])
            in_fast.remove(o)
            in_fast_set.remove(o)

    return operations


def _pebble_hierarchical(dag, fast_memory_budget, deadline):
    """
    Hierarchical pebbling: divide DAG into layers, process each layer.
    For layered DAGs, this can be more efficient.
    """
    start_time = time.time()

    all_v = _get_all_vertices(dag)
    if not all_v:
        return []

    topo = _topological_order(dag)
    outputs = _get_outputs(dag)
    output_set = set(outputs)
    inputs = _get_inputs(dag)
    input_set = set(inputs)

    # Determine depth of each vertex
    depth = {}
    for v in topo:
        parents = _get_parents(dag, v)
        if not parents:
            depth[v] = 0
        else:
            depth[v] = max(depth[p] for p in parents) + 1

    max_depth = max(depth.values()) if depth else 0

    # Group vertices by depth
    layers = defaultdict(list)
    for v in topo:
        layers[depth[v]].append(v)

    operations = []
    in_fast = []
    in_fast_set = set()

    def load_vertex(v):
        if v in in_fast_set:
            return
        while len(in_fast) >= fast_memory_budget:
            evict = in_fast.pop(0)
            in_fast_set.remove(evict)
            operations.append(['delete', evict])
        operations.append(['load', v])
        in_fast.append(v)
        in_fast_set.add(v)

    def touch_vertex(v):
        if v in in_fast_set:
            in_fast.remove(v)
            in_fast.append(v)

    # Process layer by layer
    for d in range(max_depth + 1):
        if time.time() - deadline[0] > _DEADLINE:
            break

        layer_vertices = layers[d]
        if not layer_vertices:
            continue

        # First, ensure all parents of this layer's vertices are in fast memory
        for v in layer_vertices:
            parents = _get_parents(dag, v)
            for p in parents:
                if p not in in_fast_set:
                    load_vertex(p)
                else:
                    touch_vertex(p)

        # Compute all vertices in this layer
        for v in layer_vertices:
            if time.time() - deadline[0] > _DEADLINE:
                break
            operations.append(['compute', v])
            in_fast.append(v)
            in_fast_set.add(v)

        # After computing layer, delete vertices that won't be needed again
        # A vertex at depth d is needed if it has children at depth d+1 or later
        children_map = defaultdict(list)
        for v in all_v:
            for p in _get_parents(dag, v):
                children_map[p].append(v)

        for v in layer_vertices:
            # Check if v has children in future layers
            has_future_children = False
            for c in children_map.get(v, []):
                if depth.get(c, 0) > d:
                    has_future_children = True
                    break
            if not has_future_children and v not in output_set:
                if v in in_fast_set:
                    in_fast.remove(v)
                    in_fast_set.remove(v)
                    operations.append(['delete', v])

    # Store all outputs
    for o in outputs:
        if o in in_fast_set:
            operations.append(['store', o])
            in_fast.remove(o)
            in_fast_set.remove(o)
        else:
            load_vertex(o)
            operations.append(['store', o])
            in_fast.remove(o)
            in_fast_set.remove(o)

    return operations


def _best_schedule(dag, fast_memory_budget, deadline):
    """Try multiple strategies and return the best one."""
    start_time = time.time()

    best_ops = None
    best_transfers = float('inf')

    # Strategy 1: LRU
    try:
        ops1 = _pebble_lru(dag, fast_memory_budget, deadline)
        if ops1 and _validate_operations(ops1, dag, fast_memory_budget) and len(ops1) <= 20000:
            t1 = _count_transfers(ops1)
            if t1 < best_transfers:
                best_transfers = t1
                best_ops = ops1
    except Exception:
        pass

    # Strategy 2: Hierarchical
    try:
        ops2 = _pebble_hierarchical(dag, fast_memory_budget, deadline)
        if ops2 and _validate_operations(ops2, dag, fast_memory_budget) and len(ops2) <= 20000:
            t2 = _count_transfers(ops2)
            if t2 < best_transfers:
                best_transfers = t2
                best_ops = ops2
    except Exception:
        pass

    if best_ops is not None:
        return best_ops

    # Fallback: simple topological order
    topo = _topological_order(dag)
    outputs = _get_outputs(dag)
    ops = []
    in_fast = set()

    for v in topo:
        if time.time() - deadline[0] > _DEADLINE:
            break
        parents = _get_parents(dag, v)
        if not parents:
            continue
        for p in parents:
            if p not in in_fast:
                ops.append(['load', p])
                in_fast.add(p)
        ops.append(['compute', v])
        in_fast.add(v)

    for o in outputs:
        if o not in in_fast:
            ops.append(['load', o])
            in_fast.add(o)
        ops.append(['store', o])
        in_fast.remove(o)

    return ops


def pebble(dag, fast_memory_budget):
    deadline = [time.time()]

    try:
        ops = _best_schedule(dag, fast_memory_budget, deadline)

        if ops and _validate_operations(ops, dag, fast_memory_budget) and len(ops) <= 20000:
            return ops

        # Final fallback
        topo = _topological_order(dag)
        outputs = _get_outputs(dag)
        ops = []
        in_fast = set()

        for v in topo:
            parents = _get_parents(dag, v)
            if not parents:
                continue
            for p in parents:
                if p not in in_fast:
                    ops.append(['load', p])
                    in_fast.add(p)
            ops.append(['compute', v])
            in_fast.add(v)

        for o in outputs:
            if o not in in_fast:
                ops.append(['load', o])
                in_fast.add(o)
            ops.append(['store', o])
            in_fast.remove(o)

        return ops

    except Exception:
        # Absolute fallback
        topo = _topological_order(dag)
        outputs = _get_outputs(dag)
        ops = []
        in_fast = set()

        for v in topo:
            parents = _get_parents(dag, v)
            if not parents:
                continue
            for p in parents:
                if p not in in_fast:
                    ops.append(['load', p])
                    in_fast.add(p)
            ops.append(['compute', v])
            in_fast.add(v)

        for o in outputs:
            if o not in in_fast:
                ops.append(['load', o])
                in_fast.add(o)
            ops.append(['store', o])
            in_fast.remove(o)

        return ops
