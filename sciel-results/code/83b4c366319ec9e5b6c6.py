"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin from 90s limit

    def _extract_dag_info(dag):
        """Extract parents, outputs, n_vertices from dag object."""
        parents = None
        outputs = None

        # Try attribute access
        if hasattr(dag, 'parents'):
            parents = dag.parents
        if hasattr(dag, 'outputs'):
            outputs = dag.outputs

        # Try dict access
        if parents is None and isinstance(dag, dict):
            if 'parents' in dag:
                parents = dag['parents']
            if 'outputs' in dag:
                outputs = dag['outputs']

        # If dag itself looks like a parents structure
        if parents is None:
            if isinstance(dag, dict):
                parents = dag
            elif isinstance(dag, (list, tuple)):
                parents = list(dag)

        if outputs is None:
            outputs = []

        # Determine n_vertices
        if isinstance(parents, dict):
            if parents:
                n_vertices = max(parents.keys()) + 1
            else:
                n_vertices = 0
        elif isinstance(parents, (list, tuple)):
            n_vertices = len(parents)
        else:
            n_vertices = 0

        return parents, outputs, n_vertices

    parents, outputs, n_vertices = _extract_dag_info(dag)

    if n_vertices == 0:
        return []

    # Normalize parents to dict format
    parent_dict = {}
    if isinstance(parents, dict):
        for v, pv in parents.items():
            parent_dict[v] = list(pv) if pv else []
    elif isinstance(parents, (list, tuple)):
        for v, pv in enumerate(parents):
            parent_dict[v] = list(pv) if pv else []

    # Fill in missing vertices
    for v in range(n_vertices):
        if v not in parent_dict:
            parent_dict[v] = []

    # Build children and in-degrees
    children = {v: [] for v in range(n_vertices)}
    in_degree = [0] * n_vertices
    for v in range(n_vertices):
        for p in parent_dict[v]:
            children[p].append(v)
            in_degree[v] += 1

    # Topological sort with layer assignment
    layer = [-1] * n_vertices
    queue = deque()
    for v in range(n_vertices):
        if in_degree[v] == 0:
            queue.append(v)
            layer[v] = 0

    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if layer[c] == -1 or layer[c] < layer[v] + 1:
                layer[c] = layer[v] + 1
            if in_degree[c] == 0:
                queue.append(c)

    # Handle disconnected components
    max_layer = max((l for l in layer if l >= 0), default=0)
    for v in range(n_vertices):
        if layer[v] == -1:
            layer[v] = max_layer + 1

    num_layers = max(layer) + 1
    layers = [[] for _ in range(num_layers)]
    for v in topo_order:
        layers[layer[v]].append(v)

    output_set = set(outputs) if outputs else set()

    # Precompute needed_by_future
    needed_by_future = [set() for _ in range(num_layers)]
    for li in range(num_layers):
        for fi in range(li + 1, num_layers):
            for fv in layers[fi]:
                for p in parent_dict[fv]:
                    needed_by_future[layer[p]].add(p)

    # Pebbling
    fast_mem = set()
    ops = []

    def load(v):
        ops.append(['load', v])
        fast_mem.add(v)

    def store(v):
        ops.append(['store', v])
        fast_mem.discard(v)

    def compute(v):
        ops.append(['compute', v])
        fast_mem.add(v)

    def delete(v):
        ops.append(['delete', v])
        fast_mem.discard(v)

    def evict_if_needed():
        if len(fast_mem) <= fast_memory_budget:
            return
        excess = len(fast_mem) - fast_memory_budget
        candidates = []
        for v in fast_mem:
            v_layer = layer[v]
            if v not in needed_by_future[v_layer] and v not in output_set:
                candidates.append((0, v))
            elif v in output_set:
                candidates.append((1, v))
            else:
                candidates.append((2, v))
        candidates.sort()
        evicted = 0
        for _, v in candidates:
            if evicted >= excess:
                break
            if v in fast_mem:
                delete(v)
                evicted += 1

    # Process layers
    for li in range(num_layers):
        if time.time() > deadline:
            break
        layer_vertices = layers[li]
        if not layer_vertices:
            continue

        # Load needed parents
        needed_parents = set()
        for v in layer_vertices:
            needed_parents.update(parent_dict[v])

        for p in sorted(needed_parents):
            if p not in fast_mem:
                load(p)

        # Compute vertices
        for v in layer_vertices:
            if v not in fast_mem:
                compute(v)

        evict_if_needed()

    # Store outputs
    if time.time() <= deadline:
        for v in sorted(output_set):
            if v in fast_mem:
                store(v)

    # Cleanup
    if time.time() <= deadline:
        for v in sorted(fast_mem):
            if v not in output_set:
                delete(v)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
# EVOLVE-BLOCK-END
