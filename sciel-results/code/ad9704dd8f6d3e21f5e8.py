import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Parse DAG structure
    if isinstance(dag, dict):
        parents = {k: list(v) for k, v in dag.get('parents', {}).items()}
        outputs = set(dag.get('outputs', []))
        inputs = set(dag.get('inputs', []))
        vertices = set()
        for v, ps in parents.items():
            vertices.add(v)
            vertices.update(ps)
        vertices.update(outputs)
        vertices.update(inputs)
    else:
        parents = {k: list(v) for k, v in getattr(dag, 'parents', {}).items()}
        outputs = set(getattr(dag, 'outputs', []))
        inputs = set(getattr(dag, 'inputs', []))
        vertices = set()
        for v, ps in parents.items():
            vertices.add(v)
            vertices.update(ps)
        vertices.update(outputs)
        vertices.update(inputs)

    # Build children adjacency
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Topological sort
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if len(topo_order) != len(vertices):
        remaining = list(vertices - set(topo_order))
        topo_order.extend(remaining)

    # State
    in_fast = set(inputs)
    in_slow = set()
    computed = set(inputs)
    operations = []

    # Remaining children to process for each vertex
    remaining_children = {v: set(cs) for v, cs in children.items()}

    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break

        if v in inputs:
            continue

        ps = parents.get(v, [])

        # Ensure all parents are in fast memory
        for p in ps:
            if p not in in_fast:
                operations.append(['load', p])
                in_fast.add(p)

        # Compute v
        operations.append(['compute', v])
        in_fast.add(v)
        computed.add(v)

        # Update remaining children count for parents
        for p in ps:
            if v in remaining_children[p]:
                remaining_children[p].discard(v)

        # Manage fast memory: evict if over budget
        if len(in_fast) > fast_memory_budget:
            # Values that must stay: just-computed v, parents with remaining children, computed outputs
            must_keep = {v}
            for p in ps:
                if remaining_children[p]:
                    must_keep.add(p)
            must_keep.update(outputs & computed)

            # Evict non-must-keep first, prioritizing those with no remaining children
            evictable = [x for x in in_fast if x not in must_keep]
            evictable.sort(key=lambda x: len(remaining_children[x]))

            num_to_evict = len(in_fast) - fast_memory_budget
            for i in range(min(num_to_evict, len(evictable))):
                val = evictable[i]
                if val in in_fast:
                    in_fast.discard(val)
                    if val not in inputs and val in computed:
                        in_slow.add(val)
                        operations.append(['store', val])

            # If still over budget, evict from must-keep (least useful first)
            if len(in_fast) > fast_memory_budget:
                all_vals = list(in_fast)
                all_vals.sort(key=lambda x: len(remaining_children[x]))
                num_to_evict = len(in_fast) - fast_memory_budget
                for i in range(min(num_to_evict, len(all_vals))):
                    val = all_vals[i]
                    if val in in_fast:
                        in_fast.discard(val)
                        if val not in inputs and val in computed:
                            in_slow.add(val)
                            operations.append(['store', val])

    # Ensure all outputs are in fast memory (blue)
    for out in outputs:
        if out not in in_fast:
            operations.append(['load', out])
            in_fast.add(out)

    # Truncate to max operations
    if len(operations) > 20000:
        operations = operations[:20000]

    return operations
