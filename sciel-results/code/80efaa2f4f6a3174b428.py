import time
from collections import deque

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using a greedy priority-queue scheduler.
    """
    start_time = time.time()
    deadline = 55.0

    # Parse DAG structure - try to understand the format
    try:
        if isinstance(dag, dict):
            parents = dag.get('parents', [])
            outputs = set(dag.get('outputs', []))
            V = len(parents)
        elif isinstance(dag, (tuple, list)) and len(dag) == 2:
            V, edges = dag
            if isinstance(V, int) and isinstance(edges, (list, tuple)):
                parents = [[] for _ in range(V)]
                for e in edges:
                    if isinstance(e, (tuple, list)) and len(e) == 2:
                        p, c = e
                        parents[c].append(p)
                outputs = set()
                children = [[] for _ in range(V)]
                for v in range(V):
                    for p in parents[v]:
                        children[p].append(v)
                outputs = set(v for v in range(V) if not children[v])
            else:
                raise ValueError("Unknown dag format")
        else:
            parents = list(dag)
            V = len(parents)
            children = [[] for _ in range(V)]
            for v in range(V):
                for p in parents[v]:
                    children[p].append(v)
            outputs = set(v for v in range(V) if not children[v])
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

    # Build children and in-degree
    children = [[] for _ in range(V)]
    for v in range(V):
        for p in parents[v]:
            children[p].append(v)
    in_degree = [len(p) for p in parents]

    # Topological order
    topo_order = []
    q = deque([v for v in range(V) if in_degree[v] == 0])
    rem_deg = in_degree[:]
    while q:
        v = q.popleft()
        topo_order.append(v)
        for c in children[v]:
            rem_deg[c] -= 1
            if rem_deg[c] == 0:
                q.append(c)

    if len(topo_order) != V:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

    # State: 0=blue, 1=red, 2=not_computed
    is_input = [len(parents[v]) == 0 for v in range(V)]
    state = [2] * V
    for v in range(V):
        if is_input[v]:
            state[v] = 0  # blue initially
        else:
            state[v] = 2  # not computed

    fast_mem = set()
    ops = []

    # needed[v] = number of children of v that have not yet been computed
    needed = [len(children[v]) for v in range(V)]

    def evict_least_needed(exclude_set=None):
        """Evict the vertex in fast_mem with the smallest needed count."""
        if exclude_set is None:
            exclude_set = set()
        candidates = [u for u in fast_mem if u not in exclude_set]
        if not candidates:
            return None
        min_needed = float('inf')
        min_v = None
        for u in candidates:
            if needed[u] < min_needed:
                min_needed = needed[u]
                min_v = u
        if min_v is not None:
            ops.append(['store', min_v])
            state[min_v] = 0
            fast_mem.remove(min_v)
            return min_v
        return None

    for v in topo_order:
        if time.time() - start_time > deadline:
            break

        if state[v] == 1:
            continue

        # Ensure parents are red
        for p in parents[v]:
            if state[p] == 0:
                if len(fast_mem) >= fast_memory_budget:
                    parent_set = set(parents[v])
                    evict_least_needed(exclude_set=parent_set)
                ops.append(['load', p])
                state[p] = 1
                fast_mem.add(p)
            elif state[p] == 2:
                # Shouldn't happen in topo order, but handle gracefully
                pass

        # Compute v
        if len(fast_mem) >= fast_memory_budget:
            parent_set = set(parents[v])
            evict_least_needed(exclude_set=parent_set)

        ops.append(['compute', v])
        state[v] = 1
        fast_mem.add(v)

        # Update needed counts
        for p in parents[v]:
            needed[p] -= 1
            if needed[p] == 0 and state[p] == 1:
                ops.append(['store', p])
                state[p] = 0
                fast_mem.remove(p)

    # Ensure outputs are blue
    for v in outputs:
        if state[v] == 1:
            ops.append(['store', v])
            state[v] = 0
            fast_mem.remove(v)

    if len(ops) > 20000:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

    return ops
# EVOLVE-BLOCK-END
