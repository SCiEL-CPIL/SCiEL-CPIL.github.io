"""Communication-minimizing DAG pebbling with stronger algorithm."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag structure expected:
      - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
      - dag['outputs']: list of output vertices
      - dag['inputs']: list of input vertices (start blue)
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}.
    """
    start_time = time.time()
    DEADLINE = 80.0

    try:
        parents = dag.get('parents', {})
        outputs = list(dag.get('outputs', []))
        inputs = list(dag.get('inputs', []))

        # Build children map
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)

        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)

        # Topological sort (Kahn's algorithm)
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0

        queue = sorted([v for v in all_vertices if in_degree.get(v, 0) == 0])
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
                    queue.sort()

        topo_index = {v: i for i, v in enumerate(topo_order)}

        # Precompute depth (longest path from any input)
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1

        # Precompute last_use: the maximum topo_index of any child, or own index if leaf
        last_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, -1)

        # Outputs must be blue at the end
        must_be_blue = set(outputs)

        ops = []
        fast_mem = set()
        stored = set()  # vertices that have been stored to slow memory

        def get_eviction_candidate():
            """Return the vertex in fast_mem to evict, or None."""
            best = None
            best_key = None
            for v in fast_mem:
                if v in must_be_blue:
                    # Outputs: only consider if no other option
                    key = (1, depth[v], last_use[v])
                else:
                    # Non-outputs: prefer low depth, early last_use
                    key = (0, depth[v], last_use[v])
                if best_key is None or key < best_key:
                    best_key = key
                    best = v
            return best

        visiting = set()

        def ensure_in_fast_mem(v):
            """
            Ensure vertex v is in fast_mem.
            Returns True on success, False on failure.
            """
            if time.time() > DEADLINE:
                return False

            if v in fast_mem:
                return True

            if v in visiting:
                return False

            ps = parents.get(v, [])

            if not ps:
                # Input vertex
                if len(fast_mem) >= fast_memory_budget:
                    cand = get_eviction_candidate()
                    if cand is None:
                        return False
                    ops.append(['delete', cand])
                    fast_mem.discard(cand)
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                # Computed vertex: ensure all parents first
                visiting.add(v)
                try:
                    for p in ps:
                        if not ensure_in_fast_mem(p):
                            return False

                    # Now compute v
                    if len(fast_mem) >= fast_memory_budget:
                        cand = get_eviction_candidate()
                        if cand is None:
                            return False
                        ops.append(['delete', cand])
                        fast_mem.discard(cand)

                    ops.append(['compute', v])
                    fast_mem.add(v)
                    return True
                finally:
                    visiting.discard(v)

        # Main loop: process in topological order
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break

            # Proactively delete dead vertices (last_use < i, not outputs, not v)
            to_delete = []
            for u in list(fast_mem):
                if u == v:
                    continue
                if u in must_be_blue:
                    continue
                if last_use[u] < i:
                    to_delete.append(u)

            for u in to_delete:
                if u in fast_mem:
                    ops.append(['delete', u])
                    fast_mem.discard(u)

            # Ensure v is in fast_mem
            if not ensure_in_fast_mem(v):
                # Failed; try to free space more aggressively
                for u in list(fast_mem):
                    if u not in must_be_blue and u != v:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                        if len(fast_mem) < fast_memory_budget:
                            break
                if not ensure_in_fast_mem(v):
                    break

        # Ensure all outputs are in fast_mem at the end
        for o in outputs:
            if time.time() > DEADLINE:
                break
            if o not in fast_mem:
                for u in list(fast_mem):
                    if u not in must_be_blue:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                        if len(fast_mem) < fast_memory_budget:
                            break
                ensure_in_fast_mem(o)

        # If we still don't have all outputs, try harder
        for o in outputs:
            if o not in fast_mem:
                for u in list(fast_mem):
                    if u not in must_be_blue:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                ensure_in_fast_mem(o)

        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            return ops[:20000]

    except Exception as e:
        return []
# EVOLVE-BLOCK-END
