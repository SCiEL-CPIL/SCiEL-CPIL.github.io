"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
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
    DEADLINE = 85.0  # wall-clock limit with safety margin

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        output_set = set(outputs)
        input_set = set(inputs)

        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)

        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)

        # Topological order
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0

        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            queue.sort()
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)

        topo_index = {v: i for i, v in enumerate(topo_order)}

        # Precompute last_use[v]: the highest topo_index of any vertex u such that v is a parent of u.
        # If v is an output, it must be kept until the end.
        last_use = {}
        for v in all_vertices:
            max_idx = -1
            for c in children.get(v, []):
                idx = topo_index.get(c, -1)
                if idx > max_idx:
                    max_idx = idx
            last_use[v] = max_idx

        # Strategy:
        # 1. Process vertices in topological order.
        # 2. For each vertex v:
        #    - If v is an input: Load it.
        #    - If v is internal: Ensure all parents are in fast memory (load if not), compute v.
        #    - After computing v, delete parents that are no longer needed (last_use[p] <= current_index).

        ops = []
        fast_mem = set()

        for v in topo_order:
            if time.time() > DEADLINE:
                break

            ps = parents.get(v, [])

            if not ps:
                # Input vertex - start blue
                if v not in fast_mem:
                    ops.append(['load', v])
                    fast_mem.add(v)
            else:
                # Compute vertex - need all parents in fast memory
                for p in ps:
                    if p not in fast_mem:
                        ops.append(['load', p])
                        fast_mem.add(p)

                # Compute v
                ops.append(['compute', v])
                fast_mem.add(v)

                # Delete parents that are no longer needed
                # A parent p can be deleted if:
                # 1. p is not an output (outputs must finish blue)
                # 2. last_use[p] <= topo_index[v] (all children of p have been processed)
                for p in ps:
                    if p in output_set:
                        continue

                    if last_use.get(p, -1) <= topo_index[v]:
                        ops.append(['delete', p])
                        fast_mem.discard(p)

        # Ensure all outputs are in fast memory
        for o in outputs:
            if o not in fast_mem:
                ops.append(['load', o])
                fast_mem.add(o)

        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            return ops[:20000]

    except Exception as e:
        return []
# EVOLVE-BLOCK-END
