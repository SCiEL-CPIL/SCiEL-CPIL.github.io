"""Communication-minimizing DAG pebbling - optimized approach."""
# EVOLVE-BLOCK-START
import time
import sys
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
    DEADLINE = 80.0  # Wall-clock limit with safety margin

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])

        if not parents and not outputs and not inputs:
            return []

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

        remaining = all_vertices - set(topo_order)
        if remaining:
            topo_order.extend(sorted(remaining))

        topo_index = {v: i for i, v in enumerate(topo_order)}

        is_input = set(inputs)

        # For each output, compute its ancestor set
        def get_ancestors(target):
            anc = set()
            stack = [target]
            while stack:
                cur = stack.pop()
                if cur in anc:
                    continue
                anc.add(cur)
                for p in parents.get(cur, []):
                    if p not in anc:
                        stack.append(p)
            return anc

        # Precompute ancestors for all outputs
        output_ancestors = {}
        for o in outputs:
            output_ancestors[o] = get_ancestors(o)

        ops = []
        fast_mem = set()

        def time_check():
            return time.time() - start_time < DEADLINE

        def make_space(needed_set):
            """
            Ensure there's room in fast_mem by deleting vertices not in needed_set.
            needed_set: set of vertices that must be kept.
            """
            while len(fast_mem) >= fast_memory_budget:
                # Find a vertex to delete
                deleted = False
                for v in list(fast_mem):
                    if v not in needed_set:
                        ops.append(['delete', v])
                        fast_mem.discard(v)
                        deleted = True
                        break
                if not deleted:
                    # All vertices are needed but we're full - this is a problem
                    # Force delete one (shouldn't happen with adequate budget)
                    if fast_mem:
                        v = next(iter(fast_mem))
                        ops.append(['delete', v])
                        fast_mem.discard(v)
                    break

        def compute_vertex(v, needed_set):
            """
            Recursively ensure v is in fast_mem.
            needed_set: set of vertices that should be kept in memory.
            """
            if not time_check():
                return False

            if v in fast_mem:
                return True

            ps = parents.get(v, [])

            if not ps:
                # Input vertex
                make_space(needed_set)
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                # Computed vertex
                # Ensure all parents are in fast_mem
                for p in ps:
                    if not compute_vertex(p, needed_set):
                        return False

                # Now compute v
                make_space(needed_set)
                ops.append(['compute', v])
                fast_mem.add(v)
                return True

        # Process each output one at a time
        for o in outputs:
            if not time_check():
                break

            # The set of vertices needed for this output
            needed = output_ancestors[o] | {o}

            if not compute_vertex(o, needed):
                if pebble_baseline:
                    return pebble_baseline(dag, fast_memory_budget)
                break

            # After computing this output, free all non-output values
            for v in list(fast_mem):
                if v not in outputs:
                    ops.append(['delete', v])
                    fast_mem.discard(v)

        # Final: ensure all outputs are in fast_mem
        for o in outputs:
            if o not in fast_mem:
                if not time_check():
                    break
                needed = output_ancestors[o] | {o}
                if not compute_vertex(o, needed):
                    if pebble_baseline:
                        return pebble_baseline(dag, fast_memory_budget)
                    break
                # Free non-outputs
                for v in list(fast_mem):
                    if v not in outputs:
                        ops.append(['delete', v])
                        fast_mem.discard(v)

        if len(ops) <= 20000:
            return ops
        else:
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]

    except Exception as e:
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
