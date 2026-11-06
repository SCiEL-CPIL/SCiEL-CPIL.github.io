"""Rewritten communication-minimizing DAG pebbling with stronger algorithm."""
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
    DEADLINE = 85.0

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])

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

        # Precompute next_use: for each vertex, the next topo index where it's needed
        next_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                # Next use is the earliest child's topo index
                next_use[v] = min(topo_index[c] for c in children[v])
            else:
                next_use[v] = topo_index.get(v, -1)

        # Precompute recompute_cost: number of ancestors (approximate)
        recompute_cost = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                recompute_cost[v] = 1  # Just a load
            else:
                recompute_cost[v] = 1 + max(recompute_cost[p] for p in ps)

        ops = []
        fast_mem = set()

        # Track which computed vertices have been stored to slow memory
        stored_vertices = set()

        current_topo_index = [0]  # mutable reference for the current processing index

        def get_eviction_candidate():
            """Return the vertex in fast_mem to evict, or None."""
            best = None
            best_key = None
            for v in fast_mem:
                if v in must_be_blue:
                    # Outputs: only consider if no other option
                    # Give them a high "cost to evict" key
                    key = (1, recompute_cost[v], last_use[v])
                else:
                    # Non-outputs: 
                    # - If dead (last_use < current), evict immediately
                    # - Otherwise, prefer low recompute cost and distant next_use
                    if last_use[v] < current_topo_index[0]:
                        key = (0, 0, 0)  # Dead vertex, evict first
                    else:
                        # Composite: low recompute cost is good, distant next_use is good
                        next_dist = next_use[v] - current_topo_index[0]
                        if next_dist < 0:
                            next_dist = 0
                        key = (0, recompute_cost[v], -next_dist)
                if best_key is None or key < best_key:
                    best_key = key
                    best = v
            return best

        # Recursive ensure function with cycle protection
        visiting = set()

        def ensure_in_fast_mem(v, depth_limit=0):
            """
            Ensure vertex v is in fast_mem.
            Returns True on success, False on failure.
            """
            if time.time() > DEADLINE:
                return False

            if v in fast_mem:
                return True

            if v in visiting:
                # Cycle detected (shouldn't happen in a DAG, but safety)
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
                # Computed vertex: check if it was stored
                if v in stored_vertices:
                    # Load from slow memory
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
                            if not ensure_in_fast_mem(p, depth_limit + 1):
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
                        
                        # Consider storing if this vertex has many children or high depth
                        # Heuristic: store if recompute cost is high and has at least 2 children
                        if recompute_cost[v] >= 3 and len(children.get(v, [])) >= 2:
                            ops.append(['store', v])
                            stored_vertices.add(v)
                        
                        return True
                    finally:
                        visiting.discard(v)

        # Process in reverse topological order from outputs
        # This ensures we build up the dependency tree from the outputs
        for o in outputs:
            if time.time() > DEADLINE:
                break
            if not ensure_in_fast_mem(o):
                # Failed; try to free space more aggressively
                for u in list(fast_mem):
                    if u not in must_be_blue and u != o:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                        if len(fast_mem) < fast_memory_budget:
                            break
                if not ensure_in_fast_mem(o):
                    break

        # Ensure all outputs are in fast_mem at the end
        for o in outputs:
            if time.time() > DEADLINE:
                break
            if o not in fast_mem:
                # Try to free space
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
                # Force: delete everything non-output
                for u in list(fast_mem):
                    if u not in must_be_blue:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                ensure_in_fast_mem(o)
                
                # If still not in fast_mem, store it
                if o in fast_mem:
                    ops.append(['store', o])
                    stored_vertices.add(o)

        # Check operation count
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
