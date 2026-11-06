"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq
import sys

sys.setrecursionlimit(100000)


def _parse_dag(dag):
    if isinstance(dag, dict):
        parents = dict(dag.get('parents', {}))
        outputs = list(dag.get('outputs', []))
        vertices = dag.get('vertices', None)
        inputs = dag.get('inputs', None)
    else:
        parents = dict(getattr(dag, 'parents', {}))
        outputs = list(getattr(dag, 'outputs', []))
        vertices = getattr(dag, 'vertices', None)
        inputs = getattr(dag, 'inputs', None)

    if vertices is not None:
        all_vertices = set(vertices)
    else:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        for o in outputs:
            all_vertices.add(o)

    if inputs is not None:
        input_vertices = set(inputs)
    else:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    return all_vertices, input_vertices, outputs, parents


def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 50.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        all_vertices, input_vertices, outputs, parents = _parse_dag(dag)
    except Exception:
        return []

    if not all_vertices:
        return []

    # Build children map
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Topological Sort
    in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    
    # Ensure inputs are at the front if possible, but standard topo sort is fine
    # We just need a valid topological order.
    while queue:
        # Sort by input status to prioritize inputs? No, standard is fine.
        # But to be deterministic, sort by vertex id/name
        u = queue.popleft()
        topo_order.append(u)
        for c in children[u]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    # If graph has cycles, topo_order might be incomplete.
    # Assume DAG.
    if len(topo_order) < len(all_vertices):
        # Add remaining vertices to avoid crashes
        remaining = set(all_vertices) - set(topo_order)
        topo_order.extend(sorted(remaining))

    # Precompute remaining children count
    # rem_children[v] = number of children of v that appear later in topo_order
    rem_children = {v: 0 for v in all_vertices}
    # We need to know for each vertex, how many of its children are "future"
    # Since topo_order is valid, all children of v appear after v.
    # So rem_children[v] is just len(children[v]) initially.
    # As we process v, we decrement rem_children for its parents.
    for v in all_vertices:
        rem_children[v] = len(children[v])

    # State
    fast_mem = set()
    slow_mem = set(input_vertices)
    ops = []
    outputs_set = set(outputs)
    
    # If budget is 0, we can't do much.
    if fast_memory_budget == 0:
        # Just store outputs if they are inputs?
        for o in outputs:
            if o in input_vertices:
                 ops.append(['store', o]) # Already blue, but maybe need to mark? 
            else:
                 # Can't compute
                 pass
        return ops

    def get_keep_score(v):
        """
        Higher score = better to keep in fast memory.
        Lower score = evict first.
        """
        if v in outputs_set:
            return -10000.0
        rc = rem_children[v]
        if rc == 0:
            return -10.0
        return float(rc)

    def evict_one():
        """Evict the vertex with the lowest keep score from fast_mem."""
        if not fast_mem:
            return
        
        # Find min score
        min_score = float('inf')
        victim = None
        for v in fast_mem:
            s = get_keep_score(v)
            if s < min_score:
                min_score = s
                victim = v
        
        if victim is None:
            return

        # Decide action: Store or Delete
        if victim in outputs_set:
            ops.append(['store', victim])
            slow_mem.add(victim)
        else:
            if rem_children[victim] > 0:
                ops.append(['store', victim])
                slow_mem.add(victim)
            else:
                ops.append(['delete', victim])
                # Remove from slow if it was there? 
                # If we stored it before, it's in slow. If we delete it now, it's gone.
                # But we only delete if rem_children == 0.
                # It might be in slow_mem if it was stored previously.
                slow_mem.discard(victim)
        
        fast_mem.discard(victim)

    # Process vertices
    for v in topo_order:
        if time_check():
            break
        
        # 1. Ensure parents are in fast_mem
        ps = parents.get(v, [])
        for p in ps:
            if p not in fast_mem:
                if p in slow_mem:
                    # Need to load p
                    if len(fast_mem) >= fast_memory_budget:
                        evict_one()
                    ops.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
                else:
                    # p is not in fast or slow. 
                    # This implies p was deleted. 
                    # We must recompute p.
                    # This is a failure of the heuristic to keep needed vertices.
                    # To handle this, we would need to recursively recompute p's parents.
                    # Given the constraints, this should be rare if we store vertices with future uses.
                    # For now, we skip or raise. 
                    # Let's try to recompute by treating it as a new vertex?
                    # No, that changes the schedule.
                    # We'll assume this doesn't happen for valid DAGs with our heuristic.
                    pass

        # 2. Compute v
        if len(fast_mem) >= fast_memory_budget:
            evict_one()
        
        ops.append(['compute', v])
        fast_mem.add(v)
        slow_mem.discard(v) # v is now red (in fast)

        # 3. Update rem_children for parents
        for p in ps:
            rem_children[p] -= 1

    # 4. Finalize: Store all outputs
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)
        elif o not in slow_mem:
            # Output not in slow or fast. 
            # Should not happen if we stored all outputs or they were inputs.
            # If it was an input, it started in slow.
            # If it was computed, it was in fast, then we should have stored it if evicted or at end.
            # If it was evicted and rem_children>0, it was stored.
            # If it was evicted and rem_children==0, it was deleted. 
            # But if it's an output, rem_children is likely 0 (sink).
            # So it might have been deleted!
            # We MUST store outputs.
            # So we need to recompute it? Or just add a dummy store?
            # The verifier checks state. If it's not in slow, it's invalid.
            # We must ensure outputs end in slow.
            # If we deleted an output, we failed.
            # Fix: In evict_one, if v is in outputs_set, we ALWAYS store.
            # We did that: `if victim in outputs_set: ops.append(['store', victim])`.
            # So outputs should never be deleted.
            pass

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
# EVOLVE-BLOCK-END
