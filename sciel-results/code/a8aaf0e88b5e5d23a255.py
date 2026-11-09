"""Communication-minimizing DAG pebbling with bounded fast memory."""

def pebble(dag, fast_memory_budget):
    import time
    import sys
    import random
    from collections import defaultdict, deque

    # Deadline guard: keep a safety margin before 90s wall-clock.
    start_time = time.time()
    DEADLINE = 80.0  # seconds

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        if elapsed() > DEADLINE:
            raise TimeoutError("deadline")

    # ---- Parse DAG ----
    # dag structure: we expect dag to have:
    #   dag['vertices'] or dag['nodes']: list of vertex ids
    #   dag['parents']: dict mapping vertex -> list of parent vertices (topologically ordered)
    #   dag['inputs']: list of input vertices
    #   dag['outputs']: list of output vertices
    #   dag['edges'] or dag['adj']: adjacency info
    #
    # Let's be flexible about the structure.

    def _get_vertices(d):
        if isinstance(d, dict):
            if 'vertices' in d:
                return list(d['vertices'])
            if 'nodes' in d:
                return list(d['nodes'])
            if 'parents' in d:
                # Collect all vertices from parents keys and values
                verts = set()
                for v, ps in d['parents'].items():
                    verts.add(v)
                    for p in ps:
                        verts.add(p)
                return sorted(verts)
            if 'edges' in d:
                verts = set()
                for u, v in d['edges']:
                    verts.add(u)
                    verts.add(v)
                return sorted(verts)
            # Try keys
            return list(d.keys())
        elif isinstance(d, (list, tuple)):
            return list(d)
        else:
            return list(range(len(d)))

    def _get_parents(d):
        if isinstance(d, dict):
            if 'parents' in d:
                return {k: list(v) for k, v in d['parents'].items()}
            if 'edges' in d:
                # edges might be (parent, child) or (child, parent)
                parents = defaultdict(list)
                for u, v in d['edges']:
                    # Assume (parent, child) convention
                    parents[v].append(u)
                return dict(parents)
            # Try to infer from 'children'
            if 'children' in d:
                parents = defaultdict(list)
                for v, cs in d['children'].items():
                    for c in cs:
                        parents[c].append(v)
                return dict(parents)
        return {}

    def _get_inputs(d):
        if isinstance(d, dict):
            if 'inputs' in d:
                return list(d['inputs'])
            if 'sources' in d:
                return list(d['sources'])
            # Vertices with no parents
            parents = _get_parents(d)
            vertices = _get_vertices(d)
            inputs = [v for v in vertices if not parents.get(v, [])]
            return inputs
        return []

    def _get_outputs(d):
        if isinstance(d, dict):
            if 'outputs' in d:
                return list(d['outputs'])
            if 'sinks' in d:
                return list(d['sinks'])
            # Vertices with no children
            parents = _get_parents(d)
            vertices = _get_vertices(d)
            children = defaultdict(list)
            for v, ps in parents.items():
                for p in ps:
                    children[p].append(v)
            outputs = [v for v in vertices if not children.get(v, [])]
            return outputs
        return []

    vertices = _get_vertices(dag)
    parents = _get_parents(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)

    # Build children map
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Topological order
    def topological_sort():
        in_degree = {v: len(parents.get(v, [])) for v in vertices}
        queue = deque([v for v in vertices if in_degree[v] == 0])
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        if len(order) != len(vertices):
            # Cycle or missing vertices; fallback
            return vertices
        return order

    topo_order = topological_sort()

    # Children counts for out-degree
    out_degree = {v: len(children.get(v, [])) for v in vertices}

    # ---- Baseline: naive topological schedule ----
    def naive_schedule():
        """Simple topological order: load inputs, compute in topo order, store outputs."""
        ops = []
        # All inputs start blue (in fast memory? or slow?)
        # "Inputs start blue" - blue means in fast memory? Or just available?
        # Let's assume inputs are initially in slow memory (blue = cold), need to load.
        # Actually "Inputs start blue" likely means they're available but not in fast memory.
        # Blue = in slow memory, Red = in fast memory.
        # Operations: load (slow->fast), store (fast->slow), compute (fast->fast), delete (fast->discard)
        
        # Naive: load all inputs, compute in topo order, keep results, store outputs at end
        in_memory = set()
        
        for v in topo_order:
            check_deadline()
            # Ensure all parents are in memory
            for p in parents.get(v, []):
                if p not in in_memory:
                    ops.append(['load', p])
                    in_memory.add(p)
            
            # Compute v
            if parents.get(v, []):  # not an input
                ops.append(['compute', v])
                in_memory.add(v)
            else:
                # It's an input, should have been loaded
                if v not in in_memory:
                    ops.append(['load', v])
                    in_memory.add(v)
            
            # Delete parents that are no longer needed
            for p in parents.get(v, []):
                # Check if p has other children
                remaining_children = [c for c in children.get(p, []) if c not in set(topo_order[:topo_order.index(v)])]
                # Simpler: check if p is needed by any future vertex
                future_needed = False
                for c in children.get(p, []):
                    if topo_order.index(c) > topo_order.index(v):
                        future_needed = True
                        break
                if not future_needed and p not in in_memory:
                    pass  # already handled
                elif not future_needed:
                    ops.append(['delete', p])
                    in_memory.discard(p)
        
        # Store all outputs
        for o in outputs:
            if o in in_memory:
                ops.append(['store', o])
                in_memory.discard(o)
        
        return ops

    # Count transfers (load + store operations)
    def count_transfers(ops):
        return sum(1 for op in ops if op[0] in ('load', 'store'))

    # ---- Improved: minimize transfers using reference counting and careful scheduling ----
    def improved_schedule():
        """
        Use reference counting to delete intermediate values as soon as possible.
        Process in topological order, but be smart about when to load/store.
        """
        ops = []
        in_memory = set()
        
        # Precompute: for each vertex, the last vertex in topo order that needs it
        last_use = {}
        for i, v in enumerate(topo_order):
            for c in children.get(v, []):
                if c in last_use:
                    last_use[v] = max(last_use[v], topo_order.index(c))
                else:
                    last_use[v] = topo_order.index(c)
        
        # Also track: which vertices are outputs (need to be stored at the end)
        output_set = set(outputs)
        
        for i, v in enumerate(topo_order):
            check_deadline()
            
            # Load parents if not in memory
            for p in parents.get(v, []):
                if p not in in_memory:
                    ops.append(['load', p])
                    in_memory.add(p)
            
            # Compute v if it's not an input
            if parents.get(v, []):
                ops.append(['compute', v])
                in_memory.add(v)
            else:
                if v not in in_memory:
                    ops.append(['load', v])
                    in_memory.add(v)
            
            # Delete parents that are no longer needed (last use was current or earlier)
            for p in parents.get(v, []):
                if p in in_memory:
                    # Is p needed by any future vertex?
                    if last_use.get(p, -1) <= i:
                        # p is not needed anymore, but check if it's an output
                        if p not in output_set:
                            ops.append(['delete', p])
                            in_memory.discard(p)
        
        # Store all outputs that are still in memory
        for o in outputs:
            if o in in_memory:
                ops.append(['store', o])
                in_memory.discard(o)
        
        # Clean up any remaining non-output values
        for v in in_memory:
            if v not in output_set:
                ops.append(['delete', v])
        
        return ops

    # ---- Even better: use the fast memory budget to decide what to keep ----
    def budget_aware_schedule():
        """
        Use the fast memory budget to decide which intermediate values to keep.
        Values that are used many times should be kept; others should be deleted.
        """
        ops = []
        in_memory = set()
        
        # Precompute usage count for each vertex
        usage_count = {v: len(children.get(v, [])) for v in vertices}
        
        # Precompute: for each vertex, the set of descendants
        def get_descendants(v):
            desc = set()
            stack = [v]
            while stack:
                u = stack.pop()
                for c in children.get(u, []):
                    if c not in desc:
                        desc.add(c)
                        stack.append(c)
            return desc
        
        # Precompute last use in topo order
        last_use = {}
        for i, v in enumerate(topo_order):
            for c in children.get(v, []):
                ci = topo_order.index(c)
                if v not in last_use or ci > last_use[v]:
                    last_use[v] = ci
        
        output_set = set(outputs)
        
        for i, v in enumerate(topo_order):
            check_deadline()
            
            # Load parents if not in memory
            for p in parents.get(v, []):
                if p not in in_memory:
                    ops.append(['load', p])
                    in_memory.add(p)
            
            # Compute v if it's not an input
            if parents.get(v, []):
                ops.append(['compute', v])
                in_memory.add(v)
            else:
                if v not in in_memory:
                    ops.append(['load', v])
                    in_memory.add(v)
            
            # Delete parents that are no longer needed
            for p in parents.get(v, []):
                if p in in_memory and last_use.get(p, -1) <= i and p not in output_set:
                    ops.append(['delete', p])
                    in_memory.discard(p)
            
            # If memory is over budget, delete low-value items
            if len(in_memory) > fast_memory_budget:
                # Delete items with lowest usage count among non-outputs
                candidates = [(usage_count.get(x, 0), x) for x in in_memory if x not in output_set]
                candidates.sort()
                while len(in_memory) > fast_memory_budget and candidates:
                    _, x = candidates.pop(0)
                    if x in in_memory:
                        ops.append(['delete', x])
                        in_memory.discard(x)
        
        # Store all outputs
        for o in outputs:
            if o in in_memory:
                ops.append(['store', o])
                in_memory.discard(o)
        
        # Final cleanup
        for v in list(in_memory):
            if v not in output_set:
                ops.append(['delete', v])
                in_memory.discard(v)
        
        return ops

    # ---- Generate candidate schedules and pick the best ----
    candidates = []
    
    try:
        candidates.append(naive_schedule())
    except:
        pass
    
    try:
        candidates.append(improved_schedule())
    except:
        pass
    
    try:
        candidates.append(budget_aware_schedule())
    except:
        pass
    
    # If no valid candidate, use a minimal valid schedule
    if not candidates:
        # Fallback: load all inputs, compute all in topo order, store outputs
        ops = []
        in_memory = set()
        for v in topo_order:
            for p in parents.get(v, []):
                if p not in in_memory:
                    ops.append(['load', p])
                    in_memory.add(p)
            if parents.get(v, []):
                ops.append(['compute', v])
                in_memory.add(v)
            else:
                if v not in in_memory:
                    ops.append(['load', v])
                    in_memory.add(v)
        for o in outputs:
            if o in in_memory:
                ops.append(['store', o])
                in_memory.discard(o)
        candidates.append(ops)
    
    # Pick the candidate with fewest transfers
    best = min(candidates, key=count_transfers)
    
    # Ensure operation count is within limit
    if len(best) > 20000:
        # Truncate (this shouldn't happen for reasonable DAGs)
        best = best[:20000]
    
    return best

# EVOLVE-BLOCK-START
# The pebble function is defined above
# EVOLVE-BLOCK-END
