# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    1. Parse the DAG structure
    2. Use a recursive pebbling algorithm that minimizes communication
    3. Track fast memory usage carefully
    4. Return operations list
    
    For layered DAGs, use divide-and-conquer:
    - If budget >= number of intermediate values, compute in topo order
    - Otherwise, use recursive strategy that recomputes values
    """
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    try:
        # Parse DAG structure
        vertices = dag.get('vertices', [])
        if not vertices:
            if 'nodes' in dag:
                vertices = dag['nodes']
            elif isinstance(dag, dict):
                vertices = list(dag.keys())
            else:
                vertices = []
        
        parents_map = dag.get('parents', {})
        if not parents_map and isinstance(dag, dict):
            parents_map = {v: list(parents) for v, parents in dag.items()}
        
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        
        if not inputs:
            inputs = set(v for v in vertices if not parents_map.get(v, []))
        
        if not outputs:
            children_map = {v: [] for v in vertices}
            for v in vertices:
                for p in parents_map.get(v, []):
                    if p in children_map:
                        children_map[p].append(v)
            outputs = set(v for v in vertices if not children_map.get(v, []))
        
        # Build children map
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents_map.get(v, []):
                if p in children_map:
                    children_map[p].append(v)
        
        topo_order = list(vertices)
        n_vertices = len(topo_order)
        
        # Count non-input vertices (need computation)
        non_inputs = [v for v in topo_order if v not in inputs]
        n_compute = len(non_inputs)
        
        # Strategy: if budget is large enough, just do topo order with minimal ops
        # Naive: load each input once, compute each non-input, store each output
        # Total transfers = len(inputs) + len(outputs)
        
        # If budget >= max concurrent needs, we can do better
        # For now, use a simple strategy:
        # 1. Load all inputs into fast memory (if budget allows)
        # 2. Compute in topo order
        # 3. Store outputs
        # 4. Delete intermediate values as we go
        
        # Better strategy: use a recursive approach for small budgets
        # But first, let's handle the simple case where budget is sufficient
        
        # Determine if we can keep all intermediate values in fast memory
        # Max intermediate values needed at once = max over all vertices of (number of parents that are not inputs)
        # Actually, it's more complex. Let's just try the simple approach first.
        
        ops = []
        in_fast = set()
        is_blue = set(inputs)
        
        # Precompute last use for each vertex
        last_use = {}
        for i, v in enumerate(reversed(topo_order)):
            pos = n_vertices - 1 - i
            if v in outputs:
                last_use[v] = n_vertices  # Needed until the end
            elif not children_map.get(v, []):
                last_use[v] = n_vertices
            else:
                max_pos = 0
                for child in children_map.get(v, []):
                    child_pos = topo_order.index(child) if child in topo_order else n_vertices
                    max_pos = max(max_pos, child_pos)
                last_use[v] = max_pos
        
        # Simple strategy: process in topo order
        # Load parents as needed, compute, delete when no longer needed
        for i, v in enumerate(topo_order):
            if time.time() > deadline:
                break
            
            if v in inputs:
                continue
            
            parents = parents_map.get(v, [])
            
            # Ensure parents are available
            for p in parents:
                if p not in is_blue and p not in in_fast:
                    # Need to load p
                    if len(in_fast) >= fast_memory_budget:
                        # Find a vertex to delete
                        # Delete one that's not needed for current or next few vertices
                        next_set = set(topo_order[i:min(i+3, n_vertices)])
                        next_parents = set()
                        for nv in next_set:
                            next_parents.update(parents_map.get(nv, []))
                        
                        candidates = [x for x in in_fast if x not in next_parents]
                        if candidates:
                            # Delete the one with latest last_use
                            to_delete = max(candidates, key=lambda x: last_use.get(x, 0))
                        else:
                            to_delete = max(in_fast, key=lambda x: last_use.get(x, 0))
                        
                        ops.append(['delete', to_delete])
                        in_fast.remove(to_delete)
                    
                    ops.append(['load', p])
                    in_fast.add(p)
            
            # Compute v
            ops.append(['compute', v])
            in_fast.add(v)
            
            # If v is an output, store it
            if v in outputs:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
            
            # Delete parents that are no longer needed
            for p in parents:
                if p in in_fast and last_use.get(p, 0) <= i:
                    ops.append(['delete', p])
                    in_fast.remove(p)
        
        # Ensure all outputs are blue
        for v in outputs:
            if v not in is_blue and v in in_fast:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
            elif v not in is_blue and v not in in_fast:
                # Need to load and store
                if len(in_fast) >= fast_memory_budget:
                    to_delete = max(in_fast, key=lambda x: last_use.get(x, 0)) if in_fast else None
                    if to_delete:
                        ops.append(['delete', to_delete])
                        in_fast.remove(to_delete)
                ops.append(['load', v])
                in_fast.add(v)
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
        
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
        
    except Exception as e:
        return []
# EVOLVE-BLOCK-END
