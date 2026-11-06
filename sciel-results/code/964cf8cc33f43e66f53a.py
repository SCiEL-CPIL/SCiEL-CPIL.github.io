"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = t0 + 75  # 15s safety margin
    
    # --- Parse DAG ---
    def _get(obj, *names, default=None):
        if isinstance(obj, dict):
            for n in names:
                if n in obj:
                    return obj[n]
        else:
            for n in names:
                if hasattr(obj, n):
                    return getattr(obj, n)
        return default
    
    vertices = _get(dag, 'vertices', 'nodes', 'V', default=[])
    parents_raw = _get(dag, 'parents', 'edges', 'preds', 'in_edges', default={})
    outputs = _get(dag, 'outputs', 'sinks', 'Y', default=[])
    inputs = _get(dag, 'inputs', 'sources', 'X', default=[])
    
    # Normalize
    vertices = list(vertices) if vertices is not None else []
    outputs = list(outputs) if outputs is not None else []
    inputs = list(inputs) if inputs is not None else []
    
    # Build parent map: v -> list of parents
    parent_map = defaultdict(list)
    if isinstance(parents_raw, dict):
        for k, v in parents_raw.items():
            parent_map[k] = list(v) if v else []
    elif isinstance(parents_raw, list):
        for item in parents_raw:
            if len(item) == 2:
                p, c = item
                parent_map[c].append(p)
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                c = item[0]
                for p in item[1:]:
                    parent_map[c].append(p)
    
    # All vertices
    all_v = set(vertices)
    for v, ps in parent_map.items():
        all_v.add(v)
        all_v.update(ps)
    all_v.update(outputs)
    all_v.update(inputs)
    
    # Children map
    child_map = defaultdict(list)
    for v, ps in parent_map.items():
        for p in ps:
            child_map[p].append(v)
    
    # In-degree for topo sort
    in_deg = {v: len(parent_map.get(v, [])) for v in all_v}
    queue = deque([v for v in all_v if in_deg[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in child_map.get(v, []):
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    # Add any remaining (shouldn't happen for DAG)
    for v in all_v:
        if v not in topo:
            topo.append(v)
    
    # Identify inputs (no parents) and outputs
    input_set = set(inputs) if inputs else set()
    if not input_set:
        input_set = set(v for v in all_v if not parent_map.get(v))
    
    output_set = set(outputs) if outputs else set()
    if not output_set:
        output_set = set(v for v in all_v if not child_map.get(v))
    
    # Use count: how many times each vertex is used as a parent
    use_count = {v: len(child_map.get(v, [])) for v in all_v}
    
    # --- Generate schedule ---
    ops = []
    in_fast = set()       # vertices in fast memory
    computed = set()      # vertices that have been computed
    stored_slow = set()   # vertices stored to slow memory
    
    def elapsed():
        return time.time() - t0
    
    for v in topo:
        if elapsed() > DEADLINE:
            break
        if len(ops) >= 20000:
            break
        
        is_input = v in input_set
        is_output = v in output_set
        
        if is_input:
            # Load input into fast memory
            if v not in in_fast:
                ops.append(['load', v])
                in_fast.add(v)
            computed.add(v)
        else:
            # Need to compute v from its parents
            parents = parent_map.get(v, [])
            # Ensure all parents are in fast memory
            for p in parents:
                if p not in in_fast:
                    ops.append(['load', p])
                    in_fast.add(p)
            
            # Compute v
            ops.append(['compute', v])
            in_fast.add(v)
            computed.add(v)
            
            # Decrement use count for parents
            for p in parents:
                use_count[p] -= 1
        
        # If output, store to slow memory
        if is_output:
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
                stored_slow.add(v)
        
        # Enforce fast memory budget
        while len(in_fast) > fast_memory_budget:
            # Find a vertex to evict: not an output, use_count == 0
            candidates = [x for x in in_fast if x not in output_set and use_count.get(x, 0) <= 0]
            if not candidates:
                # Evict any non-output
                candidates = [x for x in in_fast if x not in output_set]
            if not candidates:
                break
            # Pick the one with smallest use_count
            to_evict = min(candidates, key=lambda x: use_count.get(x, 0))
            ops.append(['delete', to_evict])
            in_fast.discard(to_evict)
    
    # Final: ensure all outputs are stored
    for v in output_set:
        if v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
    
    return ops[:20000]
# EVOLVE-BLOCK-END
