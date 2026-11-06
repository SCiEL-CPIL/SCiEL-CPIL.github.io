"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85
    
    # Parse DAG - handle multiple possible formats
    if isinstance(dag, dict):
        # Try common key names
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('preds', dag.get('ancestors', {})))
        outputs = set(dag.get('outputs', dag.get('sinks', dag.get('terminal', []))))
        inputs = set(dag.get('inputs', dag.get('sources', dag.get('root', []))))
        
        if vertices is None:
            # Infer from parents keys
            all_v = set()
            for v, ps in parents.items():
                all_v.add(v)
                for p in ps:
                    all_v.add(p)
            vertices = list(all_v)
            if not inputs:
                inputs = set(v for v in vertices if not parents.get(v, []))
            if not outputs:
                children = defaultdict(set)
                for v, ps in parents.items():
                    for p in ps:
                        children[p].add(v)
                outputs = set(v for v in vertices if not children.get(v, set()))
    elif hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
        parents = {v: list(dag.parents(v)) for v in vertices} if hasattr(dag, 'parents') else {}
        outputs = set(getattr(dag, 'outputs', []))
        inputs = set(getattr(dag, 'inputs', []))
    else:
        # Fallback: assume dag is a list of (vertex, parents) tuples
        vertices = []
        parents = {}
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v, ps = item[0], item[1]
                vertices.append(v)
                parents[v] = list(ps)
            elif isinstance(item, dict):
                v = item.get('vertex', item.get('v', len(vertices)))
                ps = item.get('parents', item.get('p', []))
                vertices.append(v)
                parents[v] = list(ps)
        outputs = set()
        inputs = set(v for v in vertices if not parents.get(v, []))
    
    if not vertices:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Ensure inputs/outputs are determined
    if not inputs:
        inputs = set(v for v in vertices if not parents.get(v, []))
    if not outputs:
        outputs = set(v for v in vertices if not children.get(v, []))
    
    # Topological sort (Kahn's algorithm)
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) < len(vertices):
        # Cycle detected, append remaining
        remaining = set(vertices) - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    n = len(topo_order)
    vertex_to_idx = {v: i for i, v in enumerate(topo_order)}
    
    # Precompute next use
    next_use = {}
    for i, v in enumerate(topo_order):
        next_idx = -1
        for j in range(i + 1, n):
            if v in parents.get(topo_order[j], []):
                next_idx = j
                break
        next_use[v] = next_idx
    
    # Greedy schedule
    in_fast = set()
    schedule = []
    
    # Load inputs
    for v in inputs:
        in_fast.add(v)
        schedule.append(['load', v])
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        if v in inputs:
            continue
        
        # Ensure parents are in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    evict_v = None
                    max_next = -2
                    for cand in in_fast:
                        nu = next_use.get(cand, -1)
                        if nu > max_next:
                            max_next = nu
                            evict_v = cand
                    if evict_v is not None:
                        in_fast.remove(evict_v)
                        schedule.append(['store', evict_v])
                    else:
                        break
                in_fast.add(p)
                schedule.append(['load', p])
        
        schedule.append(['compute', v])
        
        needs_v = (next_use.get(v, -1) != -1) or (v in outputs)
        if needs_v:
            while len(in_fast) >= fast_memory_budget:
                evict_v = None
                max_next = -2
                for cand in in_fast:
                    nu = next_use.get(cand, -1)
                    if nu > max_next:
                        max_next = nu
                        evict_v = cand
                if evict_v is not None:
                    in_fast.remove(evict_v)
                    schedule.append(['store', evict_v])
                else:
                    break
            in_fast.add(v)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if time.time() > deadline:
            break
        if v not in in_fast:
            while len(in_fast) >= fast_memory_budget:
                evict_v = None
                max_next = -2
                for cand in in_fast:
                    if cand not in outputs:
                        nu = next_use.get(cand, -1)
                        if nu > max_next:
                            max_next = nu
                            evict_v = cand
                if evict_v is not None:
                    in_fast.remove(evict_v)
                    schedule.append(['store', evict_v])
                else:
                    break
            in_fast.add(v)
            schedule.append(['load', v])
    
    return schedule
# EVOLVE-BLOCK-END

