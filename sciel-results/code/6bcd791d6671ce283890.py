"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve communication-minimizing DAG pebbling.
    
    dag: dict with keys:
        - 'vertices': list of vertex IDs
        - 'edges': list of (parent, child) pairs  
        - 'inputs': list of input vertex IDs (start with pebbles/blue)
        - 'outputs': list of output vertex IDs (must finish with pebbles/blue)
        - 'parents': dict mapping vertex -> list of parent vertices
    
    fast_memory_budget: int, max number of pebbles (fast memory slots)
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 55.0
    
    vertices = dag.get('vertices', [])
    n = len(vertices)
    if n == 0:
        return []
    
    # Build parent/children maps
    parents_map = dag.get('parents', {})
    if not parents_map:
        edges = dag.get('edges', [])
        parents_map = defaultdict(list)
        for parent, child in edges:
            parents_map[child].append(parent)
    
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Topological order
    topo_order = list(vertices)
    if not dag.get('topologically_ordered', True):
        in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
        queue = deque([v for v in vertices if in_degree[v] == 0])
        topo_order = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            for c in children_map.get(v, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
    
    # Compute layers (longest path from inputs)
    layer = {}
    for v in topo_order:
        if v in inputs:
            layer[v] = 0
        else:
            parent_layers = [layer.get(p, -1) for p in parents_map.get(v, [])]
            layer[v] = max(parent_layers) + 1 if parent_layers else 0
    
    # Usage count: how many children use this vertex
    usage_count = {v: len(children_map.get(v, [])) for v in vertices}
    
    # Cache score: how valuable it is to keep this vertex in fast memory
    cache_score = {}
    for v in vertices:
        score = 0
        if v in outputs:
            score += 1000
        score += usage_count[v] * 10
        score += layer[v] * 0.1
        cache_score[v] = score
    
    # State
    has_pebble = set(inputs)  # Vertices with pebbles (blue)
    computed = set(inputs)  # Vertices that have been computed
    ops = []
    
    # Remaining children count: how many children of v haven't been processed yet
    remaining_children = {v: len(children_map.get(v, [])) for v in vertices}
    
    # Process in topological order
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            # Emergency exit: ensure outputs have pebbles
            for out in outputs:
                if out not in has_pebble:
                    if out in computed:
                        ops.append(['load', out])
                        has_pebble.add(out)
            break
        
        if v in inputs:
            continue  # Already has pebble
        
        # Need to compute v
        # Ensure all parents have pebbles
        for p in parents_map.get(v, []):
            if p not in has_pebble:
                if p in computed:
                    # Load parent
                    ops.append(['load', p])
                    has_pebble.add(p)
                else:
                    # This shouldn't happen in topological order
                    pass
        
        # Check memory: if over budget, evict something
        while len(has_pebble) > fast_memory_budget:
            # Find a vertex to evict
            # Prefer evicting vertices that are no longer needed
            # (all their children have been processed) and are not outputs
            evict_candidates = []
            for mem_v in has_pebble:
                if mem_v in outputs:
                    continue
                # Check if all children are processed
                all_children_done = all(
                    c in computed for c in children_map.get(mem_v, [])
                )
                if all_children_done:
                    evict_candidates.append(mem_v)
            
            if evict_candidates:
                # Evict lowest cache score
                evict_v = min(evict_candidates, key=lambda x: cache_score.get(x, 0))
                ops.append(['delete', evict_v])
                has_pebble.discard(evict_v)
            else:
                # No safe candidate, evict lowest score non-output
                non_outputs = [mem_v for mem_v in has_pebble if mem_v not in outputs]
                if non_outputs:
                    evict_v = min(non_outputs, key=lambda x: cache_score.get(x, 0))
                    ops.append(['delete', evict_v])
                    has_pebble.discard(evict_v)
                else:
                    # Can't evict, break
                    break
        
        # Compute v
        ops.append(['compute', v])
        computed.add(v)
        has_pebble.add(v)
        
        # Update remaining children for parents
        for p in parents_map.get(v, []):
            remaining_children[p] = remaining_children.get(p, 0) - 1
    
    # Ensure outputs have pebbles
    for out in outputs:
        if out not in has_pebble:
            if out in computed:
                ops.append(['load', out])
                has_pebble.add(out)
    
    return ops

# EVOLVE-BLOCK-END
