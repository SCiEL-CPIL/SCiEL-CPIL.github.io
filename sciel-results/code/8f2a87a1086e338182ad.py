"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _parse_dag(dag):
    """Parse DAG structure into usable form."""
    if isinstance(dag, list):
        vertices = []
        parents_map = {}
        outputs = set()
        for i, entry in enumerate(dag):
            if isinstance(entry, dict):
                v = entry.get('vertex', i)
                par = entry.get('parents', [])
                is_out = entry.get('is_output', False)
                vertices.append(v)
                parents_map[v] = par
                if is_out:
                    outputs.add(v)
            elif isinstance(entry, (list, tuple)):
                if len(entry) >= 2:
                    v = entry[0]
                    par = entry[1] if isinstance(entry[1], list) else []
                    is_out = entry[2] if len(entry) > 2 else False
                    vertices.append(v)
                    parents_map[v] = par
                    if is_out:
                        outputs.add(v)
        return vertices, parents_map, outputs
    
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = dag['vertices']
            parents_map = dag.get('parents', {})
            outputs = set(dag.get('outputs', []))
            return vertices, parents_map, outputs
        vertices = list(dag.keys())
        parents_map = {v: dag[v] for v in vertices}
        outputs = set()
        return vertices, parents_map, outputs
    
    return [], {}, set()


def _compute_layers(vertices, parents_map):
    """Compute topological layers."""
    if not vertices:
        return []
    
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children[p].append(v)
    
    layer = [v for v in vertices if in_degree[v] == 0]
    layers = []
    visited = set(layer)
    
    while layer:
        layers.append(layer)
        next_layer = []
        for v in layer:
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0 and c not in visited:
                    next_layer.append(c)
                    visited.add(c)
        layer = next_layer
    
    return layers


def _pebble_layered_greedy(dag, fast_memory_budget):
    """Layer-based greedy pebbling with recompute-on-demand."""
    vertices, parents_map, outputs = _parse_dag(dag)
    if not vertices:
        return []
    
    layers = _compute_layers(vertices, parents_map)
    if not layers:
        return []
    
    # Build future-use map: vertex -> set of layer indices that use it as input
    future_use = defaultdict(set)
    for layer_idx, layer in enumerate(layers):
        for v in layer:
            for p in parents_map.get(v, []):
                future_use[p].add(layer_idx)
    
    input_vertices = set(v for v in vertices if not parents_map.get(v))
    
    fast_memory = set()
    ops = []
    
    for layer_idx, layer in enumerate(layers):
        # Determine which parents are needed for this layer
        needed_parents = set()
        for v in layer:
            for p in parents_map.get(v, []):
                needed_parents.add(p)
        
        # Load needed parents not in fast memory
        for p in needed_parents:
            if p not in fast_memory:
                ops.append(('load', p))
                fast_memory.add(p)
        
        # Compute each vertex in this layer
        for v in layer:
            ops.append(('compute', v))
            fast_memory.add(v)
        
        # Determine what to keep: vertices used in future layers or outputs
        keep = set()
        for v in fast_memory:
            future_layers = [l for l in future_use.get(v, set()) if l > layer_idx]
            if future_layers:
                keep.add(v)
            if v in outputs:
                keep.add(v)
        
        # Enforce memory budget
        if len(keep) > fast_memory_budget:
            use_count = {}
            for v in keep:
                future_layers = [l for l in future_use.get(v, set()) if l > layer_idx]
                use_count[v] = len(future_layers)
            sorted_keep = sorted(keep, key=lambda x: use_count[x], reverse=True)
            keep = set(sorted_keep[:fast_memory_budget])
        
        # Delete vertices not kept
        for v in fast_memory - keep:
            if v in outputs:
                ops.append(('store', v))
            else:
                ops.append(('delete', v))
            fast_memory.discard(v)
        
        # Store outputs that are kept (to ensure they're blue at end)
        for v in keep:
            if v in outputs:
                ops.append(('store', v))
    
    # Final cleanup: ensure all outputs are stored
    for v in outputs:
        if v in fast_memory:
            ops.append(('store', v))
            fast_memory.discard(v)
    
    return ops


def pebble(dag, fast_memory_budget):
    """Main entry point for DAG pebbling."""
    start_time = time.time()
    
    try:
        ops = _pebble_layered_greedy(dag, fast_memory_budget)
        if len(ops) <= 20000:
            return ops
    except Exception:
        pass
    
    # Fallback: simple topological order
    try:
        vertices, parents_map, outputs = _parse_dag(dag)
        if not vertices:
            return []
        
        ops = []
        in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
        children = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children[p].append(v)
        
        queue = [v for v in vertices if in_degree[v] == 0]
        computed = set()
        
        while queue:
            v = queue.pop(0)
            if v in computed:
                continue
            for p in parents_map.get(v, []):
                if p not in computed:
                    ops.append(('load', p))
            ops.append(('compute', v))
            computed.add(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        for v in outputs:
            ops.append(('store', v))
        
        return ops
    except Exception:
        return []
# EVOLVE-BLOCK-END
