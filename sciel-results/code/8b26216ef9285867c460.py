"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling with layered strategy.
    
    Strategy:
    1. Parse DAG structure robustly.
    2. For layered DAGs, use layer-by-layer processing with smart memory management.
    3. Track which vertices are still needed as parents for future layers.
    4. Minimize transfers by keeping parents in memory only when needed.
    """
    start_time = time.time()
    DEADLINE = start_time + 80  # Safety margin before 90s kill
    
    def get_baseline():
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
    
    try:
        # --- 1. Parse DAG Structure ---
        vertices = None
        if hasattr(dag, 'vertices'):
            vertices = list(dag.vertices)
        elif isinstance(dag, dict) and 'vertices' in dag:
            vertices = list(dag['vertices'])
        elif isinstance(dag, (list, tuple)):
            vertices = list(dag)
        else:
            for attr in ['vertices', 'nodes', 'V']:
                if hasattr(dag, attr):
                    vertices = list(getattr(dag, attr))
                    break
                elif isinstance(dag, dict) and attr in dag:
                    vertices = list(dag[attr])
                    break
            if vertices is None:
                raise ValueError("Cannot extract vertices")
        
        n = len(vertices)
        
        # Extract parents
        parents = {}
        if hasattr(dag, 'parents'):
            parents = dag.parents
        elif isinstance(dag, dict) and 'parents' in dag:
            parents = dag['parents']
        
        if not isinstance(parents, dict):
            if isinstance(parents, (list, tuple)):
                parents = {v: list(parents[i]) for i, v in enumerate(vertices)}
            else:
                parents = {v: [] for v in vertices}
        
        for v in vertices:
            if v not in parents:
                parents[v] = []
        
        # Extract inputs and outputs
        inputs = set()
        if hasattr(dag, 'inputs'):
            inputs = set(dag.inputs)
        elif isinstance(dag, dict) and 'inputs' in dag:
            inputs = set(dag['inputs'])
        else:
            inputs = set(v for v in vertices if not parents[v])
        
        outputs = set()
        if hasattr(dag, 'outputs'):
            outputs = set(dag.outputs)
        elif isinstance(dag, dict) and 'outputs' in dag:
            outputs = set(dag['outputs'])
        else:
            children = defaultdict(set)
            for v in vertices:
                for p in parents[v]:
                    children[p].add(v)
            outputs = set(v for v in vertices if not children[v])
        
        # Build children map
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents[v]:
                children_map[p].append(v)
        
        # --- 2. Compute depth/layer for each vertex ---
        depth = {}
        for v in vertices:
            if not parents[v]:
                depth[v] = 0
            else:
                depth[v] = max(depth.get(p, 0) + 1 for p in parents[v])
        
        max_depth = max(depth.values()) if depth else 0
        layers = defaultdict(list)
        for v in vertices:
            layers[depth[v]].append(v)
        
        # --- 3. Precompute: for each vertex, which future layers need it as parent ---
        # needed_in_layer[v][d] = True if v is a parent of any vertex in layer d
        needed_in_layer = defaultdict(lambda: defaultdict(bool))
        for v in vertices:
            for c in children_map[v]:
                needed_in_layer[v][depth[c]] = True
        
        # --- 4. Main pebbling algorithm ---
        ops = []
        in_fast = set()
        
        # Track which vertices have been computed
        computed = set()
        
        # For each layer, determine which parents to keep
        # Process layers from 0 to max_depth
        for d in range(max_depth + 1):
            if time.time() > DEADLINE:
                break
            
            layer_vertices = sorted(layers.get(d, []))
            
            # Determine which parents from previous layers are needed for this layer
            # and which can be evicted
            parents_needed_this_layer = set()
            for v in layer_vertices:
                for p in parents[v]:
                    parents_needed_this_layer.add(p)
            
            # Also track which parents will be needed in future layers
            parents_needed_future = set()
            for future_d in range(d + 1, max_depth + 1):
                for v in layers.get(future_d, []):
                    for p in parents[v]:
                        parents_needed_future.add(p)
            
            # Decide which parents to keep in fast memory
            # Keep: parents_needed_this_layer ∪ (parents_needed_future if budget allows)
            # Evict: parents not needed in this layer or future layers
            
            # First, ensure all parents for this layer are in fast memory
            for v in layer_vertices:
                for p in parents[v]:
                    if p not in in_fast:
                        if p in inputs:
                            ops.append(['load', p])
                            in_fast.add(p)
                        else:
                            # p was computed earlier but evicted - need to reload or recompute
                            # For simplicity, load it (recomputation would require redoing work)
                            ops.append(['load', p])
                            in_fast.add(p)
            
            # Now compute all vertices in this layer
            for v in layer_vertices:
                if v in inputs:
                    # Input vertex - already loaded or load now
                    if v not in in_fast:
                        ops.append(['load', v])
                        in_fast.add(v)
                    continue
                
                # Compute v
                ops.append(['compute', v])
                in_fast.add(v)
                computed.add(v)
            
            # After computing this layer, decide what to evict
            # Keep vertices that are:
            # 1. Needed as parents for future layers
            # 2. Outputs (must be stored at the end)
            
            # Determine which vertices in current fast memory to keep
            vertices_to_keep = set()
            
            # Keep parents needed for future layers
            for v in list(in_fast):
                if v in parents_needed_future:
                    vertices_to_keep.add(v)
            
            # Keep outputs
            for v in list(in_fast):
                if v in outputs:
                    vertices_to_keep.add(v)
            
            # Evict vertices not in keep set (but only if budget is tight)
            # If we have room, keep more to avoid future reloads
            current_size = len(in_fast)
            num_to_evict = max(0, current_size - fast_memory_budget)
            
            if num_to_evict > 0:
                # Evict vertices not in keep set, prioritizing those not needed future
                evictable = [v for v in in_fast if v not in vertices_to_keep]
                # Sort by: not needed future first, then by depth (evict deeper first? or shallower?)
                # Evict vertices that are least likely to be needed soon
                evictable.sort(key=lambda v: (v in parents_needed_future, depth.get(v, 0)))
                
                for v in evictable[:num_to_evict]:
                    if v in in_fast:
                        ops.append(['delete', v])
                        in_fast.remove(v)
            
            # If still over budget, evict from keep set (least important first)
            current_size = len(in_fast)
            if current_size > fast_memory_budget:
                # Evict from vertices_to_keep, prioritizing those not outputs
                keep_evictable = [v for v in in_fast if v in vertices_to_keep]
                keep_evictable.sort(key=lambda v: (v in outputs, -depth.get(v, 0)))
                for v in keep_evictable:
                    if len(in_fast) <= fast_memory_budget:
                        break
                    if v in in_fast:
                        ops.append(['delete', v])
                        in_fast.remove(v)
        
        # --- 5. Store all outputs ---
        for v in sorted(outputs):
            if v in in_fast:
                ops.append(['store', v])
                in_fast.remove(v)
            elif v in computed:
                # Output was evicted - need to reload
                ops.append(['load', v])
                in_fast.add(v)
                ops.append(['store', v])
                in_fast.remove(v)
            else:
                # Output was never computed - this is an error
                # Try to compute it now (shouldn't happen in valid DAG)
                for p in parents[v]:
                    if p not in in_fast:
                        ops.append(['load', p])
                        in_fast.add(p)
                ops.append(['compute', v])
                in_fast.add(v)
                ops.append(['store', v])
                in_fast.remove(v)
        
        # Validate: check we didn't exceed operation limit
        if len(ops) > 20000:
            return get_baseline()
        
        return ops
    
    except Exception as e:
        return get_baseline()

# EVOLVE-BLOCK-END
