"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Parse DAG structure
    def get_vertices(d):
        if hasattr(d, 'vertices'):
            return list(d.vertices)
        elif isinstance(d, dict) and 'vertices' in d:
            return list(d['vertices'])
        elif hasattr(d, 'nodes'):
            return list(d.nodes)
        elif isinstance(d, dict) and 'nodes' in d:
            return list(d['nodes'])
        else:
            if hasattr(d, 'parents'):
                parents = d.parents
            elif isinstance(d, dict) and 'parents' in d:
                parents = d['parents']
            else:
                raise ValueError("Cannot determine vertices from dag")
            verts = set()
            for v, ps in parents.items():
                verts.add(v)
                verts.update(ps)
            return list(verts)
    
    def get_parents(d):
        if hasattr(d, 'parents'):
            return dict(d.parents)
        elif isinstance(d, dict) and 'parents' in d:
            return dict(d['parents'])
        elif hasattr(d, 'edges'):
            parents = {}
            for e in d.edges:
                if len(e) == 2:
                    child, parent = e
                    parents.setdefault(child, []).append(parent)
            return parents
        else:
            raise ValueError("Cannot determine parents from dag")
    
    def get_inputs(d):
        if hasattr(d, 'inputs'):
            return list(d.inputs)
        elif isinstance(d, dict) and 'inputs' in d:
            return list(d['inputs'])
        else:
            parents = get_parents(d)
            return [v for v in get_vertices(d) if not parents.get(v, [])]
    
    def get_outputs(d):
        if hasattr(d, 'outputs'):
            return list(d.outputs)
        elif isinstance(d, dict) and 'outputs' in d:
            return list(d['outputs'])
        else:
            parents = get_parents(d)
            vertices = get_vertices(d)
            children = {}
            for v, ps in parents.items():
                for p in ps:
                    children.setdefault(p, []).append(v)
            return [v for v in vertices if not children.get(v, [])]
    
    try:
        vertices = get_vertices(dag)
        parents_map = get_parents(dag)
        inputs = get_inputs(dag)
        outputs = get_outputs(dag)
    except Exception:
        # Fallback to baseline if parsing fails
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    # Build children map
    children_map = defaultdict(list)
    for v, ps in parents_map.items():
        for p in ps:
            children_map[p].append(v)
    
    # Topological sort
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
    
    if len(topo_order) != len(vertices):
        topo_order = list(vertices)
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Precompute next uses for each vertex
    # next_uses[v] = list of indices in topo_order where v is needed as a parent
    next_uses = {v: [] for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents_map.get(v, []):
            next_uses[p].append(i)
    
    # Track current use index for each vertex
    use_index = {v: 0 for v in vertices}
    
    def get_next_use_time(v):
        idx = use_index[v]
        uses = next_uses[v]
        if idx < len(uses):
            return uses[idx]
        return float('inf')
    
    def advance_use(v):
        use_index[v] += 1
    
    # Fast memory state
    fast_mem = {}  # vertex -> True
    operations = []
    op_count = 0
    MAX_OPS = 20000
    
    def evict_farthest():
        """Evict the vertex with the farthest next use, or one with no more uses."""
        nonlocal op_count
        if not fast_mem:
            return None
        # First priority: evict vertices with no more uses
        no_use_v = None
        for v in list(fast_mem.keys()):
            if get_next_use_time(v) == float('inf'):
                no_use_v = v
                break
        if no_use_v is not None:
            del fast_mem[no_use_v]
            operations.append(['store', no_use_v])
            op_count += 1
            return no_use_v
        # Second priority: evict vertex with farthest next use
        worst_v = None
        worst_time = -1
        for v in list(fast_mem.keys()):
            t = get_next_use_time(v)
            if t > worst_time:
                worst_time = t
                worst_v = v
        if worst_v is not None:
            del fast_mem[worst_v]
            operations.append(['store', worst_v])
            op_count += 1
        return worst_v
    
    def ensure_space():
        """Ensure fast memory has at least one free slot."""
        while len(fast_mem) >= fast_memory_budget:
            evict_farthest()
    
    def load_vertex(v):
        """Load vertex v into fast memory if not already there."""
        nonlocal op_count
        if v in fast_mem:
            return
        ensure_space()
        fast_mem[v] = True
        operations.append(['load', v])
        op_count += 1
    
    # Process vertices in topological order
    for v in topo_order:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
        
        if v in input_set:
            # Inputs start blue; load into fast memory when needed
            # We don't load them proactively; we load them when a child needs them
            continue
        
        # Non-input vertex: compute it
        # Ensure all parents are in fast memory
        for p in parents_map.get(v, []):
            load_vertex(p)
            advance_use(p)
        
        # Compute v
        operations.append(['compute', v])
        op_count += 1
        
        # v is now in fast memory - add it first, then ensure space if needed
        fast_mem[v] = True
        
        # If we're over budget, evict (but prefer evicting other vertices, not v)
        if len(fast_mem) > fast_memory_budget:
            # Temporarily remove v to avoid evicting it
            was_in_mem = True
            del fast_mem[v]
            ensure_space()
            fast_mem[v] = True
            # If still over budget after re-adding v, we need to evict something else
            if len(fast_mem) > fast_memory_budget:
                evict_farthest()
        
        # If v is an output, we'll handle storing it at the end
    
    # At the end, ensure all outputs are blue (stored to disk)
    for v in outputs:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
        
        if v in fast_mem:
            operations.append(['store', v])
            op_count += 1
            del fast_mem[v]
        else:
            # Output was evicted - need to load it back to store it
            # But we can't load if we don't have the value... 
            # Actually, if it's an output, it must have been computed, so its value exists
            # We need to load it from slow memory
            if len(fast_mem) >= fast_memory_budget:
                evict_farthest()
            operations.append(['load', v])
            op_count += 1
            fast_mem[v] = True
            operations.append(['store', v])
            op_count += 1
            del fast_mem[v]
    
    return operations[:MAX_OPS]

# EVOLVE-BLOCK-END
