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
    
    # Precompute recomputation cost (number of descendants that need this vertex)
    # Using a simple metric: number of times vertex is used as a parent
    recomputation_cost = {v: len(next_uses[v]) for v in vertices}
    
    # Compute depth for logarithmic weighting
    depth = {v: 0 for v in vertices}
    for v in topo_order:
        ps = parents_map.get(v, [])
        if ps:
            depth[v] = max(depth[p] for p in ps) + 1
    
    # Weight for eviction priority: higher means more important to keep
    # Combines next-use distance (closer = more important) with recomputation cost
    def get_next_use_time(v):
        idx = use_index[v]
        uses = next_uses[v]
        if idx < len(uses):
            return uses[idx]
        return float('inf')
    
    def advance_use(v):
        use_index[v] += 1
    
    # Precompute recomputation cost more carefully: number of descendants that transitively depend on v
    # and how many times those descendants will be computed
    def compute_recomputation_cost():
        """Compute the cost of recomputing vertex v if evicted."""
        cost = {v: 0 for v in vertices}
        # Process in reverse topological order
        for v in reversed(topo_order):
            ps = parents_map.get(v, [])
            if not ps:
                cost[v] = 0
            else:
                # Cost to recompute v = 1 (compute v itself) + sum of costs to recompute parents
                # But only if parents are not cached
                # Approximate: 1 + sum of parent costs weighted by probability of eviction
                cost[v] = 1
                for p in ps:
                    cost[v] += cost.get(p, 0) * 0.5  # discount for potential caching
        return cost
    
    recomputation_cost = compute_recomputation_cost()
    
    # Lookahead: for each vertex, compute how many steps until next use
    # and the number of vertices that will be computed in between
    def compute_lookahead():
        """Compute lookahead information for better eviction decisions."""
        lookahead = {}
        for v in vertices:
            next_t = get_next_use_time(v)
            if next_t == float('inf'):
                lookahead[v] = (float('inf'), 0)
            else:
                # Count vertices between current position and next use
                current_pos = topo_order.index(v) if v in topo_order else 0
                steps = max(0, next_t - current_pos)
                lookahead[v] = (next_t, steps)
        return lookahead
    
    # Eviction weight: lower = more likely to be evicted
    def eviction_weight(v, current_pos):
        """Compute eviction weight. Lower weight = more likely to be evicted."""
        next_t = get_next_use_time(v)
        if next_t == float('inf'):
            # No more uses - safe to evict
            return -1000000 - recomputation_cost.get(v, 0)
        
        # Distance to next use
        dist = max(1, next_t - current_pos)
        
        # Cost if we have to recompute (higher cost = more important to keep)
        recompute_cost = recomputation_cost.get(v, 0)
        
        # Weight: we want to keep vertices that are:
        # 1. Used soon (small dist)
        # 2. Expensive to recompute (high recompute_cost)
        # Combined: priority = recompute_cost / (dist + 1)
        # We want to EVICT low priority, so weight = -priority
        import math
        priority = recompute_cost / (dist + 1.0)
        # Add a small bonus for vertices with very next use
        if dist <= 2:
            priority += 100
        return -priority
    
    # Fast memory state
    fast_mem = {}  # vertex -> True
    operations = []
    op_count = 0
    MAX_OPS = 20000
    
    def evict_worst(current_pos):
        """Evict the vertex with the lowest priority (highest eviction weight)."""
        nonlocal op_count
        if not fast_mem:
            return None
        worst_v = None
        worst_weight = float('inf')
        for v in list(fast_mem.keys()):
            w = eviction_weight(v, current_pos)
            if w > worst_weight:
                worst_weight = w
                worst_v = v
        if worst_v is not None:
            del fast_mem[worst_v]
            operations.append(['store', worst_v])
            op_count += 1
        return worst_v
    
    def ensure_space(current_pos):
        """Ensure fast memory has at least one free slot."""
        while len(fast_mem) >= fast_memory_budget:
            evict_worst(current_pos)
    
    def load_vertex(v, current_pos):
        """Load vertex v into fast memory if not already there."""
        nonlocal op_count
        if v in fast_mem:
            return
        ensure_space(current_pos)
        fast_mem[v] = True
        operations.append(['load', v])
        op_count += 1
    
    # Process vertices in topological order
    current_pos = 0
    for v in topo_order:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
        
        current_pos = topo_order.index(v)
        
        if v in input_set:
            # Inputs start blue; load into fast memory when needed
            # Only load if it will be used soon
            next_t = get_next_use_time(v)
            if next_t != float('inf') and (next_t - current_pos) <= fast_memory_budget:
                load_vertex(v, current_pos)
            continue
        
        # Non-input vertex: compute it
        # Ensure all parents are in fast memory
        for p in parents_map.get(v, []):
            load_vertex(p, current_pos)
            advance_use(p)
        
        # Compute v
        operations.append(['compute', v])
        op_count += 1
        
        # v is now in fast memory
        fast_mem[v] = True
        
        # If full, evict one
        if len(fast_mem) > fast_memory_budget:
            evict_worst(current_pos)
    
    # At the end, ensure all outputs are blue (stored to disk)
    for v in outputs:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
        
        if v in fast_mem:
            operations.append(['store', v])
            op_count += 1
            del fast_mem[v]
        # If output is not in fast memory, it's already on disk (blue)
        # No action needed
    
    return operations[:MAX_OPS]

# EVOLVE-BLOCK-END
