"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque, defaultdict
import math

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
    next_uses = {v: [] for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents_map.get(v, []):
            next_uses[p].append(i)
    
    # Track current use index for each vertex
    use_index = {v: 0 for v in vertices}
    
    # Compute depth
    depth = {v: 0 for v in vertices}
    for v in topo_order:
        ps = parents_map.get(v, [])
        if ps:
            depth[v] = max(depth[p] for p in ps) + 1
    
    def get_next_use_time(v):
        idx = use_index[v]
        uses = next_uses[v]
        if idx < len(uses):
            return uses[idx]
        return float('inf')
    
    def advance_use(v):
        use_index[v] += 1
    
    # Compute the number of times each vertex will be needed (total uses)
    total_uses = {v: len(next_uses[v]) for v in vertices}
    
    # Compute recomputation cost: the number of vertices that would need to be recomputed
    # if we evict v. This is the size of the subtree rooted at v (in terms of vertices that
    # depend on v and are not yet computed).
    def compute_recomputation_costs():
        """Compute recomputation costs using dynamic programming."""
        # For each vertex, compute the number of descendants that depend on it
        # and haven't been computed yet
        recompute_cost = {v: 0 for v in vertices}
        
        # Process in reverse topological order
        for v in reversed(topo_order):
            ps = parents_map.get(v, [])
            if not ps:
                recompute_cost[v] = 1  # Input: cost is just loading
            else:
                # Cost to recompute v = 1 (compute v) + sum of costs to recompute parents
                # But we need to be careful: if a parent is shared, we count it once
                # For simplicity, use the max cost path (conservative)
                max_parent_cost = 0
                for p in ps:
                    max_parent_cost = max(max_parent_cost, recompute_cost.get(p, 0))
                recompute_cost[v] = 1 + max_parent_cost
        
        return recompute_cost
    
    recomputation_cost = compute_recomputation_costs()
    
    # Fast memory state
    fast_mem = {}  # vertex -> True
    operations = []
    op_count = 0
    MAX_OPS = 20000
    
    def eviction_priority(v, current_pos):
        """Compute eviction priority. Higher priority = more important to keep."""
        next_t = get_next_use_time(v)
        if next_t == float('inf'):
            # No more uses - safe to evict
            return -1000000
        
        # Distance to next use
        dist = max(1, next_t - current_pos)
        
        # Cost if we have to recompute (higher cost = more important to keep)
        recompute_cost = recomputation_cost.get(v, 0)
        
        # Priority: we want to keep vertices that are:
        # 1. Used soon (small dist)
        # 2. Expensive to recompute (high recompute_cost)
        # Use a harmonic-like combination
        priority = recompute_cost * 10.0 / (dist + 1.0)
        
        # Bonus for vertices that are used very soon
        if dist <= 2:
            priority += 50.0
        
        # Bonus for outputs (they must be stored at the end)
        if v in output_set:
            priority += 20.0
        
        return priority
    
    def evict_worst(current_pos):
        """Evict the vertex with the lowest priority."""
        nonlocal op_count
        if not fast_mem:
            return None
        worst_v = None
        worst_priority = float('inf')
        for v in list(fast_mem.keys()):
            p = eviction_priority(v, current_pos)
            if p < worst_priority:
                worst_priority = p
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
            next_t = get_next_use_time(v)
            if next_t != float('inf') and (next_t - current_pos) <= max(fast_memory_budget, 2):
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
