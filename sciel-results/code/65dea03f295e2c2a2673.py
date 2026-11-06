"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

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
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            # Last resort: return empty operations (will likely fail verification but avoid crash)
            return []
    
    # Build children map
    children_map = {}
    for v, ps in parents_map.items():
        for p in ps:
            children_map.setdefault(p, []).append(v)
    
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
    
    # Precompute all uses: for each vertex, list of positions in topo_order where it's needed
    all_uses = {v: [] for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents_map.get(v, []):
            all_uses[p].append(i)
    
    # Precompute total remaining uses for each vertex
    total_uses = {v: len(all_uses[v]) for v in vertices}
    
    # Track current use index for each vertex
    use_idx = {v: 0 for v in vertices}
    
    def get_next_use(v):
        idx = use_idx[v]
        if idx < len(all_uses[v]):
            return all_uses[v][idx]
        return float('inf')
    
    def get_remaining_uses(v):
        return total_uses[v] - use_idx[v]
    
    def eviction_score(v):
        """
        Score for eviction: higher score = more likely to be evicted.
        We want to evict vertices that are:
        1. Not needed for a long time (far next use)
        2. Have few remaining uses (low future demand)
        
        Combined heuristic: normalize next_use distance and remaining uses.
        """
        nu = get_next_use(v)
        rem = get_remaining_uses(v)
        
        # If no more uses, definitely evict
        if nu == float('inf'):
            return float('inf')
        
        # Current position
        cur_pos = current_pos[0]
        distance = nu - cur_pos
        
        # Higher distance and lower remaining uses = higher eviction score
        # Weight: distance matters more, but remaining uses also important
        # Use a combined metric: distance weighted by inverse of remaining uses
        # This penalizes evicting values that will be used many times soon
        if rem > 0:
            score = distance * 1.0 + (1.0 / rem) * 100
        else:
            score = distance * 1.0 + 10000
        
        return score
    
    # Global tracking of current position in topo_order
    current_pos = [0]
    
    fast_mem = {}
    operations = []
    op_count = 0
    MAX_OPS = 20000
    
    def evict_one():
        """Evict the vertex with the highest eviction score."""
        if not fast_mem:
            return None
        best_v = None
        best_score = -1
        for v in list(fast_mem.keys()):
            s = eviction_score(v)
            if s > best_score:
                best_score = s
                best_v = v
        if best_v is not None:
            del fast_mem[best_v]
            operations.append(['store', best_v])
            op_count += 1
            return best_v
        return None
    
    def ensure_space():
        """Ensure we have at least one free slot in fast memory."""
        while len(fast_mem) >= fast_memory_budget:
            evict_one()
    
    # Process vertices in topological order
    for i, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        if op_count >= MAX_OPS:
            break
        
        current_pos[0] = i
        
        if v in input_set:
            # Input: load if not in memory
            if v not in fast_mem:
                ensure_space()
                fast_mem[v] = True
                operations.append(['load', v])
                op_count += 1
            continue
        
        # Non-input: compute
        # Ensure all parents are in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                ensure_space()
                fast_mem[p] = True
                operations.append(['load', p])
                op_count += 1
            # Advance use index for parent
            use_idx[p] += 1
        
        # Compute v
        operations.append(['compute', v])
        op_count += 1
        
        # v is now in fast memory; ensure space if needed
        if v not in fast_mem:
            ensure_space()
            fast_mem[v] = True
        else:
            # Already in memory (shouldn't happen for non-inputs in topo order, but handle)
            pass
        
        # If v is no longer needed as a parent for any future vertex, store it to free space
        # This is safe if v is an output (it needs to finish blue) or if we don't need it anymore
        rem_uses = get_remaining_uses(v)
        if rem_uses == 0:
            # v is no longer needed; store it to free a slot
            if v in fast_mem:
                operations.append(['store', v])
                op_count += 1
                del fast_mem[v]
    
    # Final: ensure all outputs are blue (stored)
    for v in outputs:
        if time.time() > deadline:
            break
        if op_count >= MAX_OPS:
            break
        
        if v in fast_mem:
            operations.append(['store', v])
            op_count += 1
            del fast_mem[v]
        elif v not in output_set:
            # This shouldn't happen, but if an output wasn't computed, we need to handle it
            pass
    
    # Optionally clean up remaining fast memory (not required, but can help)
    # We don't need to delete everything; just ensure outputs are blue.
    
    if op_count > MAX_OPS:
        operations = operations[:MAX_OPS]
    
    return operations

# EVOLVE-BLOCK-END
