# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Uses a greedy strategy with lookahead-based eviction to minimize memory transfers.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Parse DAG - handle multiple possible formats
    vertices = []
    edges = []
    input_verts = set()
    output_verts = set()
    
    if isinstance(dag, dict):
        # Try various key combinations
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
        elif 'nodes' in dag:
            vertices = list(dag['nodes'])
        
        if 'edges' in dag:
            edges = list(dag['edges'])
        
        if 'inputs' in dag:
            input_verts = set(dag['inputs'])
        if 'outputs' in dag:
            output_verts = set(dag['outputs'])
        
        # If no explicit inputs/outputs, derive from structure
        if not input_verts or not output_verts:
            has_parent = set()
            has_child = set()
            for p, c in edges:
                has_parent.add(c)
                has_child.add(p)
            if not input_verts:
                input_verts = set(v for v in vertices if v not in has_parent)
            if not output_verts:
                output_verts = set(v for v in vertices if v not in has_child)
    
    elif isinstance(dag, list):
        # Could be list of (vertex, parents) or list of dicts
        for i, item in enumerate(dag):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v = item[0]
                parents = item[1]
                if v not in vertices:
                    vertices.append(v)
                for p in parents:
                    edges.append((p, v))
                    if p not in vertices:
                        vertices.append(p)
            elif isinstance(item, dict):
                v = item.get('vertex', item.get('id', i))
                parents = item.get('parents', item.get('inputs', []))
                if v not in vertices:
                    vertices.append(v)
                for p in parents:
                    edges.append((p, v))
                    if p not in vertices:
                        vertices.append(p)
        
        # Derive inputs and outputs
        has_parent = set()
        has_child = set()
        for p, c in edges:
            has_parent.add(c)
            has_child.add(p)
        input_verts = set(v for v in vertices if v not in has_parent)
        output_verts = set(v for v in vertices if v not in has_child)
    
    else:
        # Unknown format - fall back to baseline
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Build adjacency lists
    adj = defaultdict(list)
    radj = defaultdict(list)
    for p, c in edges:
        adj[p].append(c)
        radj[c].append(p)
    
    # Topological sort
    in_degree = {v: len(radj.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in adj.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Precompute: for each vertex, the list of topo positions where it's needed as a parent
    vertex_use_positions = defaultdict(list)
    for idx, v in enumerate(topo_order):
        for p in radj.get(v, []):
            vertex_use_positions[p].append(idx)
    
    # Simulation
    in_memory = set()
    operations = []
    
    # Initialize: inputs start blue (in fast memory)
    for v in input_verts:
        if v in vertices:
            in_memory.add(v)
    
    def get_next_use(v, current_idx):
        """Get the next position where vertex v is needed after current_idx."""
        for pos in vertex_use_positions.get(v, []):
            if pos > current_idx:
                return pos
        return float('inf')
    
    def evict_one(current_idx):
        """Evict one vertex from memory, choosing the one with farthest next use."""
        best_evict = None
        best_next_use = -1
        for mem_v in in_memory:
            if mem_v in output_verts:
                continue  # Avoid evicting outputs
            next_use = get_next_use(mem_v, current_idx)
            if next_use > best_next_use:
                best_next_use = next_use
                best_evict = mem_v
        
        if best_evict is None:
            # Must evict an output
            best_evict = next(iter(in_memory))
        
        in_memory.remove(best_evict)
        operations.append(['delete', best_evict])
    
    # Process in topological order
    for idx, v in enumerate(topo_order):
        if time.time() > deadline - 5.0:
            break
        
        if v in input_verts:
            continue
        
        if v in in_memory:
            continue
        
        # Need to compute v - ensure all parents are in memory
        parents = radj.get(v, [])
        for p in parents:
            if p not in in_memory:
                if len(in_memory) >= fast_memory_budget:
                    evict_one(idx)
                in_memory.add(p)
                operations.append(['load', p])
        
        # Make room for v if needed
        if v not in in_memory and len(in_memory) >= fast_memory_budget:
            evict_one(idx)
        
        in_memory.add(v)
        operations.append(['compute', v])
    
    # Ensure all outputs are in memory
    for v in output_verts:
        if v not in in_memory:
            if len(in_memory) >= fast_memory_budget:
                # Evict a non-output
                non_outputs = [mem_v for mem_v in in_memory if mem_v not in output_verts]
                if non_outputs:
                    evict_v = non_outputs[0]
                    in_memory.remove(evict_v)
                    operations.append(['delete', evict_v])
            in_memory.add(v)
            operations.append(['load', v])
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
