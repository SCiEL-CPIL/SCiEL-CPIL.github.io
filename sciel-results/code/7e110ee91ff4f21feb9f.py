# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 75.0  # 75s safety margin
    
    # Parse DAG structure
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif hasattr(dag, 'nodes'):
        vertices = list(dag.nodes)
    elif isinstance(dag, dict) and 'vertices' in dag:
        vertices = list(dag['vertices'])
    elif isinstance(dag, dict) and 'nodes' in dag:
        vertices = list(dag['nodes'])
    elif isinstance(dag, list):
        vertices = list(range(len(dag)))
    else:
        vertices = list(range(len(dag)))
    
    # Get edges
    if hasattr(dag, 'edges'):
        edges = dag.edges
    elif isinstance(dag, dict) and 'edges' in dag:
        edges = dag['edges']
    elif isinstance(dag, dict) and 'parents' in dag:
        edges = dag['parents']
    else:
        edges = None
    
    # Get input/output vertices
    if hasattr(dag, 'input_vertices'):
        input_vertices = set(dag.input_vertices)
    elif isinstance(dag, dict) and 'input_vertices' in dag:
        input_vertices = set(dag['input_vertices'])
    elif isinstance(dag, dict) and 'inputs' in dag:
        input_vertices = set(dag['inputs'])
    else:
        input_vertices = set()
    
    if hasattr(dag, 'output_vertices'):
        output_vertices = set(dag.output_vertices)
    elif isinstance(dag, dict) and 'output_vertices' in dag:
        output_vertices = set(dag['output_vertices'])
    elif isinstance(dag, dict) and 'outputs' in dag:
        output_vertices = set(dag['outputs'])
    else:
        output_vertices = set()
    
    # Build parent and child maps
    parents_map = defaultdict(set)
    children_map = defaultdict(set)
    
    if edges is not None:
        if isinstance(edges, dict):
            for child, par_list in edges.items():
                for p in par_list:
                    parents_map[child].add(p)
                    children_map[p].add(child)
        elif isinstance(edges, list):
            for e in edges:
                if isinstance(e, (tuple, list)) and len(e) == 2:
                    p, c = e
                    parents_map[c].add(p)
                    children_map[p].add(c)
    
    # Infer inputs/outputs if not specified
    if not input_vertices:
        for v in vertices:
            if v not in parents_map or len(parents_map[v]) == 0:
                input_vertices.add(v)
    
    if not output_vertices:
        for v in vertices:
            if v not in children_map or len(children_map[v]) == 0:
                output_vertices.add(v)
    
    # Strategy: Use a greedy approach with lookahead
    # Process vertices in topological order
    # Keep track of what's in fast memory (red set)
    # When computing a vertex, load its parents if not already in fast memory
    # After computing, decide whether to keep the result in fast memory or store it
    
    # Topological sort
    in_degree = {v: len(parents_map.get(v, set())) for v in vertices}
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children_map.get(v, set()):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle or disconnected, add remaining
        remaining = set(vertices) - set(topo_order)
        topo_order.extend(list(remaining))
    
    # Fast memory state
    fast_mem = set()  # vertices currently in fast memory (red)
    computed = set()  # vertices that have been computed
    operations = []
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        # If v is an input, it's already available (blue)
        if v in input_vertices:
            # We need to load it to use it in computations
            if v not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    # Need to free space: delete something
                    # Delete a vertex that won't be needed again
                    # For now, delete the oldest non-output
                    for fv in list(fast_mem):
                        if fv not in output_vertices:
                            fast_mem.remove(fv)
                            operations.append(['delete', fv])
                            break
                fast_mem.add(v)
                operations.append(['load', v])
            computed.add(v)
            continue
        
        # For non-input vertices, we need to compute them
        # First, ensure all parents are in fast memory
        for p in parents_map.get(v, set()):
            if p not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    # Free space
                    for fv in list(fast_mem):
                        if fv not in output_vertices:
                            fast_mem.remove(fv)
                            operations.append(['delete', fv])
                            break
                fast_mem.add(p)
                operations.append(['load', p])
        
        # Compute v
        if len(fast_mem) >= fast_memory_budget and v not in fast_mem:
            # Need space for v
            for fv in list(fast_mem):
                if fv not in output_vertices and fv != v:
                    fast_mem.remove(fv)
                    operations.append(['delete', fv])
                    break
        
        fast_mem.add(v)
        computed.add(v)
        operations.append(['compute', v])
    
    # Now store all output vertices
    for v in output_vertices:
        if time.time() > deadline:
            break
        if v in fast_mem:
            fast_mem.remove(v)
            operations.append(['store', v])
        elif v in computed:
            # Need to reload and store
            if len(fast_mem) >= fast_memory_budget:
                for fv in list(fast_mem):
                    if fv not in output_vertices:
                        fast_mem.remove(fv)
                        operations.append(['delete', fv])
                        break
            fast_mem.add(v)
            operations.append(['load', v])
            fast_mem.remove(v)
            operations.append(['store', v])
    
    # Clean up remaining fast memory
    for v in list(fast_mem):
        if time.time() > deadline:
            break
        fast_mem.remove(v)
        operations.append(['delete', v])
    
    return operations
# EVOLVE-BLOCK-END
