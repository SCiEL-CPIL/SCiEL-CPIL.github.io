"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize communication while respecting fast memory budget.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        edges = dag.get('edges', dag.get('dependencies', []))
        inputs = dag.get('inputs', dag.get('input_vertices', []))
        outputs = dag.get('outputs', dag.get('output_vertices', []))
    else:
        vertices = list(dag) if not isinstance(dag, (list, tuple)) else dag
        edges = []
        inputs = []
        outputs = []
    
    # Build graph
    adj = defaultdict(list)  # parent -> children
    in_edges = defaultdict(list)  # child -> parents
    all_vertices = set()
    
    for v in vertices:
        if isinstance(v, dict):
            vid = v.get('id', v.get('name', v))
        else:
            vid = v
        all_vertices.add(vid)
    
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            p, c = e
            if isinstance(p, dict):
                p = p.get('id', p.get('name', p))
            if isinstance(c, dict):
                c = c.get('id', c.get('name', c))
            adj[p].append(c)
            in_edges[c].append(p)
            all_vertices.add(p)
            all_vertices.add(c)
        elif isinstance(e, dict):
            p = e.get('from', e.get('parent', e.get('src')))
            c = e.get('to', e.get('child', e.get('dst')))
            if p is not None and c is not None:
                if isinstance(p, dict):
                    p = p.get('id', p.get('name', p))
                if isinstance(c, dict):
                    c = c.get('id', c.get('name', c))
                adj[p].append(c)
                in_edges[c].append(p)
                all_vertices.add(p)
                all_vertices.add(c)
    
    # If no edges, try to infer from vertex structure
    if not edges and vertices:
        for v in vertices:
            if isinstance(v, dict):
                vid = v.get('id', v.get('name', v))
                deps = v.get('deps', v.get('parents', v.get('dependencies', [])))
                if deps:
                    for d in deps:
                        if isinstance(d, dict):
                            d = d.get('id', d.get('name', d))
                        adj[d].append(vid)
                        in_edges[vid].append(d)
                        all_vertices.add(d)
    
    # Topological sort
    in_degree = {v: len(in_edges[v]) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in adj[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) < len(all_vertices):
        topo_order.extend([v for v in all_vertices if v not in set(topo_order)])
    
    # Identify inputs and outputs
    input_set = set()
    output_set = set()
    for v in inputs:
        if isinstance(v, dict):
            input_set.add(v.get('id', v.get('name', v)))
        else:
            input_set.add(v)
    for v in outputs:
        if isinstance(v, dict):
            output_set.add(v.get('id', v.get('name', v)))
        else:
            output_set.add(v)
    
    if not input_set:
        input_set = set(v for v in all_vertices if in_degree[v] == 0)
    if not output_set:
        output_set = set(v for v in all_vertices if not adj[v])
    
    # For each vertex, compute when it's last needed
    # last_needed[v] = the position in topo_order of the last child of v, or len(topo_order) if v is an output with no children
    last_needed = {}
    for i, v in enumerate(topo_order):
        children = adj[v]
        if children:
            max_child_pos = max(topo_order.index(c) for c in children if c in topo_order)
            last_needed[v] = max_child_pos
        else:
            if v in output_set:
                last_needed[v] = len(topo_order)
            else:
                last_needed[v] = i
    
    # Process in topological order
    fast_mem = set()  # vertices currently in fast memory (blue)
    computed = set()  # vertices that have been computed
    operations = []
    
    # Inputs start blue
    for v in input_set:
        if v in all_vertices:
            fast_mem.add(v)
            computed.add(v)
    
    for i, v in enumerate(topo_order):
        if time.time() - start_time > DEADLINE:
            break
        
        # If v is an input, it's already in fast_mem
        if v in input_set:
            continue
        
        # Ensure all parents are in fast memory
        for p in in_edges[v]:
            if p not in fast_mem:
                # Need to load p
                if len(fast_mem) >= fast_memory_budget:
                    # Evict the one needed farthest in the future
                    best_evict = None
                    best_ln = -1
                    for mem_v in fast_mem:
                        ln = last_needed.get(mem_v, 0)
                        if ln > best_ln:
                            best_ln = ln
                            best_evict = mem_v
                    if best_evict is not None:
                        operations.append(['store', best_evict])
                        fast_mem.discard(best_evict)
                
                operations.append(['load', p])
                fast_mem.add(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        
        # Decide whether to keep v in fast memory
        v_last_needed = last_needed.get(v, i)
        will_be_needed = v_last_needed > i or (v in output_set and i < len(topo_order) - 1)
        
        if will_be_needed:
            if len(fast_mem) < fast_memory_budget:
                fast_mem.add(v)
            else:
                # Consider evicting to keep v
                best_evict = None
                best_ln = -1
                for mem_v in fast_mem:
                    ln = last_needed.get(mem_v, 0)
                    if ln > best_ln:
                        best_ln = ln
                        best_evict = mem_v
                
                if best_evict is not None and v_last_needed < best_ln:
                    operations.append(['store', best_evict])
                    fast_mem.discard(best_evict)
                    fast_mem.add(v)
        
        # If v is an output and this is the last time we need it, keep it
        # Actually, outputs must finish blue, so they should remain in fast_mem
    
    # Ensure all outputs are in fast memory at the end
    for v in output_set:
        if v in computed and v not in fast_mem:
            # Need to load it back
            if len(fast_mem) >= fast_memory_budget:
                best_evict = None
                best_ln = -1
                for mem_v in fast_mem:
                    if mem_v not in output_set:  # Don't evict other outputs
                        ln = last_needed.get(mem_v, 0)
                        if ln > best_ln:
                            best_ln = ln
                            best_evict = mem_v
                if best_evict is not None:
                    operations.append(['store', best_evict])
                    fast_mem.discard(best_evict)
            operations.append(['load', v])
            fast_mem.add(v)
    
    return operations
# EVOLVE-BLOCK-END
