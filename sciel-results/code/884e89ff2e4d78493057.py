"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using a robust greedy topological approach.
    """
    deadline = time.time() + 50  # Safety margin
    
    # 1. Robust Parsing
    vertices = set()
    edges = []
    inputs = set()
    outputs = set()
    
    if isinstance(dag, dict):
        # Handle explicit keys
        if 'vertices' in dag:
            vertices = set(dag['vertices'])
        if 'edges' in dag:
            edges = dag['edges']
            for e in edges:
                if len(e) == 2:
                    vertices.add(e[0])
                    vertices.add(e[1])
        if 'inputs' in dag:
            inputs = set(dag['inputs'])
        if 'outputs' in dag:
            outputs = set(dag['outputs'])
        if 'parents' in dag:
            # parents: {child: [parents]} or {parent: [children]}? 
            # Contract says "topologically ordered parents". 
            # Usually "parents" map is child -> list of parents.
            p_map = dag['parents']
            for child, p_list in p_map.items():
                vertices.add(child)
                for p in p_list:
                    vertices.add(p)
                    edges.append((p, child))
            # If edges was empty, we built it from parents
            if 'edges' not in dag or not edges:
                 # We already added to edges above if we used this block
                 pass
        elif 'children' in dag:
            c_map = dag['children']
            for parent, c_list in c_map.items():
                vertices.add(parent)
                for c in c_list:
                    vertices.add(c)
                    edges.append((parent, c))
            if 'edges' not in dag or not edges:
                 pass
        
        # If no inputs/outputs specified, infer
        if not inputs:
            parents_map = defaultdict(set)
            for p, c in edges:
                parents_map[c].add(p)
            inputs = {v for v in vertices if v not in parents_map}
        if not outputs:
            children_map = defaultdict(set)
            for p, c in edges:
                children_map[p].add(c)
            outputs = {v for v in vertices if v not in children_map}

    elif isinstance(dag, list):
        # Heuristic: List of lists or list of tuples
        # Check if list of tuples (edges)
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in dag):
            edges = dag
            for p, c in edges:
                vertices.add(p)
                vertices.add(c)
        else:
            # Assume list of parent lists: dag[i] is list of parents for vertex i
            # OR dag[i] is list of children for vertex i?
            # "topologically ordered parents" suggests index i corresponds to vertex i?
            # Let's assume vertex IDs are 0..N-1
            n = len(dag)
            vertices = set(range(n))
            # Check if it looks like parent lists (values are smaller than index?)
            is_parent_list = True
            for i, lst in enumerate(dag):
                if any(p > i for p in lst):
                    is_parent_list = False
                    break
            if is_parent_list:
                # dag[i] contains parents of i
                for i, p_list in enumerate(dag):
                    for p in p_list:
                        edges.append((p, i))
            else:
                # Assume dag[i] contains children of i
                for i, c_list in enumerate(dag):
                    for c in c_list:
                        edges.append((i, c))
        
        # Infer inputs/outputs
        parents_map = defaultdict(set)
        children_map = defaultdict(set)
        for p, c in edges:
            parents_map[c].add(p)
            children_map[p].add(c)
        inputs = {v for v in vertices if v not in parents_map}
        outputs = {v for v in vertices if v not in children_map}
    else:
        # Fallback: Empty
        return []

    # Ensure vertices is a set
    vertices = set(vertices)
    
    # Build adjacency structures
    children = defaultdict(list)
    parents = defaultdict(list)
    for p, c in edges:
        children[p].append(c)
        parents[c].append(p)
        
    # Ensure all vertices have entries
    for v in vertices:
        if v not in children: children[v] = []
        if v not in parents: parents[v] = []

    # Topological Sort (Kahn's Algorithm)
    in_degree = {v: len(parents[v]) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for c in children[u]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(vertices):
        # Cycle detected or error, return empty or fallback
        # For now, return empty to avoid crash
        return []

    # Pre-compute "next use" index for each vertex
    # next_use[v] = the index in topo_order of the next vertex that depends on v
    # We need this to decide what to evict.
    # To be efficient, we can just check children during the loop.
    
    in_memory = set(inputs)
    operations = []
    
    # Map vertex to its index in topo_order for quick lookup
    topo_index = {v: i for i, v in enumerate(topo_order)}
    
    # To find "next use", we can pre-compute for each vertex v, the list of children indices.
    # child_indices[v] = sorted list of topo_index[c] for c in children[v]
    child_indices = {v: [] for v in vertices}
    for v in vertices:
        for c in children[v]:
            child_indices[v].append(topo_index[c])
        child_indices[v].sort()
        
    # current_child_ptr[v] = pointer to the next child in child_indices[v] that is > current topo index
    # We can just use a pointer that advances as we move through topo_order.
    # Actually, simpler: for each vertex v in memory, check if any child has topo_index > current_i.
    # If we evict v, we will have to load it again when we reach its next child.
    # Cost of eviction = 0 (store) + 1 (load later).
    # We want to evict the vertex that minimizes future loads.
    # So evict the vertex with the *furthest* next use (or no next use).
    
    for i, v in enumerate(topo_order):
        if time.time() > deadline:
            break
            
        # If v is an input, it's already in memory (or should be)
        # If v is not an input, we need to compute it
        if v not in inputs:
            # Ensure parents are in memory
            for p in parents[v]:
                if p not in in_memory:
                    operations.append(['load', p])
                    in_memory.add(p)
                    
            # Compute v
            operations.append(['compute', v])
            in_memory.add(v)
            
        # Now, manage memory
        # If over budget, evict
        while len(in_memory) > fast_memory_budget:
            # Find victim
            # Candidates: vertices in in_memory that are NOT outputs
            # We prefer to evict vertices that are not needed soon.
            # next_use = min(topo_index[c] for c in children[w] if topo_index[c] > i)
            # If no such c, next_use = infinity
            
            best_victim = None
            best_next_use = -1 # We want to maximize next_use (furthest)
            
            for w in in_memory:
                if w in outputs:
                    continue
                # Find next child of w after index i
                next_use = float('inf')
                for c in children[w]:
                    idx = topo_index[c]
                    if idx > i:
                        if idx < next_use:
                            next_use = idx
                        break # Since children are not sorted by index necessarily, we can't break. 
                        # Wait, child_indices[w] is sorted. Let's use that.
                
                # Using child_indices
                # child_indices[w] is sorted list of indices of children.
                # We need the first index > i.
                # Linear scan is O(degree). Total O(V*E) worst case? 
                # V=20000, E might be larger. 20000 * 20000 is too slow.
                # Optimize: Use bisect.
                
                # Let's just use a simple check for now, assuming sparse graph.
                # If performance is issue, we can pre-compute.
                
                # Re-doing with bisect on child_indices[w]
                import bisect
                ci = child_indices[w]
                pos = bisect.bisect_right(ci, i)
                if pos < len(ci):
                    next_use = ci[pos]
                else:
                    next_use = float('inf')
                    
                if next_use > best_next_use:
                    best_next_use = next_use
                    best_victim = w
                    
            if best_victim is None:
                # All in memory are outputs? We can't evict outputs.
                # This implies budget is too small to hold all outputs + current working set.
                # In this case, we might have to evict an output and reload it later?
                # But "Outputs must finish blue".
                # If we evict an output, we must load it back before end.
                # So we can evict an output if we plan to reload it.
                # But we don't know if we will.
                # Safest: If no non-output, evict any output with highest next_use (which is inf, so all are inf).
                # Just pick the first one.
                best_victim = next(iter(in_memory))
                
            operations.append(['store', best_victim])
            in_memory.discard(best_victim)

    # Final check: Ensure all outputs are in memory
    for v in outputs:
        if v not in in_memory:
            operations.append(['load', v])
            in_memory.add(v)
            
    return operations
# EVOLVE-BLOCK-END
