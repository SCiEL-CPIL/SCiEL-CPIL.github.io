from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimizes communication for DAG pebbling using a Greedy Eviction strategy
    based on Descendant Count and LRU.
    """
    
    # Robustly extract graph structure
    def get_attr(d, key):
        if isinstance(d, dict):
            return d.get(key, None)
        return getattr(d, key, None)

    parents = get_attr(dag, 'parents')
    children = get_attr(dag, 'children')
    inputs = get_attr(dag, 'inputs')
    outputs = get_attr(dag, 'outputs')
    
    if parents is None or children is None:
        # Fallback: try to find keys
        if isinstance(dag, dict):
            for k in dag.keys():
                if 'parent' in k.lower(): parents = dag[k]
                if 'child' in k.lower(): children = dag[k]
                if 'input' in k.lower(): inputs = dag[k]
                if 'output' in k.lower(): outputs = dag[k]
        if parents is None or children is None:
            raise ValueError("Could not find parents/children in dag")
            
    if inputs is None: inputs = []
    if outputs is None: outputs = []
    
    # Normalize to dicts with list values
    parents = {v: list(p) for v, p in parents.items()}
    children = {v: list(c) for v, c in children.items()}
    
    all_vertices = set(parents.keys()) | set(children.keys()) | set(inputs) | set(outputs)
    
    # Topological Sort
    in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for c in children.get(u, []):
            if c in in_degree: # Safety check
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
                
    if len(topo_order) != len(all_vertices):
        # If cycle or missing nodes, append remaining
        for v in all_vertices:
            if v not in topo_order:
                topo_order.append(v)

    # Precompute descendant counts
    desc_count = {v: 0 for v in all_vertices}
    for v in reversed(topo_order):
        for c in children.get(v, []):
            if c in desc_count:
                desc_count[v] += 1 + desc_count[c]
            
    # State
    blue_set = set(inputs)
    last_access = {}
    current_time = 0
    ops = []
    
    # Initialize last_access for inputs
    for i, v in enumerate(inputs):
        last_access[v] = i
    current_time = len(inputs)
    
    # If budget is 0, we can't hold anything. But task implies > 0.
    # If budget < len(inputs), we must evict some inputs immediately?
    # "Inputs start blue". If budget < len(inputs), the initial state is invalid?
    # Or we assume we can only hold `fast_memory_budget` inputs.
    # Let's assume we start with min(len(inputs), fast_memory_budget) inputs?
    # No, "Inputs start blue" is a condition.
    # If budget is too small, we might have to store some inputs right away?
    # But we can't store what isn't blue?
    # Let's assume the problem guarantees budget >= 1.
    # If len(inputs) > fast_memory_budget, we keep the first `fast_memory_budget` inputs?
    # This is an edge case. Let's just keep them all in blue_set logically,
    # but if we need to load something else, we evict.
    # However, if blue_set size > budget, we are in violation.
    # So, if len(inputs) > fast_memory_budget, we should store the excess inputs immediately?
    # But they are "start blue".
    # Let's assume the test cases have fast_memory_budget >= max fan-in or similar.
    # To be safe, if len(blue_set) > fast_memory_budget, evict LRU inputs until it fits.
    
    if len(blue_set) > fast_memory_budget:
        # Sort inputs by last_access (oldest first)
        to_evict = sorted(inputs, key=lambda x: last_access[x])
        for v in to_evict:
            if len(blue_set) <= fast_memory_budget:
                break
            ops.append(['store', v])
            blue_set.remove(v)
            # last_access[v] remains, but v is not in blue_set
            
    for v in topo_order:
        if v in blue_set:
            last_access[v] = current_time
            current_time += 1
            continue
            
        # v is not blue.
        # Ensure parents are blue
        for p in parents.get(v, []):
            if p not in blue_set:
                # Need to load p
                if len(blue_set) == fast_memory_budget:
                    # Evict
                    # Candidates: blue_set \ parents(v)
                    candidates = [x for x in blue_set if x not in parents.get(v, [])]
                    if not candidates:
                        candidates = list(blue_set)
                    
                    if candidates:
                        evict_candidate = min(candidates, key=lambda x: (desc_count.get(x, 0), last_access.get(x, -1)))
                        ops.append(['store', evict_candidate])
                        blue_set.remove(evict_candidate)
                
                ops.append(['load', p])
                blue_set.add(p)
                last_access[p] = current_time
                current_time += 1
        
        # Compute or Load v
        if v in inputs:
            ops.append(['load', v])
        else:
            ops.append(['compute', v])
            
        blue_set.add(v)
        last_access[v] = current_time
        current_time += 1
        
    return ops
