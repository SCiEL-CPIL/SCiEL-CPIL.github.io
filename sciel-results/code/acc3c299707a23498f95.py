"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def _extract_dag_info(dag):
    """
    Extract DAG structure. Handles common formats:
    - dag as dict with 'nodes', 'edges', 'inputs', 'outputs', 'layers'
    - dag as list of (node, parents) tuples
    - dag as list of dicts
    """
    if isinstance(dag, dict):
        nodes = dag.get('nodes', [])
        if not nodes and 'vertices' in dag:
            nodes = dag['vertices']
        edges = dag.get('edges', [])
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        parents = dag.get('parents', {})
        layers = dag.get('layers', [])
        
        # Build parent map if not provided
        if not parents:
            parents = {n: [] for n in nodes}
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    parents[e[1]].append(e[0])
        
        # Determine inputs/outputs if not provided
        if not inputs:
            inputs = [n for n in nodes if len(parents.get(n, [])) == 0]
        if not outputs:
            # Outputs are nodes with no children
            children = {n: set() for n in nodes}
            for n in nodes:
                for p in parents.get(n, []):
                    children[p].add(n)
            outputs = [n for n in nodes if len(children[n]) == 0]
        
        return nodes, parents, inputs, outputs, layers
    
    elif isinstance(dag, (list, tuple)):
        # List of (node, parents) or list of dicts
        nodes = []
        parents = {}
        inputs = []
        outputs = []
        layers = []
        
        for item in dag:
            if isinstance(item, dict):
                node = item.get('node', item.get('id', item.get('name')))
                par = item.get('parents', item.get('inputs', []))
                is_input = item.get('is_input', item.get('input', False))
                is_output = item.get('is_output', item.get('output', False))
                layer = item.get('layer', item.get('depth', None))
                nodes.append(node)
                parents[node] = list(par)
                if is_input:
                    inputs.append(node)
                if is_output:
                    outputs.append(node)
                if layer is not None:
                    layers.append((node, layer))
            elif isinstance(item, (list, tuple)):
                node = item[0]
                par = list(item[1]) if len(item) > 1 else []
                nodes.append(node)
                parents[node] = par
        
        # Determine inputs/outputs
        if not inputs:
            inputs = [n for n in nodes if len(parents.get(n, [])) == 0]
        if not outputs:
            children = {n: set() for n in nodes}
            for n in nodes:
                for p in parents.get(n, []):
                    children[p].add(n)
            outputs = [n for n in nodes if len(children[n]) == 0]
        
        return nodes, parents, inputs, outputs, layers
    
    else:
        raise ValueError(f"Unsupported dag format: {type(dag)}")

def _compute_layers(nodes, parents):
    """Compute layer assignment for each node based on longest path from inputs."""
    # Topological sort and compute depth
    in_degree = {n: 0 for n in nodes}
    children = defaultdict(list)
    for n in nodes:
        for p in parents.get(n, []):
            in_degree[n] += 1
            children[p].append(n)
    
    # BFS from inputs to compute depth
    depth = {}
    queue = []
    for n in nodes:
        if in_degree[n] == 0:
            depth[n] = 0
            queue.append(n)
    
    while queue:
        n = queue.pop(0)
        for c in children[n]:
            new_depth = depth[n] + 1
            if c not in depth or new_depth > depth[c]:
                depth[c] = new_depth
            queue.append(c)
    
    # Ensure all nodes have a depth
    for n in nodes:
        if n not in depth:
            depth[n] = 0
    
    return depth

def _greedy_layered_schedule(nodes, parents, inputs, outputs, fast_memory_budget, layers=None):
    """
    Greedy layer-by-layer pebbling schedule.
    For each layer:
      1. Load all input nodes needed for this layer
      2. Compute all nodes in this layer
      3. Store outputs and nodes needed for future layers; delete others
    """
    ops = []
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Compute layers if not provided
    if not layers:
        depth = _compute_layers(nodes, parents)
        layers = {n: depth.get(n, 0) for n in nodes}
    else:
        # Convert list of (node, layer) to dict
        if isinstance(layers, list):
            layers = {n: l for n, l in layers}
        elif isinstance(layers, dict):
            pass
        else:
            depth = _compute_layers(nodes, parents)
            layers = {n: depth.get(n, 0) for n in nodes}
    
    # Group nodes by layer
    layer_nodes = defaultdict(list)
    for n in nodes:
        layer_nodes[layers.get(n, 0)].append(n)
    
    max_layer = max(layer_nodes.keys()) if layer_nodes else 0
    
    # Track which nodes are currently red (in fast memory)
    red_set = set()
    # Track which nodes have been stored to slow memory (blue)
    stored_set = set()
    
    # Precompute: for each node, which future layers need it
    node_to_layers = {}
    for n in nodes:
        node_to_layers[n] = layers.get(n, 0)
    
    # For each node, compute the set of descendants (transitively)
    # to determine if it's needed in future layers
    children_map = defaultdict(set)
    for n in nodes:
        for p in parents.get(n, []):
            children_map[p].add(n)
    
    def get_descendants(n):
        """Get all transitive descendants of n."""
        result = set()
        stack = [n]
        while stack:
            cur = stack.pop()
            for c in children_map.get(cur, []):
                if c not in result:
                    result.add(c)
                    stack.append(c)
        return result
    
    # Precompute descendants for all nodes (for efficiency)
    descendants = {}
    for n in nodes:
        descendants[n] = get_descendants(n)
    
    for layer in range(max_layer + 1):
        if layer not in layer_nodes:
            continue
        
        layer_node_list = layer_nodes[layer]
        
        # Step 1: Load all parents needed for this layer that are not already red
        for node in layer_node_list:
            for p in parents.get(node, []):
                if p not in red_set and p not in stored_set:
                    # Need to load p
                    if len(red_set) < fast_memory_budget:
                        red_set.add(p)
                        ops.append(['load', p])
                    else:
                        # Need to free space: delete a red node that won't be needed
                        # Find a red node that is not needed for current or future layers
                        freed = False
                        for r in list(red_set):
                            # Check if r is needed for any node in current layer or future
                            needed = False
                            for ln in layer_node_list:
                                if r in parents.get(ln, []):
                                    needed = True
                                    break
                            if not needed:
                                # Check if r is needed for future layers
                                future_needed = False
                                for dl in descendants.get(r, []):
                                    if layers.get(dl, 0) > layer:
                                        future_needed = True
                                        break
                                if not future_needed:
                                    red_set.discard(r)
                                    ops.append(['delete', r])
                                    freed = True
                                    break
                        if not freed and red_set:
                            # Just delete any red node
                            r = next(iter(red_set))
                            red_set.discard(r)
                            ops.append(['delete', r])
                        if len(red_set) < fast_memory_budget:
                            red_set.add(p)
                            ops.append(['load', p])
        
        # Step 2: Compute all nodes in this layer
        for node in layer_node_list:
            if node in red_set:
                continue
            if node in input_set:
                continue  # Inputs are loaded, not computed
            par = parents.get(node, [])
            if all(p in red_set for p in par):
                if len(red_set) < fast_memory_budget:
                    red_set.add(node)
                    ops.append(['compute', node])
                else:
                    # Need to free space
                    freed = False
                    for r in list(red_set):
                        if r not in par:  # Don't delete a parent we need
                            # Check if r is needed for future
                            future_needed = False
                            for dl in descendants.get(r, []):
                                if layers.get(dl, 0) > layer:
                                    future_needed = True
                                    break
                            if not future_needed:
                                red_set.discard(r)
                                ops.append(['delete', r])
                                freed = True
                                break
                    if not freed and red_set:
                        r = next(iter(red_set))
                        red_set.discard(r)
                        ops.append(['delete', r])
                    if len(red_set) < fast_memory_budget:
                        red_set.add(node)
                        ops.append(['compute', node])
        
        # Step 3: After computing layer, decide what to store/delete
        # Store all outputs
        for node in layer_node_list:
            if node in output_set and node in red_set:
                red_set.discard(node)
                stored_set.add(node)
                ops.append(['store', node])
        
        # For non-output nodes, keep them if needed for future layers
        for node in layer_node_list:
            if node in red_set and node not in output_set:
                # Check if any descendant is in a future layer
                keep = False
                for dl in descendants.get(node, []):
                    if layers.get(dl, 0) > layer:
                        keep = True
                        break
                if not keep:
                    # Delete it (free)
                    red_set.discard(node)
                    ops.append(['delete', node])
    
    # Ensure all outputs are stored
    for node in outputs:
        if node not in stored_set and node in red_set:
            red_set.discard(node)
            stored_set.add(node)
            ops.append(['store', node])
    
    return ops

def _astar_search(dag, nodes, parents, inputs, outputs, fast_memory_budget, deadline):
    """A* search for small instances."""
    input_set = set(inputs)
    output_set = set(outputs)
    
    def heuristic(red_frozen, completed_frozen):
        # Better heuristic: count uncompleted outputs + estimate memory pressure
        uncompleted = len(output_set - completed_frozen)
        # Add penalty if fast memory is likely to be full
        red_count = len(red_frozen)
        if red_count >= fast_memory_budget:
            uncompleted += 1  # Extra cost for potential eviction
        return uncompleted
    
    initial_state = (frozenset(), frozenset())
    
    def is_goal(state):
        return state[1] == output_set
    
    g_score = {initial_state: 0}
    parent = {initial_state: None}
    f_score = {initial_state: heuristic(*initial_state)}
    pq = [(f_score[initial_state], 0, initial_state)]
    counter = 0
    
    best_score = float('inf')
    best_state = None
    
    while pq:
        if time.time() > deadline:
            break
        
        f, g, state = heapq.heappop(pq)
        
        if g > g_score.get(state, float('inf')):
            continue
        
        if is_goal(state):
            best_score = g
            best_state = state
            break
        
        red = set(state[0])
        completed = set(state[1])
        red_frozen = state[0]
        completed_frozen = state[1]
        
        # 1. Load an input node
        for inp in inputs:
            if inp not in red:
                if len(red) < fast_memory_budget:
                    new_red = red | {inp}
                    new_state = (frozenset(new_red), completed_frozen)
                    cost = 1
                    new_g = g + cost
                    if new_g < g_score.get(new_state, float('inf')):
                        g_score[new_state] = new_g
                        parent[new_state] = (state, 'load', inp)
                        h = heuristic(new_state[0], new_state[1])
                        f = new_g + h
                        counter += 1
                        heapq.heappush(pq, (f, counter, new_state))
        
        # 2. Compute a non-input node
        for node in nodes:
            if node in red or node in input_set:
                continue
            par = parents.get(node, [])
            if all(p in red for p in par):
                if len(red) < fast_memory_budget:
                    new_red = red | {node}
                    new_completed = completed_frozen | {node} if node in output_set else completed_frozen
                    new_state = (frozenset(new_red), new_completed)
                    cost = 0
                    new_g = g + cost
                    if new_g < g_score.get(new_state, float('inf')):
                        g_score[new_state] = new_g
                        parent[new_state] = (state, 'compute', node)
                        h = heuristic(new_state[0], new_state[1])
                        f = new_g + h
                        counter += 1
                        heapq.heappush(pq, (f, counter, new_state))
        
        # 3. Store a red node
        for node in red:
            new_red = red - {node}
            new_completed = completed_frozen | {node} if node in output_set else completed_frozen
            new_state = (frozenset(new_red), new_completed)
            cost = 1
            new_g = g + cost
            if new_g < g_score.get(new_state, float('inf')):
                g_score[new_state] = new_g
                parent[new_state] = (state, 'store', node)
                h = heuristic(new_state[0], new_state[1])
                f = new_g + h
                counter += 1
                heapq.heappush(pq, (f, counter, new_state))
        
        # 4. Delete a red node
        for node in red:
            new_red = red - {node}
            new_state = (frozenset(new_red), completed_frozen)
            cost = 0
            new_g = g + cost
            if new_g < g_score.get(new_state, float('inf')):
                g_score[new_state] = new_g
                parent[new_state] = (state, 'delete', node)
                h = heuristic(new_state[0], new_state[1])
                f = new_g + h
                counter += 1
                heapq.heappush(pq, (f, counter, new_state))
    
    if best_state is not None:
        ops = []
        state = best_state
        while parent[state] is not None:
            prev_state, op, vertex = parent[state]
            ops.append([op, vertex])
            state = prev_state
        ops.reverse()
        if len(ops) <= 20000:
            return ops
    
    return None

def pebble(dag, fast_memory_budget):
    """
    Find a minimum-transfer pebbling schedule.
    Uses layered greedy for larger DAGs, A* for small ones.
    """
    deadline = time.time() + 80.0  # 10 second safety margin
    
    try:
        nodes, parents, inputs, outputs, layers = _extract_dag_info(dag)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # For small instances, try A* first
    if len(nodes) <= 15 and fast_memory_budget <= 5:
        result = _astar_search(dag, nodes, parents, inputs, outputs, fast_memory_budget, deadline)
        if result is not None:
            return result
    
    # Use greedy layered schedule
    try:
        result = _greedy_layered_schedule(nodes, parents, inputs, outputs, fast_memory_budget, layers)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    
    # Fallback to baseline
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
