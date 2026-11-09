"""DAG pebbling with bounded fast memory; optimized search and heuristics."""
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
        
        return nodes, parents, inputs, outputs
    
    elif isinstance(dag, (list, tuple)):
        # List of (node, parents) or list of dicts
        nodes = []
        parents = {}
        inputs = []
        outputs = []
        
        for item in dag:
            if isinstance(item, dict):
                node = item.get('node', item.get('id', item.get('name')))
                par = item.get('parents', item.get('inputs', []))
                is_input = item.get('is_input', item.get('input', False))
                is_output = item.get('is_output', item.get('output', False))
                nodes.append(node)
                parents[node] = list(par)
                if is_input:
                    inputs.append(node)
                if is_output:
                    outputs.append(node)
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
        
        return nodes, parents, inputs, outputs
    
    else:
        raise ValueError(f"Unsupported dag format: {type(dag)}")

def pebble(dag, fast_memory_budget):
    """
    Find a minimum-transfer pebbling schedule using A* search
    over the state space of red-pebble configurations.
    
    State: (frozenset of red nodes, frozenset of completed outputs)
    Goal: all outputs completed
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 40 second safety margin

    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        # Fallback to baseline
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    input_set = set(inputs)
    output_set = set(outputs)
    all_nodes = set(nodes)
    node_list = list(nodes)
    node_index = {n: i for i, n in enumerate(node_list)}
    
    # Precompute: for each node, whether it's an input or output
    is_input = {n: (n in input_set) for n in all_nodes}
    is_output = {n: (n in output_set) for n in all_nodes}
    
    # Precompute parent lists as tuples for faster lookup
    parent_tuples = {n: tuple(parents.get(n, [])) for n in all_nodes}
    parent_index = {n: tuple(node_index[p] for p in parents.get(n, [])) for n in all_nodes}
    
    # Precompute children map for better heuristics
    children_map = defaultdict(set)
    for n in nodes:
        for p in parents.get(n, []):
            children_map[p].add(n)
    
    # Heuristic: minimum additional transfers needed
    def heuristic(red_frozen, completed_frozen):
        # Each output that hasn't been completed needs at least 1 store
        uncompleted_outputs = output_set - completed_frozen
        # Also need to account for red nodes that are not needed for any uncompleted output
        # and are not themselves outputs
        return len(uncompleted_outputs)
    
    initial_state = (frozenset(), frozenset())
    
    def is_goal(state):
        return state[1] == output_set
    
    # A* search
    g_score = {initial_state: 0}
    parent = {initial_state: None}
    f_score = {initial_state: heuristic(*initial_state)}
    pq = [(f_score[initial_state], 0, initial_state)]
    counter = 0  # tie-breaker
    
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
        
        # Generate all valid next states
        
        # 1. Load an input node (if not already red)
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
        
        # 2. Compute a non-input node (if all parents are red, and node not in red)
        for node in node_list:
            if node in red:
                continue
            if node in input_set:
                continue  # Inputs are loaded, not computed
            par = parent_tuples[node]
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
        
        # 3. Store a red node to slow memory
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
        
        # 4. Delete a red node (free)
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
    
    # If we found a solution, reconstruct the path
    if best_state is not None:
        ops = []
        state = best_state
        while parent[state] is not None:
            prev_state, op, vertex = parent[state]
            ops.append([op, vertex])
            state = prev_state
        ops.reverse()
        
        # Check if we're within operation limit
        if len(ops) <= 20000:
            return ops
    
    # Fallback to baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        return []
# EVOLVE-BLOCK-END
