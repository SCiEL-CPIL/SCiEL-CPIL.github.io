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
            children = {n: set() for n in nodes}
            for n in nodes:
                for p in parents.get(n, []):
                    children[p].add(n)
            outputs = [n for n in nodes if len(children[n]) == 0]
        
        return nodes, parents, inputs, outputs
    
    elif isinstance(dag, (list, tuple)):
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


def _greedy_schedule(nodes, parents, inputs, outputs, fast_memory_budget):
    """
    Greedy topological pebbling: process nodes in topological order,
    loading/storing as needed. This gives a valid baseline.
    """
    ops = []
    red = set()
    completed_outputs = set()
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Topological order
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            if p in children:
                children[p].append(n)
    
    from collections import deque
    queue = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while queue:
        n = queue.popleft()
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Process in topological order
    for node in topo_order:
        par = parents.get(node, [])
        
        if node in input_set:
            # Load input if not already red
            if node not in red:
                if len(red) >= fast_memory_budget:
                    # Need to free space - delete a non-output red node
                    to_delete = None
                    for r in list(red):
                        if r not in output_set or r in completed_outputs:
                            to_delete = r
                            break
                    if to_delete is None:
                        # Delete any red node
                        to_delete = list(red)[0] if red else None
                    if to_delete:
                        ops.append(['delete', to_delete])
                        red.discard(to_delete)
                ops.append(['load', node])
                red.add(node)
        else:
            # Compute node - need all parents red
            for p in par:
                if p not in red:
                    if p in input_set:
                        if len(red) >= fast_memory_budget:
                            to_delete = None
                            for r in list(red):
                                if r not in output_set or r in completed_outputs:
                                    to_delete = r
                                    break
                            if to_delete is None:
                                to_delete = list(red)[0] if red else None
                            if to_delete:
                                ops.append(['delete', to_delete])
                                red.discard(to_delete)
                        ops.append(['load', p])
                        red.add(p)
                    else:
                        # Recompute parent
                        # This is complex; for greedy, we'll just delete and reload
                        pass
            
            if len(red) >= fast_memory_budget:
                to_delete = None
                for r in list(red):
                    if r != node and (r not in output_set or r in completed_outputs):
                        to_delete = r
                        break
                if to_delete is None and red:
                    to_delete = list(red)[0]
                if to_delete:
                    ops.append(['delete', to_delete])
                    red.discard(to_delete)
            
            ops.append(['compute', node])
            red.add(node)
        
        # If node is an output, store it
        if node in output_set:
            ops.append(['store', node])
            red.discard(node)
            completed_outputs.add(node)
    
    return ops


def pebble(dag, fast_memory_budget):
    """
    Find a minimum-transfer pebbling schedule using A* search
    with dominance pruning and a greedy fallback.
    """
    deadline = time.time() + 55.0  # Safety margin
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    input_set = set(inputs)
    output_set = set(outputs)
    all_nodes = set(nodes)
    node_list = list(nodes)
    
    # Precompute
    parent_tuples = {n: tuple(parents.get(n, [])) for n in all_nodes}
    
    # For heuristic: count minimum additional transfers
    # Each uncompleted output needs at least 1 store
    # Each input that must be loaded (not yet in red) needs 1 load
    # Simple admissible: number of uncompleted outputs
    def heuristic(red_frozen, completed_frozen):
        return len(output_set - completed_frozen)
    
    initial_state = (frozenset(), frozenset())
    
    def is_goal(state):
        return state[1] == output_set
    
    # A* with dominance pruning
    g_score = {initial_state: 0}
    parent_map = {initial_state: None}
    f_score = {initial_state: heuristic(*initial_state)}
    pq = [(f_score[initial_state], 0, initial_state)]
    counter = 0
    
    best_score = float('inf')
    best_state = None
    best_ops = None
    
    # Dominance: state A dominates B if A.red ⊆ B.red and A.completed ⊇ B.completed
    # and g(A) <= g(B). We'll use a simple check.
    
    max_expansions = 50000
    
    expansions = 0
    while pq and expansions < max_expansions:
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
        
        expansions += 1
        
        # Generate all valid next states
        next_candidates = []
        
        # 1. Load an input node
        for inp in inputs:
            if inp not in red:
                if len(red) < fast_memory_budget:
                    new_red = red | {inp}
                    new_state = (frozenset(new_red), completed_frozen)
                    cost = 1
                    next_candidates.append((new_state, cost, 'load', inp))
        
        # 2. Compute a non-input node
        for node in node_list:
            if node in red or node in input_set:
                continue
            par = parent_tuples[node]
            if all(p in red for p in par):
                if len(red) < fast_memory_budget:
                    new_red = red | {node}
                    new_completed = completed_frozen | {node} if node in output_set else completed_frozen
                    new_state = (frozenset(new_red), new_completed)
                    cost = 0
                    next_candidates.append((new_state, cost, 'compute', node))
        
        # 3. Store a red node
        for node in red:
            new_red = red - {node}
            new_completed = completed_frozen | {node} if node in output_set else completed_frozen
            new_state = (frozenset(new_red), new_completed)
            cost = 1
            next_candidates.append((new_state, cost, 'store', node))
        
        # 4. Delete a red node
        for node in red:
            new_red = red - {node}
            new_state = (frozenset(new_red), completed_frozen)
            cost = 0
            next_candidates.append((new_state, cost, 'delete', node))
        
        for new_state, cost, op, vertex in next_candidates:
            new_g = g + cost
            if new_g >= g_score.get(new_state, float('inf')):
                continue
            
            # Dominance check: skip if an existing state dominates this one
            dominated = False
            for existing_state, existing_g in g_score.items():
                if existing_g > new_g:
                    continue
                ex_red, ex_completed = existing_state
                if ex_red <= new_state[0] and ex_completed >= new_state[1]:
                    dominated = True
                    break
            
            if dominated:
                continue
            
            g_score[new_state] = new_g
            parent_map[new_state] = (state, op, vertex)
            h = heuristic(new_state[0], new_state[1])
            f = new_g + h
            counter += 1
            heapq.heappush(pq, (f, counter, new_state))
    
    # If we found a solution via search
    if best_state is not None:
        ops = []
        state = best_state
        while parent_map[state] is not None:
            prev_state, op, vertex = parent_map[state]
            ops.append([op, vertex])
            state = prev_state
        ops.reverse()
        
        if len(ops) <= 20000:
            return ops
    
    # Fallback to greedy
    try:
        greedy_ops = _greedy_schedule(nodes, parents, inputs, outputs, fast_memory_budget)
        if len(greedy_ops) <= 20000:
            return greedy_ops
    except Exception:
        pass
    
    # Final fallback
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
