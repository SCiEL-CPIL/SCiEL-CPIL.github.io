"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

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

def _state_key(red_set):
    """Create a hashable key from a set of red nodes."""
    return frozenset(red_set)

def _compute_cost(operation, vertex, is_input_node, is_output_node, parents_map):
    """
    Compute the transfer cost for an operation.
    - load: 1 transfer (from slow to fast)
    - store: 1 transfer (from fast to slow)
    - compute: 0 transfers (if all parents are red, or computed in-place)
    - delete: 0 transfers (free)
    """
    if operation == 'load':
        return 1
    elif operation == 'store':
        return 1
    elif operation == 'compute':
        # Computing requires parents to be available (red)
        # If parents are blue, we need to load them first (handled separately)
        # The compute operation itself has 0 transfer cost
        return 0
    elif operation == 'delete':
        return 0
    else:
        return 0

def pebble(dag, fast_memory_budget):
    """
    Find a minimum-transfer pebbling schedule using Dijkstra's algorithm
    over the state space of red-pebble configurations.
    
    State: frozenset of red nodes
    Goal: all inputs and outputs are blue (not in red set), and all outputs have been computed
    """
    deadline = time.time() + 85.0  # 5 second safety margin
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        # Fallback to baseline
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    input_set = set(inputs)
    output_set = set(outputs)
    all_nodes = set(nodes)
    
    # State: frozenset of red nodes
    # Initial state: no red nodes (all blue)
    # But we need to track which outputs have been computed
    # Actually, the problem says "all outputs must finish blue"
    # So we need to compute each output, then it becomes blue (stored)
    
    # Let me reconsider the model:
    # - Inputs start blue (in slow memory)
    # - We can load inputs to fast memory (become red)
    # - We can compute a node if all its parents are red (or can be made red)
    # - Computed nodes are red in fast memory
    # - We can store a red node to slow memory (becomes blue)
    # - We can delete a red node (frees fast memory)
    # - All outputs must finish blue (in slow memory)
    
    # State needs to track:
    # 1. Which nodes are red (in fast memory)
    # 2. Which outputs have been stored (completed)
    
    # For Dijkstra, let's use state = (frozenset(red_nodes), frozenset(completed_outputs))
    
    initial_state = (frozenset(), frozenset())
    
    # Goal: all outputs completed, and we don't need any red nodes
    # Actually, the final state just needs all outputs blue
    # Red nodes can be anything as long as outputs are done
    
    def is_goal(state):
        red, completed = state
        return completed == output_set
    
    # Use Dijkstra
    # dist[state] = minimum transfers to reach state
    # parent[state] = (prev_state, operation, vertex)
    
    dist = {initial_state: 0}
    parent = {initial_state: None}
    pq = [(0, initial_state)]
    
    best_score = float('inf')
    best_state = None
    
    while pq:
        if time.time() > deadline:
            break
        
        d, state = heapq.heappop(pq)
        
        if d > dist.get(state, float('inf')):
            continue
        
        if is_goal(state):
            best_score = d
            best_state = state
            break
        
        red = set(state[0])
        completed = set(state[1])
        
        # Generate all valid next states
        next_states = []
        
        # 1. Load an input node (if not already red and not completed)
        for inp in inputs:
            if inp not in red and inp not in completed:
                if len(red) < fast_memory_budget:
                    new_red = red | {inp}
                    new_state = (_state_key(new_red), _state_key(completed))
                    cost = 1  # load costs 1 transfer
                    new_d = d + cost
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent[new_state] = (state, 'load', inp)
                        heapq.heappush(pq, (new_d, new_state))
        
        # 2. Compute a node (if all parents are red, and node not completed)
        for node in nodes:
            if node in completed:
                continue
            if node in red:
                continue
            # Check if all parents are red
            par = parents.get(node, [])
            if all(p in red for p in par):
                # Can compute
                if len(red) < fast_memory_budget:
                    new_red = red | {node}
                    new_completed = completed | {node} if node in output_set else completed
                    new_state = (_state_key(new_red), _state_key(new_completed))
                    cost = 0  # compute costs 0 transfers
                    new_d = d + cost
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent[new_state] = (state, 'compute', node)
                        heapq.heappush(pq, (new_d, new_state))
        
        # 3. Store a red node (if it's an output or we want to free space)
        for node in red:
            # We can store any red node, but only outputs need to be stored
            # For non-outputs, storing is optional (to free space)
            new_red = red - {node}
            new_completed = completed | {node} if node in output_set else completed
            new_state = (_state_key(new_red), _state_key(new_completed))
            cost = 1  # store costs 1 transfer
            new_d = d + cost
            if new_d < dist.get(new_state, float('inf')):
                dist[new_state] = new_d
                parent[new_state] = (state, 'store', node)
                heapq.heappush(pq, (new_d, new_state))
        
        # 4. Delete a red node (free, to free space)
        for node in red:
            new_red = red - {node}
            new_state = (_state_key(new_red), _state_key(completed))
            cost = 0  # delete costs 0 transfers
            new_d = d + cost
            if new_d < dist.get(new_state, float('inf')):
                dist[new_state] = new_d
                parent[new_state] = (state, 'delete', node)
                heapq.heappush(pq, (new_d, new_state))
    
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
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
