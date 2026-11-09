import time
import heapq
from collections import defaultdict

def _extract_dag_info(dag):
    """Extract DAG structure information."""
    raw_parents = dag.get('parents', {})
    
    parents = {}
    if isinstance(raw_parents, dict):
        # Keys might be strings or ints
        for k, v in raw_parents.items():
            parents[int(k)] = [int(x) for x in v]
    elif isinstance(raw_parents, list):
        for i, ps in enumerate(raw_parents):
            parents[i] = [int(x) for x in ps]
    else:
        parents = {}

    outputs = dag.get('outputs', [])
    outputs = [int(o) for o in outputs]
    
    all_vertices = set()
    for v, ps in parents.items():
        all_vertices.add(v)
        all_vertices.update(ps)
    for v in outputs:
        all_vertices.add(v)
    
    inputs = set()
    for v in all_vertices:
        if v not in parents or not parents[v]:
            inputs.add(v)

    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
            
    return {
        'parents': parents,
        'outputs': outputs,
        'inputs': inputs,
        'children': children,
        'all_vertices': all_vertices
    }

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    info = _extract_dag_info(dag)
    budget = fast_memory_budget
    outputs = info['outputs']
    inputs = info['inputs']
    all_vertices = info['all_vertices']
    parents = info['parents']
    
    if not all_vertices:
        return []
        
    # Map vertices to indices 0..N-1 for bitmask representation
    vertices = sorted(list(all_vertices))
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    n_vertices = len(vertices)
    
    # Precompute parent bitmasks and input mask
    parent_masks = [0] * n_vertices
    input_mask = 0
    output_mask = 0
    
    for v in vertices:
        idx = v_to_idx[v]
        if v in inputs:
            input_mask |= (1 << idx)
        if v in outputs:
            output_mask |= (1 << idx)
        
        if v in parents:
            for p in parents[v]:
                if p in v_to_idx:
                    parent_masks[idx] |= (1 << v_to_idx[p])
                else:
                    # Parent not in mapped vertices? Should not happen if extraction is correct.
                    pass

    # Dijkstra using bitmasks
    # State: int bitmask
    # Initial state: 0 (no red vertices)
    # Goal state: (state & output_mask) == output_mask
    
    initial_state = 0
    goal_mask = output_mask
    
    # dist[state] = min cost
    dist = {initial_state: 0}
    # parent[state] = (prev_state, op, vertex_idx)
    parent = {}
    
    pq = [(0, initial_state)]
    visited = set()
    
    # Precompute list of vertices for iteration
    v_list = vertices
    
    while pq:
        if time.time() > deadline:
            break
            
        cost, state = heapq.heappop(pq)
        
        if state in visited:
            continue
        visited.add(state)
        
        if (state & goal_mask) == goal_mask:
            # Reconstruct path
            path = []
            cur = state
            while cur != initial_state:
                prev_state, op, v_idx = parent[cur]
                v = v_list[v_idx]
                path.append([op, v])
                cur = prev_state
            path.reverse()
            return path
        
        red_count = state.bit_count()
        
        # 1. Load: add an input vertex
        # Inputs are those bits set in input_mask but not in state
        # We can iterate through input vertices
        for v in inputs:
            idx = v_to_idx[v]
            bit = 1 << idx
            if not (state & bit) and red_count < budget:
                new_state = state | bit
                new_cost = cost + 1
                if new_state not in dist or new_cost < dist[new_state]:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'load', idx)
                    heapq.heappush(pq, (new_cost, new_state))
                    
        # 2. Store: remove a non-input vertex
        # Iterate through set bits in state that are not inputs
        # non_input_mask = ~input_mask & ((1 << n_vertices) - 1)
        non_input_mask = ((1 << n_vertices) - 1) ^ input_mask
        candidates_to_store = state & non_input_mask
        
        while candidates_to_store:
            # Get lowest set bit
            lsb = candidates_to_store & -candidates_to_store
            idx = lsb.bit_length() - 1
            candidates_to_store ^= lsb
            
            new_state = state ^ lsb
            new_cost = cost + 1
            if new_state not in dist or new_cost < dist[new_state]:
                dist[new_state] = new_cost
                parent[new_state] = (state, 'store', idx)
                heapq.heappush(pq, (new_cost, new_state))
                
        # 3. Compute: add a vertex if all parents are red
        # Iterate through vertices not in state
        # For each, check if (state & parent_mask[v]) == parent_mask[v]
        # Optimization: Only check if parent_mask[v] is subset of state
        for v in v_list:
            idx = v_to_idx[v]
            bit = 1 << idx
            if not (state & bit) and red_count < budget:
                pm = parent_masks[idx]
                if (state & pm) == pm:
                    new_state = state | bit
                    new_cost = cost
                    if new_state not in dist or new_cost < dist[new_state]:
                        dist[new_state] = new_cost
                        parent[new_state] = (state, 'compute', idx)
                        heapq.heappush(pq, (new_cost, new_state))
                        
    # If no solution found via Dijkstra (timeout or no path), return empty
    # In a valid problem, a path should exist if budget > 0.
    return []
