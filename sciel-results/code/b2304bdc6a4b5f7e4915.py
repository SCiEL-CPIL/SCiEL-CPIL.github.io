"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _extract_dag_info(dag):
    """
    Extract DAG structure. Handles common formats.
    """
    if isinstance(dag, dict):
        nodes = dag.get('nodes', [])
        if not nodes and 'vertices' in dag:
            nodes = dag['vertices']
        edges = dag.get('edges', [])
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        parents = dag.get('parents', {})
        
        if not parents:
            parents = {n: [] for n in nodes}
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    parents[e[1]].append(e[0])
        
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

def pebble(dag, fast_memory_budget):
    """
    Hybrid pebbling: 
    1. Bitmask Dijkstra for small DAGs (N <= 20).
    2. Greedy heuristic with improved eviction for larger DAGs.
    """
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        # Fallback if extraction fails
        return []

    input_set = set(inputs)
    output_set = set(outputs)
    all_nodes = set(nodes)
    n_nodes = len(nodes)
    
    # Topological sort
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
            
    queue = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while queue:
        n = queue.popleft()
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(nodes):
        return []

    # Precompute node indices for bitmask operations
    node_index = {n: i for i, n in enumerate(nodes)}
    inputs_mask = 0
    for i in inputs:
        inputs_mask |= (1 << node_index[i])
    outputs_mask = 0
    for o in outputs:
        outputs_mask |= (1 << node_index[o])
    
    # Precompute parent masks for each node
    parent_masks = {}
    for n in nodes:
        mask = 0
        for p in parents.get(n, []):
            mask |= (1 << node_index[p])
        parent_masks[n] = mask

    # Strategy 1: Bitmask Dijkstra for small N
    if n_nodes <= 20:
        try:
            result = _dijkstra_bitmask(nodes, parents, inputs, outputs, fast_memory_budget, node_index, parent_masks, inputs_mask, outputs_mask, deadline)
            if result is not None:
                return result
        except Exception:
            pass

    # Strategy 2: Greedy with Improved Eviction
    result = _greedy_pebble_improved(nodes, parents, inputs, outputs, fast_memory_budget, topo_order, children, node_index, deadline)
    if result is not None:
        return result
    
    return []

def _dijkstra_bitmask(nodes, parents, inputs, outputs, budget, node_index, parent_masks, inputs_mask, outputs_mask, deadline):
    """
    Dijkstra's algorithm using bitmasks for state representation.
    State: (red_mask, completed_mask)
    """
    n = len(nodes)
    init_red = 0
    init_comp = 0
    init_state = (init_red, init_comp)
    
    dist = {init_state: 0}
    parent_map = {init_state: None}
    pq = [(0, init_state)]
    
    # Precompute list of nodes as indices for iteration
    node_list = list(nodes)
    
    while pq:
        if time.time() > deadline:
            return None
            
        d, state = heapq.heappop(pq)
        if d > dist.get(state, float('inf')):
            continue
            
        red, completed = state
        
        # Check goal
        if completed == outputs_mask:
            schedule = []
            curr = state
            while parent_map[curr] is not None:
                prev, op, v_idx = parent_map[curr]
                schedule.append([op, node_list[v_idx]])
                curr = prev
            schedule.reverse()
            return schedule
        
        # Generate neighbors
        # 1. Load inputs
        # Iterate through set bits in inputs_mask
        temp_mask = inputs_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v_idx = lsb.bit_length() - 1
            temp_mask ^= lsb
            
            if not (red & lsb): # Not in red
                if red.bit_count() < budget:
                    new_red = red | lsb
                    new_state = (new_red, completed)
                    new_d = d + 1
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent_map[new_state] = (state, 'load', v_idx)
                        heapq.heappush(pq, (new_d, new_state))

        # 2. Compute nodes
        for i, node in enumerate(node_list):
            bit = 1 << i
            if red & bit or (completed & bit):
                continue
            # Check if all parents are in red
            if (red & parent_masks[node]) == parent_masks[node]:
                new_red = red | bit
                new_state = (new_red, completed)
                new_d = d
                if new_d < dist.get(new_state, float('inf')):
                    dist[new_state] = new_d
                    parent_map[new_state] = (state, 'compute', i)
                    heapq.heappush(pq, (new_d, new_state))

        # 3. Store outputs
        temp_mask = outputs_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v_idx = lsb.bit_length() - 1
            temp_mask ^= lsb
            
            if (red & lsb) and not (completed & lsb):
                new_red = red & ~lsb
                new_comp = completed | lsb
                new_state = (new_red, new_comp)
                new_d = d + 1
                if new_d < dist.get(new_state, float('inf')):
                    dist[new_state] = new_d
                    parent_map[new_state] = (state, 'store', v_idx)
                    heapq.heappush(pq, (new_d, new_state))

        # 4. Delete red nodes
        temp_red = red
        while temp_red:
            lsb = temp_red & -temp_red
            v_idx = lsb.bit_length() - 1
            temp_red ^= lsb
            
            new_red = red & ~lsb
            new_state = (new_red, completed)
            new_d = d
            if new_d < dist.get(new_state, float('inf')):
                dist[new_state] = new_d
                parent_map[new_state] = (state, 'delete', v_idx)
                heapq.heappush(pq, (new_d, new_state))

    return None

def _greedy_pebble_improved(nodes, parents, inputs, outputs, budget, topo_order, children, node_index, deadline):
    """
    Greedy pebbling with improved eviction based on future fan-out and node type.
    """
    input_set = set(inputs)
    output_set = set(outputs)
    
    red = set()
    completed = set()
    schedule = []
    
    timestamp = {}
    current_time = 0
    
    # Precompute future children counts dynamically? 
    # We can just iterate children each time.
    
    for node in topo_order:
        if time.time() > deadline:
            return None
            
        # 1. Ensure parents are in red
        for p in parents.get(node, []):
            if p not in red:
                if len(red) >= budget:
                    # Evict
                    victim = _select_victim_improved(red, completed, children, timestamp, input_set)
                    if victim:
                        red.remove(victim)
                        schedule.append(['delete', victim])
                        timestamp.pop(victim, None)
                    
                red.add(p)
                timestamp[p] = current_time
                current_time += 1
                schedule.append(['load', p])
        
        # 2. Compute node
        if node not in red:
            if len(red) >= budget:
                victim = _select_victim_improved(red, completed, children, timestamp, input_set)
                if victim:
                    red.remove(victim)
                    schedule.append(['delete', victim])
                    timestamp.pop(victim, None)
            
            red.add(node)
            timestamp[node] = current_time
            current_time += 1
            schedule.append(['compute', node])
        
        # 3. If node is output, store it
        if node in output_set:
            red.remove(node)
            completed.add(node)
            schedule.append(['store', node])
            timestamp.pop(node, None)
            
            # Proactive deletion of nodes with no future children
            for r in list(red):
                if r in completed:
                    continue
                has_future_child = False
                for c in children[r]:
                    if c not in completed:
                        # Check if c is in the future (topo index > current)
                        # Since we process in topo order, if c is not completed, 
                        # and c != r, it is likely in the future or being processed.
                        # If c is not completed, it hasn't been stored.
                        # If c is an intermediate, it's not in 'completed'.
                        # So if c is not completed, it's a future need.
                        has_future_child = True
                        break
                if not has_future_child:
                    red.remove(r)
                    schedule.append(['delete', r])
                    timestamp.pop(r, None)

    if completed != output_set:
        return None
        
    return schedule

def _select_victim_improved(red, completed, children, timestamp, input_set):
    """
    Select victim: Minimize (future_children_weighted, timestamp).
    Weighted future children: Inputs are cheaper to reload, so we prefer evicting them?
    No, we want to keep what is expensive to recompute.
    Intermediates are expensive. So we want to keep Intermediates with high fan-out.
    So we evict the one with LOWEST "value".
    Value = future_children * (1 if input else 2)
    """
    min_val = (float('inf'), float('inf'))
    victim = None
    
    for r in red:
        if r in completed:
            continue
        fut = len([c for c in children[r] if c not in completed])
        # If it's an input, reloading is 1 transfer.
        # If it's intermediate, reloading is 1 transfer + recompute.
        # So intermediate is "more valuable" to keep.
        weight = 1 if r in input_set else 2
        val = (fut * weight, timestamp.get(r, 0))
        
        if val < min_val:
            min_val = val
            victim = r
            
    return victim
# EVOLVE-BLOCK-END
