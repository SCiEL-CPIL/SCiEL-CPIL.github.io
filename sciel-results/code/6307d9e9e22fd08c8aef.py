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
    Communication-minimizing DAG pebbling:
    1. For small DAGs (<=18 nodes): exact Dijkstra search over state space
    2. For medium/large DAGs: improved greedy with layered processing and smart eviction
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    input_set = set(inputs)
    output_set = set(outputs)
    all_nodes = set(nodes)
    n_nodes = len(nodes)
    
    # Precompute children map
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
            
    # Topological sort
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
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
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # For small DAGs: exact Dijkstra
    if n_nodes <= 18:
        try:
            best_schedule = _dijkstra_exact(nodes, parents, inputs, outputs, 
                                           fast_memory_budget, deadline)
            if best_schedule is not None:
                return best_schedule
        except Exception:
            pass
    
    # Greedy with improved eviction strategy
    return _greedy_pebble(nodes, parents, inputs, outputs, 
                          fast_memory_budget, topo_order, children, deadline)

def _dijkstra_exact(nodes, parents, inputs, outputs, fast_memory_budget, deadline):
    """Exact Dijkstra search for small DAGs."""
    output_set = set(outputs)
    initial_state = (frozenset(), frozenset())
    
    dist = {initial_state: 0}
    parent_map = {initial_state: None}
    pq = [(0, initial_state)]
    
    found_goal = False
    best_schedule = None
    
    nodes_list = list(nodes)
    inputs_list = list(inputs)
    outputs_list = list(outputs)
    
    while pq:
        if time.time() > deadline - 5.0:
            break
            
        d, state = heapq.heappop(pq)
        if d > dist.get(state, float('inf')):
            continue
            
        red = set(state[0])
        completed = set(state[1])
        
        if completed == output_set:
            found_goal = True
            path = []
            curr = state
            while parent_map[curr] is not None:
                prev_state, op, node = parent_map[curr]
                path.append([op, node])
                curr = prev_state
            path.reverse()
            best_schedule = path
            break
        
        # Generate neighbors
        # 1. Load an input
        for inp in inputs_list:
            if inp not in red:
                if len(red) < fast_memory_budget:
                    new_red = red | {inp}
                    new_state = (frozenset(new_red), state[1])
                    new_d = d + 1
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent_map[new_state] = (state, 'load', inp)
                        heapq.heappush(pq, (new_d, new_state))
                else:
                    for victim in red:
                        new_red = red - {victim} | {inp}
                        new_state = (frozenset(new_red), state[1])
                        new_d = d + 2
                        if new_d < dist.get(new_state, float('inf')):
                            dist[new_state] = new_d
                            parent_map[new_state] = (state, 'load', inp)
                            heapq.heappush(pq, (new_d, new_state))
        
        # 2. Compute a node
        for node in nodes_list:
            if node in completed or node in red:
                continue
            can_compute = all(p in red for p in parents.get(node, []))
            if can_compute:
                if len(red) < fast_memory_budget:
                    new_red = red | {node}
                    new_state = (frozenset(new_red), state[1])
                    new_d = d
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent_map[new_state] = (state, 'compute', node)
                        heapq.heappush(pq, (new_d, new_state))
                else:
                    for victim in red:
                        new_red = red - {victim} | {node}
                        new_state = (frozenset(new_red), state[1])
                        new_d = d + 1
                        if new_d < dist.get(new_state, float('inf')):
                            dist[new_state] = new_d
                            parent_map[new_state] = (state, 'compute', node)
                            heapq.heappush(pq, (new_d, new_state))
        
        # 3. Store an output
        for node in outputs_list:
            if node in red and node not in completed:
                new_red = red - {node}
                new_completed = completed | {node}
                new_state = (frozenset(new_red), frozenset(new_completed))
                new_d = d + 1
                if new_d < dist.get(new_state, float('inf')):
                    dist[new_state] = new_d
                    parent_map[new_state] = (state, 'store', node)
                    heapq.heappush(pq, (new_d, new_state))
        
        # 4. Delete a red node
        for node in red:
            new_red = red - {node}
            new_state = (frozenset(new_red), state[1])
            new_d = d
            if new_d < dist.get(new_state, float('inf')):
                dist[new_state] = new_d
                parent_map[new_state] = (state, 'delete', node)
                heapq.heappush(pq, (new_d, new_state))
    
    return best_schedule if found_goal else None

def _greedy_pebble(nodes, parents, inputs, outputs, fast_memory_budget, 
                   topo_order, children, deadline):
    """
    Improved greedy pebbling with:
    - Layered processing
    - Future-need-aware eviction
    - Batch loading of parents
    """
    output_set = set(outputs)
    red = set()
    completed = set()
    schedule = []
    transfers = 0
    
    # Compute remaining uses: how many children still need this node
    remaining_uses = {n: len(children[n]) for n in nodes}
    
    # Compute "urgency" for each node: how soon it will be needed
    # Use topological position as proxy
    topo_pos = {n: i for i, n in enumerate(topo_order)}
    
    # Track last access time for LRU tie-breaking
    last_access = {}
    current_time = 0
    
    # Process in topological order
    for node in topo_order:
        if time.time() > deadline:
            break
            
        # Ensure all parents are in red memory
        for p in parents.get(node, []):
            if p not in red:
                # Need to load p
                if len(red) >= fast_memory_budget:
                    # Find best victim to evict
                    victim = _find_best_victim(red, remaining_uses, last_access, 
                                               topo_pos, node, completed)
                    if victim is not None:
                        red.remove(victim)
                        schedule.append(['delete', victim])
                        if victim in last_access:
                            del last_access[victim]
                
                red.add(p)
                last_access[p] = current_time
                current_time += 1
                schedule.append(['load', p])
                transfers += 1
        
        # Compute the node
        if node not in red:
            if len(red) >= fast_memory_budget:
                victim = _find_best_victim(red, remaining_uses, last_access, 
                                           topo_pos, node, completed)
                if victim is not None:
                    red.remove(victim)
                    schedule.append(['delete', victim])
                    if victim in last_access:
                        del last_access[victim]
            
            red.add(node)
            last_access[node] = current_time
            current_time += 1
            schedule.append(['compute', node])
        
        # If this is an output, store it
        if node in output_set:
            if node in red:
                red.remove(node)
            completed.add(node)
            schedule.append(['store', node])
            transfers += 1
            if node in last_access:
                del last_access[node]
        
        # Update remaining uses for parents
        for p in parents.get(node, []):
            if p in remaining_uses:
                remaining_uses[p] = max(0, remaining_uses[p] - 1)
    
    # Verify all outputs completed
    if completed != output_set:
        # Try to complete remaining outputs
        for node in outputs:
            if node not in completed:
                # Load all parents
                for p in parents.get(node, []):
                    if p not in red:
                        if len(red) >= fast_memory_budget:
                            victim = _find_best_victim(red, remaining_uses, last_access,
                                                       topo_pos, node, completed)
                            if victim is not None:
                                red.remove(victim)
                                schedule.append(['delete', victim])
                                if victim in last_access:
                                    del last_access[victim]
                        red.add(p)
                        last_access[p] = current_time
                        current_time += 1
                        schedule.append(['load', p])
                        transfers += 1
                
                if node not in red:
                    if len(red) >= fast_memory_budget:
                        victim = _find_best_victim(red, remaining_uses, last_access,
                                                   topo_pos, node, completed)
                        if victim is not None:
                            red.remove(victim)
                            schedule.append(['delete', victim])
                            if victim in last_access:
                                del last_access[victim]
                    red.add(node)
                    last_access[node] = current_time
                    current_time += 1
                    schedule.append(['compute', node])
                
                if node in red:
                    red.remove(node)
                completed.add(node)
                schedule.append(['store', node])
                transfers += 1
    
    return schedule

def _find_best_victim(red, remaining_uses, last_access, topo_pos, current_node, completed):
    """
    Find the best node to evict from red memory.
    Strategy: minimize (remaining_uses, time_since_access, topological_distance)
    """
    best_victim = None
    best_score = float('inf')
    
    for r in red:
        if r in completed:
            continue
        
        ru = remaining_uses.get(r, 0)
        ta = last_access.get(r, 0)
        time_since_access = time.time()  # Use a proxy: lower access time = older
        
        # Score: prefer evicting nodes with low remaining uses, old access, 
        # and far from current node in topological order
        dist_to_current = abs(topo_pos.get(r, 0) - topo_pos.get(current_node, 0))
        
        # Composite score: primarily remaining uses, then age, then distance
        score = ru * 1000 + (10000 - ta) * 0.1 + dist_to_current * 0.01
        
        if score < best_score:
            best_score = score
            best_victim = r
    
    return best_victim

# EVOLVE-BLOCK-END
