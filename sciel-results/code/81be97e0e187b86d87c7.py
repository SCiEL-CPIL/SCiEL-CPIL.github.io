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
    Improved greedy pebbling with depth-aware cost and precomputed urgency.
    """
    deadline = time.time() + 85.0 # Safety margin
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    input_set = set(inputs)
    output_set = set(outputs)
    all_nodes = set(nodes)
    
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
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Precompute topo index
    topo_index = {n: i for i, n in enumerate(topo_order)}
    
    # Precompute Depth (longest path from input)
    depth = {n: 0 for n in nodes}
    for n in topo_order:
        if parents.get(n):
            max_parent_depth = max(depth[p] for p in parents[n])
            depth[n] = max_parent_depth + 1
        # Inputs have depth 0
    
    # Precompute next use indices for each node
    # For each node u, we need to know the next index in topo_order where u is a parent.
    # We can store a list of children indices for each node, sorted.
    # Then during simulation, we maintain a pointer for each node to its next child index.
    
    children_indices = {n: [] for n in nodes}
    for n in nodes:
        for c in children[n]:
            children_indices[n].append(topo_index[c])
        children_indices[n].sort()
    
    # Pointer for each node to the next child to be processed
    next_child_ptr = {n: 0 for n in nodes}
    
    red = set()
    completed = set()
    schedule = []
    transfers = 0
    
    timestamp = {}
    current_time = 0
    
    def get_eviction_score(node, current_idx):
        """
        Lower score = better candidate for eviction.
        Score = RecomputeCost * Urgency
        """
        # Recompute Cost
        if node in input_set:
            recompute_cost = 1.0
        else:
            recompute_cost = float(depth[node] + 1)
            
        # Urgency
        # Find the next child index > current_idx
        ptr = next_child_ptr[node]
        ch_idx = children_indices[node]
        
        # Advance pointer if current child is already processed
        while ptr < len(ch_idx) and ch_idx[ptr] <= current_idx:
            ptr += 1
        next_child_ptr[node] = ptr
        
        if ptr < len(ch_idx):
            next_use_idx = ch_idx[ptr]
            distance = next_use_idx - current_idx
            if distance <= 0:
                urgency = 1.0 # Should not happen if ptr logic is correct, but safe
            else:
                urgency = 1.0 / distance
        else:
            urgency = 0.0
            
        score = recompute_cost * urgency
        
        # Tie-break by LRU
        t = timestamp.get(node, 0)
        # Older nodes (lower t) get a slight penalty added to score? 
        # No, we want to evict older nodes. 
        # So we want older nodes to have LOWER score? 
        # If score is "value to keep", we evict min score.
        # If we want to evict older nodes, we should make their score lower.
        # So subtract a small term based on age?
        # Or just use timestamp as secondary sort key.
        # Let's add a small epsilon based on timestamp to break ties.
        # Higher timestamp (newer) should have higher score (keep longer).
        # So score += timestamp * small_epsilon?
        # If we evict min score, and newer has higher score, newer is kept. Correct.
        score += t * 1e-9
        
        return score

    for idx, node in enumerate(topo_order):
        if time.time() > deadline:
            break
            
        # 1. Ensure parents are in red
        for p in parents.get(node, []):
            if p not in red:
                if len(red) >= fast_memory_budget:
                    # Evict
                    min_score = float('inf')
                    victim = None
                    for r in red:
                        s = get_eviction_score(r, idx)
                        if s < min_score:
                            min_score = s
                            victim = r
                    
                    if victim:
                        red.remove(victim)
                        schedule.append(['delete', victim])
                
                red.add(p)
                timestamp[p] = current_time
                current_time += 1
                schedule.append(['load', p])
                transfers += 1
        
        # 2. Compute node
        if node not in red:
            if len(red) >= fast_memory_budget:
                min_score = float('inf')
                victim = None
                for r in red:
                    s = get_eviction_score(r, idx)
                    if s < min_score:
                        min_score = s
                        victim = r
                
                if victim:
                    red.remove(victim)
                    schedule.append(['delete', victim])
            
            red.add(node)
            timestamp[node] = current_time
            current_time += 1
            schedule.append(['compute', node])
        
        # 3. If node is output, store it
        if node in output_set:
            if node in red:
                red.remove(node)
            completed.add(node)
            schedule.append(['store', node])
            transfers += 1
    
    if completed != output_set:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
        
    return schedule
# EVOLVE-BLOCK-END
