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
    Greedy pebbling with lookahead eviction.
    """
    deadline = time.time() + 50.0
    
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
        # Cycle or error, fallback
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

    # Precompute: for each node, list of children (for lookahead)
    # We need to know which children are "upcoming" (not yet processed in topo_order)
    # Since we process in topo_order, "upcoming" means index > current index.
    
    red = set()
    completed = set()
    schedule = []
    transfers = 0
    
    # Track when nodes were loaded/computed for LRU tie-breaking
    timestamp = {}
    current_time = 0
    
    # To speed up eviction, we can pre-compute the "future fan-out" for each node at each step?
    # No, that's expensive. We'll calculate it on the fly.
    
    for node in topo_order:
        if time.time() > deadline:
            break
            
        # 1. Ensure parents are in red
        for p in parents.get(node, []):
            if p not in red:
                # Load parent
                if len(red) >= fast_memory_budget:
                    # Evict
                    # Find node in red with min future children (excluding completed)
                    # Future children = children[c] that are not in completed and index > current index
                    # Approximation: children not in completed
                    min_val = float('inf')
                    victim = None
                    for r in red:
                        if r in completed:
                            continue
                        # Count children of r that are not completed
                        fut = len([c for c in children[r] if c not in completed])
                        # Tie-break by timestamp (LRU)
                        t = timestamp.get(r, 0)
                        if (fut, t) < (min_val, 0): # This comparison is wrong, need tuple compare
                            pass
                        # Proper tuple compare
                        if victim is None:
                            victim = r
                            min_val = (fut, t)
                        else:
                            if (fut, t) < min_val:
                                victim = r
                                min_val = (fut, t)
                    
                    if victim:
                        # Delete victim
                        red.remove(victim)
                        schedule.append(['delete', victim])
                        # No transfer cost for delete
                    
                red.add(p)
                timestamp[p] = current_time
                current_time += 1
                schedule.append(['load', p])
                transfers += 1
        
        # 2. Compute node
        if node not in red:
            if len(red) >= fast_memory_budget:
                # Evict
                min_val = float('inf')
                victim = None
                for r in red:
                    if r in completed:
                        continue
                    fut = len([c for c in children[r] if c not in completed])
                    t = timestamp.get(r, 0)
                    if victim is None:
                        victim = r
                        min_val = (fut, t)
                    else:
                        if (fut, t) < min_val:
                            victim = r
                            min_val = (fut, t)
                
                if victim:
                    red.remove(victim)
                    schedule.append(['delete', victim])
            
            red.add(node)
            timestamp[node] = current_time
            current_time += 1
            schedule.append(['compute', node])
        
        # 3. If node is output, store it
        if node in output_set:
            # Store node
            red.remove(node)
            completed.add(node)
            schedule.append(['store', node])
            transfers += 1
            # timestamp[node] is no longer relevant as it's not in red
    
    # Check if all outputs are completed
    if completed != output_set:
        # Fallback if we didn't finish (e.g. time limit)
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Verify schedule length
    if len(schedule) > 20000:
        # This shouldn't happen for reasonable DAGs with greedy
        pass
        
    return schedule
# EVOLVE-BLOCK-END
