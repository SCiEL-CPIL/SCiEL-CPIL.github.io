"""Improved DAG pebbling solver with better eviction heuristics."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin (90s limit)
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', dag.get('sinks', []))
        inputs = dag.get('inputs', dag.get('sources', []))
    elif isinstance(dag, (list, tuple)):
        vertices = list(range(len(dag)))
        parents = {}
        outputs = []
        inputs = []
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                p = v.get('parents', v.get('deps', []))
                parents[i] = list(p) if p else []
            else:
                parents[i] = []
    else:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    if not vertices:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    # Build adjacency
    n = len(vertices)
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Compute topological order
    in_degree = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] += 1
    
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    # Identify input and output vertices
    input_set = set(inputs) if inputs else set(v for v in vertices if in_degree[v] == 0)
    output_set = set(outputs) if outputs else set(v for v in vertices if len(children[v]) == 0)
    
    # Compute "remaining dependents" count for each vertex (number of descendants)
    # This helps prioritize keeping high-reuse values in fast memory
    remaining_deps = {}
    for v in reversed(topo_order):
        if children[v]:
            remaining_deps[v] = 1 + max(remaining_deps.get(c, 0) for c in children[v])
        else:
            remaining_deps[v] = 0
    
    # Compute total fan-out (number of direct children)
    fan_out = {v: len(children[v]) for v in vertices}
    
    # Compute "persistence cost" = fan_out * (1 + remaining_deps)
    # Higher values indicate more expensive to recompute
    persistence_cost = {v: fan_out[v] * (1 + remaining_deps[v]) for v in vertices}
    
    # State tracking
    state = {}  # vertex -> 'blue' or 'red'
    for v in vertices:
        if v in input_set:
            state[v] = 'blue'
        else:
            state[v] = 'red'
    
    fast_mem = set(v for v in vertices if state[v] == 'blue')
    operations = []
    
    # Track which vertices have been computed
    computed = set(v for v in vertices if v in input_set)
    
    # Process in topological order
    # Strategy: For each vertex, ensure parents are available, then compute
    # Use a priority-based eviction: evict vertices with lowest persistence cost
    # that are not needed soon
    
    # Precompute for each vertex, the set of vertices that depend on it (transitively)
    # This helps determine if a vertex will be needed again
    transitive_dependents = {}
    for v in reversed(topo_order):
        deps = set()
        for c in children[v]:
            deps.add(c)
            if c in transitive_dependents:
                deps.update(transitive_dependents[c])
        transitive_dependents[v] = deps
    
    # Track which vertices are still needed (have unprocessed dependents)
    # A vertex is "still needed" if it has at least one unprocessed child
    still_needed = set()
    processed = set()
    
    for v in topo_order:
        if time.time() > deadline:
            break
        
        parents_list = parents.get(v, [])
        
        # Check if all parents are blue
        all_parents_blue = all(state[p] == 'blue' for p in parents_list)
        
        if not all_parents_blue:
            # Load red parents
            for p in parents_list:
                if state[p] == 'red':
                    operations.append(['load', p])
                    state[p] = 'blue'
                    fast_mem.add(p)
            
            # Evict if over budget
            while len(fast_mem) > fast_memory_budget:
                # Find victim: lowest persistence cost, not output, not v, 
                # and not needed by any unprocessed vertex
                candidates = []
                for u in fast_mem:
                    if u in output_set:
                        continue
                    if u == v:
                        continue
                    # Check if u is needed by any unprocessed vertex
                    needed = False
                    for c in children[u]:
                        if c not in processed:
                            needed = True
                            break
                    if needed:
                        continue
                    candidates.append(u)
                
                if not candidates:
                    # No good candidates, evict lowest persistence cost non-output
                    candidates = [u for u in fast_mem if u not in output_set and u != v]
                    if not candidates:
                        break
                
                victim = min(candidates, key=lambda u: persistence_cost[u])
                operations.append(['store', victim])
                state[victim] = 'red'
                fast_mem.discard(victim)
        
        # Compute v if not already blue
        if state[v] == 'red':
            operations.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)
            computed.add(v)
        
        # Mark v as processed
        processed.add(v)
        
        # Evict if over budget after compute
        while len(fast_mem) > fast_memory_budget:
            candidates = []
            for u in fast_mem:
                if u in output_set:
                    continue
                if u == v:
                    continue
                # Check if u is needed by any unprocessed vertex
                needed = False
                for c in children[u]:
                    if c not in processed:
                        needed = True
                        break
                if needed:
                    continue
                candidates.append(u)
            
            if not candidates:
                candidates = [u for u in fast_mem if u not in output_set and u != v]
                if not candidates:
                    break
            
            victim = min(candidates, key=lambda u: persistence_cost[u])
            operations.append(['store', victim])
            state[victim] = 'red'
            fast_mem.discard(victim)
    
    # Ensure all outputs are blue
    for v in output_set:
        if state[v] == 'red':
            operations.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
        
        while len(fast_mem) > fast_memory_budget:
            candidates = [u for u in fast_mem if u not in output_set]
            if not candidates:
                break
            victim = min(candidates, key=lambda u: persistence_cost[u])
            operations.append(['store', victim])
            state[victim] = 'red'
            fast_mem.discard(victim)
    
    # Limit operations to 20000
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
