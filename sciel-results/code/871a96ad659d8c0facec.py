# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 85
    
    vertices = dag.get('vertices', [])
    parents_map = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children[p].append(v)
    
    # Topological sort
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != n:
        topo_order = list(vertices)
    
    fast_mem = set()
    slow_mem = set(inputs)
    operations = []
    
    # remaining_uses[v] = number of children of v not yet processed
    remaining_uses = {v: len(children[v]) for v in vertices}
    
    for v in topo_order:
        if time.time() > deadline:
            break
        
        parents_of_v = parents_map.get(v, [])
        
        # Load parents if not in fast memory
        for p in parents_of_v:
            if p not in fast_mem and p in slow_mem:
                operations.append(['load', p])
                fast_mem.add(p)
        
        # Free space if needed
        while len(fast_mem) + 1 > fast_memory_budget and fast_mem:
            evict_candidates = [u for u in fast_mem if remaining_uses[u] == 0]
            if evict_candidates:
                to_evict = evict_candidates[0]
            else:
                parents_set = set(parents_of_v)
                to_evict = None
                for u in fast_mem:
                    if u in parents_set:
                        to_evict = u
                        break
                if to_evict is None:
                    to_evict = next(iter(fast_mem))
            
            if to_evict not in slow_mem:
                operations.append(['store', to_evict])
                slow_mem.add(to_evict)
            operations.append(['delete', to_evict])
            fast_mem.discard(to_evict)
        
        # Compute
        operations.append(['compute', v])
        fast_mem.add(v)
        
        # Decrement remaining_uses for parents
        for p in parents_of_v:
            remaining_uses[p] -= 1
    
    # Ensure outputs end in slow memory
    for v in topo_order:
        if time.time() > deadline:
            break
        if v in outputs and v in fast_mem:
            if v not in slow_mem:
                operations.append(['store', v])
                slow_mem.add(v)
            operations.append(['delete', v])
            fast_mem.discard(v)
    
    # Clean up remaining
    for v in list(fast_mem):
        if time.time() > deadline:
            break
        if v not in slow_mem:
            operations.append(['store', v])
            slow_mem.add(v)
        operations.append(['delete', v])
        fast_mem.discard(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
