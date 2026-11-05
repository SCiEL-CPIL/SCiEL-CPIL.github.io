"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85
    
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build children map
    children = defaultdict(list)
    in_degree = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
            in_degree[v] += 1
    
    # Topological sort
    topo_order = []
    queue = deque()
    for v in vertices:
        if in_degree[v] == 0:
            queue.append(v)
    
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != n:
        return []
    
    # Remaining uses: how many times each vertex is needed as a parent
    remaining_uses = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            remaining_uses[p] += 1
    
    # Fast memory state: set of vertices in fast memory
    fast_mem = set()
    
    # Inputs start blue (in fast memory)
    for inp in inputs:
        if len(fast_mem) < fast_memory_budget:
            fast_mem.add(inp)
        # If budget exceeded, we'll handle eviction later
    
    operations = []
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        if v in inputs:
            continue
        
        # Ensure parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                # Evict if needed
                while len(fast_mem) >= fast_memory_budget:
                    # Try to evict a vertex with 0 remaining uses that's not an output
                    evicted = False
                    for candidate in sorted(fast_mem, key=lambda x: (remaining_uses[x], x)):
                        if candidate not in outputs and remaining_uses[candidate] == 0:
                            fast_mem.remove(candidate)
                            operations.append(['store', candidate])
                            evicted = True
                            break
                    if not evicted:
                        # Evict any non-output
                        for candidate in sorted(fast_mem, key=lambda x: (remaining_uses[x], x)):
                            if candidate not in outputs:
                                fast_mem.remove(candidate)
                                operations.append(['store', candidate])
                                evicted = True
                                break
                    if not evicted:
                        break
                
                operations.append(['load', p])
                fast_mem.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_mem.add(v)
        
        # Update remaining uses
        for p in parents.get(v, []):
            remaining_uses[p] -= 1
    
    # Ensure outputs are in fast memory
    for out in outputs:
        if out not in fast_mem:
            while len(fast_mem) >= fast_memory_budget:
                evicted = False
                for candidate in sorted(fast_mem, key=lambda x: (remaining_uses.get(x, 0), x)):
                    if candidate != out and candidate not in outputs:
                        fast_mem.remove(candidate)
                        operations.append(['store', candidate])
                        evicted = True
                        break
                if not evicted:
                    break
            operations.append(['load', out])
            fast_mem.add(out)
    
    return operations
# EVOLVE-BLOCK-END

