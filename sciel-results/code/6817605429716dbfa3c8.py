"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize communication (load/store operations).
    
    Returns: list of [operation, vertex]
    """
    start_time = time.time()
    DEADLINE = 85
    
    # Parse the DAG structure
    if isinstance(dag, dict):
        vertices = sorted(dag.keys())
        n = len(vertices)
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        parents_idx = [[vertex_to_idx[p] for p in dag[v]] for v in vertices]
    elif isinstance(dag, (list, tuple)):
        if len(dag) > 0:
            if isinstance(dag[0], (list, tuple)):
                n = len(dag)
                parents_idx = [list(p) for p in dag]
            else:
                n = len(dag)
                parents_idx = [[] for _ in range(n)]
        else:
            n = 0
            parents_idx = []
    else:
        n = 1
        parents_idx = [[]]
    
    if n == 0:
        return []
    
    # Build children adjacency
    children = defaultdict(list)
    for i in range(n):
        for p in parents_idx[i]:
            children[p].append(i)
    
    # Find output vertices (no children)
    output_vertices = set()
    for i in range(n):
        if len(children[i]) == 0:
            output_vertices.add(i)
    
    # Find input vertices (no parents)
    input_vertices = set()
    for i in range(n):
        if len(parents_idx[i]) == 0:
            input_vertices.add(i)
    
    # Compute topological order
    in_degree = [len(parents_idx[i]) for i in range(n)]
    topo_order = []
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != n:
        topo_order = list(range(n))
    
    # Compute next use for each vertex
    next_use = [n + 1] * n
    for i in range(n):
        for p in parents_idx[i]:
            if next_use[p] == n + 1:
                next_use[p] = i
    
    # If budget is large enough, simple optimal strategy
    if fast_memory_budget >= n:
        operations = []
        # Load all inputs
        for v in input_vertices:
            operations.append(['load', v])
        # Compute in topological order
        for v in topo_order:
            if v not in input_vertices:
                operations.append(['compute', v])
        # Store all outputs
        for v in output_vertices:
            operations.append(['store', v])
        return operations
    
    # Greedy pebbling with LRU-like eviction
    operations = []
    fast_memory = set()
    computed = set()
    
    def load(v):
        operations.append(['load', v])
        fast_memory.add(v)
    
    def store(v):
        operations.append(['store', v])
        fast_memory.discard(v)
    
    def delete(v):
        operations.append(['delete', v])
        fast_memory.discard(v)
    
    def compute(v):
        operations.append(['compute', v])
        fast_memory.add(v)
        computed.add(v)
    
    def evict_one(v):
        """Evict one vertex from fast memory to make space for v"""
        if len(fast_memory) < fast_memory_budget:
            return True
        
        candidates = []
        for u in fast_memory:
            if u in parents_idx[v]:
                # Penalize evicting parents we need
                if next_use[u] == n + 1:
                    priority = 500
                else:
                    priority = 1000 + next_use[u]
            else:
                if next_use[u] == n + 1:
                    priority = 0
                else:
                    priority = next_use[u]
            candidates.append((priority, u))
        
        if candidates:
            candidates.sort(reverse=True)
            _, u_to_evict = candidates[0]
            if next_use[u_to_evict] == n + 1 and u_to_evict not in parents_idx[v]:
                delete(u_to_evict)
            else:
                store(u_to_evict)
            return True
        return False
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in input_vertices:
            continue
        
        # Load parents not in fast memory
        for p in parents_idx[v]:
            if p not in fast_memory:
                evict_one(v)
                load(p)
        
        # Compute v
        compute(v)
        
        # Clean up if over budget
        while len(fast_memory) > fast_memory_budget:
            candidates = []
            for u in fast_memory:
                if u == v:
                    continue
                if next_use[u] == n + 1:
                    candidates.append((0, u))
                else:
                    candidates.append((next_use[u], u))
            
            if candidates:
                candidates.sort(reverse=True)
                _, u_to_evict = candidates[0]
                if next_use[u_to_evict] == n + 1:
                    delete(u_to_evict)
                else:
                    store(u_to_evict)
            else:
                break
    
    # Ensure outputs are blue
    for v in output_vertices:
        if v in fast_memory:
            store(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END

