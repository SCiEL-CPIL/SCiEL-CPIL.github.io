"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    # Parse DAG structure robustly
    n = None
    parents_map = None
    outputs_set = None
    
    if isinstance(dag, dict):
        # Try various key names
        for key in ['n', 'num_vertices', 'num_nodes', 'V', 'num_v']:
            if key in dag:
                n = int(dag[key])
                break
        
        if n is None and 'vertices' in dag:
            n = len(dag['vertices'])
        
        # Get parents
        for key in ['parents', 'preds', 'predecessors', 'in_edges']:
            if key in dag:
                parents_map = dag[key]
                break
        
        if parents_map is None and 'edges' in dag:
            parents_map = {}
            for edge in dag['edges']:
                if len(edge) >= 2:
                    u, v = int(edge[0]), int(edge[1])
                    if v not in parents_map:
                        parents_map[v] = []
                    parents_map[v].append(u)
        
        # Get outputs
        for key in ['outputs', 'output_vertices', 'sinks', 'targets']:
            if key in dag:
                outputs_set = set(int(x) for x in dag[key])
                break
    
    elif isinstance(dag, (list, tuple)):
        n = len(dag)
        parents_map = dag
        outputs_set = None
    
    # If n is still None, try to infer
    if n is None:
        if parents_map is not None:
            if isinstance(parents_map, dict):
                if parents_map:
                    n = max(parents_map.keys()) + 1
                else:
                    n = 0
            elif isinstance(parents_map, (list, tuple)):
                n = len(parents_map)
            else:
                n = 0
        else:
            return []
    
    if parents_map is None:
        parents_map = {i: [] for i in range(n)}
    
    # Normalize parents_map to dict
    if isinstance(parents_map, (list, tuple)):
        parents_dict = {}
        for i in range(n):
            if i < len(parents_map):
                parents_dict[i] = list(parents_map[i])
            else:
                parents_dict[i] = []
        parents_map = parents_dict
    else:
        # Ensure all vertices have entries
        parents_dict = {}
        for i in range(n):
            parents_dict[i] = list(parents_map.get(i, []))
        parents_map = parents_dict
    
    # Build children map
    children_map = {i: [] for i in range(n)}
    for v in range(n):
        for p in parents_map[v]:
            children_map[p].append(v)
    
    # Find inputs (no parents)
    inputs_set = set()
    for v in range(n):
        if not parents_map[v]:
            inputs_set.add(v)
    
    # Find outputs if not provided
    if outputs_set is None:
        outputs_set = set()
        for v in range(n):
            if not children_map[v]:
                outputs_set.add(v)
    
    # Topological sort
    in_degree = {v: len(parents_map[v]) for v in range(n)}
    queue = deque()
    for v in range(n):
        if in_degree[v] == 0:
            queue.append(v)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for c in children_map[u]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Handle cycles (shouldn't happen in DAG)
    if len(topo_order) < n:
        visited = set(topo_order)
        for v in range(n):
            if v not in visited:
                topo_order.append(v)
    
    # State tracking
    fast_mem = set()      # vertices in fast memory
    slow_stored = set()   # vertices stored in slow memory
    computed = set()      # vertices that have been computed
    
    # Remaining uses: how many children still need this vertex
    remaining_uses = {v: len(children_map[v]) for v in range(n)}
    
    operations = []
    
    def do_load(v):
        operations.append(['load', v])
        fast_mem.add(v)
        slow_stored.discard(v)
    
    def do_store(v):
        operations.append(['store', v])
        slow_stored.add(v)
        fast_mem.discard(v)
    
    def do_compute(v):
        operations.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
    
    def do_delete(v):
        operations.append(['delete', v])
        fast_mem.discard(v)
    
    # Initial: inputs are in fast memory (blue)
    for v in inputs_set:
        fast_mem.add(v)
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        if v in inputs_set:
            continue
        
        p_list = parents_map[v]
        
        # Ensure parents are in fast memory
        for p in p_list:
            if p not in fast_mem:
                if p in slow_stored or p in computed:
                    do_load(p)
                elif p in inputs_set:
                    fast_mem.add(p)
        
        # If fast memory is full, need to make space
        while len(fast_mem) >= fast_memory_budget and v not in fast_mem:
            # Find best vertex to evict
            # Prefer evicting vertices with fewest remaining uses
            # Don't evict parents of v (needed for compute)
            candidates = []
            for u in fast_mem:
                if u in p_list:
                    continue
                uses = remaining_uses.get(u, 0)
                # Outputs should be kept longer
                if u in outputs_set:
                    uses += 10000
                candidates.append((uses, u))
            
            if not candidates:
                # All fast memory is parents, force evict one parent
                for p in p_list:
                    if p in fast_mem:
                        do_store(p)
                        break
                else:
                    break
            else:
                candidates.sort()
                evict_v = candidates[0][1]
                do_store(evict_v)
        
        # Compute v
        do_compute(v)
        
        # Update remaining uses for parents
        for p in p_list:
            remaining_uses[p] = remaining_uses.get(p, 0) - 1
        
        # Decide whether to keep v in fast memory
        # Keep if it has remaining uses or is an output
        if remaining_uses.get(v, 0) > 0 or v in outputs_set:
            pass  # Keep in fast memory
        else:
            # No more uses, store to slow to free space
            if v in fast_mem:
                do_store(v)
    
    # Ensure all outputs end up in fast memory (blue)
    for v in outputs_set:
        if v not in fast_mem:
            if v in slow_stored or v in computed:
                do_load(v)
            elif v in inputs_set:
                fast_mem.add(v)
    
    # Clean up: remove any extra operations if over limit
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
