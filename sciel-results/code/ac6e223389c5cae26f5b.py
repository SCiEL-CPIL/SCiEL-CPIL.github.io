# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 55.0
    
    n_vertices = len(dag)
    if n_vertices == 0:
        return []
        
    parents = [set() for _ in range(n_vertices)]
    children = [set() for _ in range(n_vertices)]
    is_output = [False] * n_vertices
    
    # Parse DAG
    first_item = dag[0]
    if isinstance(first_item, (list, tuple, set)):
        for i in range(n_vertices):
            p_list = dag[i]
            parents[i] = set(p_list)
            for p in parents[i]:
                children[p].add(i)
        for i in range(n_vertices):
            if not children[i]:
                is_output[i] = True
    elif isinstance(first_item, tuple) and len(first_item) >= 3:
        for i in range(n_vertices):
            item = dag[i]
            parents[i] = set(item[1])
            is_output[i] = bool(item[2])
            for p in parents[i]:
                children[p].add(i)
    elif isinstance(first_item, dict):
        for i in range(n_vertices):
            item = dag[i]
            parents[i] = set(item.get('parents', []))
            is_output[i] = bool(item.get('output', False))
            for p in parents[i]:
                children[p].add(i)
    else:
        for i in range(n_vertices):
            if isinstance(dag[i], (list, tuple, set)):
                parents[i] = set(dag[i])
                for p in parents[i]:
                    children[p].add(i)
        for i in range(n_vertices):
            if not children[i]:
                is_output[i] = True

    # Topological Sort
    in_degree = [len(parents[i]) for i in range(n_vertices)]
    queue = deque([i for i in range(n_vertices) if in_degree[i] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != n_vertices:
        return []
        
    topo_index = {v: i for i, v in enumerate(topo_order)}
    
    operations = []
    fast_memory = set()
    slow_memory = set()
    
    # Inputs start blue
    for v in range(n_vertices):
        if not parents[v]:
            fast_memory.add(v)
            
    remaining_children = [len(children[v]) for v in range(n_vertices)]
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
            
        if not parents[v]:
            continue
            
        # Ensure parents in fast memory
        for p in parents[v]:
            if p not in fast_memory:
                if p in slow_memory:
                    operations.append(['load', p])
                    fast_memory.add(p)
                    slow_memory.discard(p)
                # If not in slow, it was deleted. This is an error in our logic.
                # We assume it doesn't happen.
        
        operations.append(['compute', v])
        fast_memory.add(v)
        
        for p in parents[v]:
            remaining_children[p] -= 1
            
        # Eviction
        while len(fast_memory) > fast_memory_budget:
            # Find candidate
            # 1. Delete: rem_children == 0, not output
            # 2. Store: rem_children == 0, is output
            # 3. Store: rem_children > 0
            
            best_candidate = None
            best_type = 0 # 1: delete, 2: store output, 3: store live
            
            for u in fast_memory:
                if remaining_children[u] == 0 and not is_output[u]:
                    # Delete candidate
                    if best_type == 0 or (best_type == 1 and topo_index[u] > topo_index[best_candidate]):
                        best_candidate = u
                        best_type = 1
                elif remaining_children[u] == 0 and is_output[u]:
                    # Store output candidate
                    if best_type == 0 or (best_type == 2 and topo_index[u] > topo_index[best_candidate]):
                        best_candidate = u
                        best_type = 2
                else:
                    # Store live candidate
                    if best_type == 0 or (best_type == 3 and topo_index[u] > topo_index[best_candidate]):
                        best_candidate = u
                        best_type = 3
                        
            if best_candidate is None:
                break
                
            if best_type == 1:
                operations.append(['delete', best_candidate])
                fast_memory.discard(best_candidate)
            else:
                operations.append(['store', best_candidate])
                fast_memory.discard(best_candidate)
                slow_memory.add(best_candidate)
                
    # Final: Load outputs
    for v in range(n_vertices):
        if is_output[v] and v not in fast_memory:
            if v in slow_memory:
                operations.append(['load', v])
                fast_memory.add(v)
                slow_memory.discard(v)
                
    return operations

# EVOLVE-BLOCK-END
