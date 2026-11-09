import time
import sys
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    # Parse DAG
    # Assuming standard keys based on prompt: 'nodes', 'parents', 'inputs', 'outputs'
    # Fallbacks for other common keys
    nodes = dag.get('nodes', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # If nodes is empty, try to infer from parents/children
    if not nodes:
        all_nodes = set()
        for n, ps in parents.items():
            all_nodes.add(n)
            all_nodes.update(ps)
        nodes = list(all_nodes)
        if not nodes:
            return []

    # Ensure all nodes in parents are in nodes list
    node_set = set(nodes)
    for n, ps in parents.items():
        for p in ps:
            if p not in node_set:
                nodes.append(p)
                node_set.add(p)
    
    # 1. Topological Sort
    in_degree = {n: 0 for n in nodes}
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            if p in in_degree:
                in_degree[n] += 1
                children[p].append(n)
            else:
                # Parent not in node list? Add it? 
                # Assuming well-formed DAG, parents should be in nodes.
                pass

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
        # Cycle or disconnected? Fallback to simple list, but this might fail verification.
        # Try to append remaining nodes
        remaining = [n for n in nodes if n not in set(topo_order)]
        topo_order.extend(remaining)

    # 2. Precompute Uses for Eviction Strategy
    # Map node to list of indices where it is a parent
    uses_by_node = {n: [] for n in nodes}
    for i, u in enumerate(topo_order):
        for p in parents.get(u, []):
            if p in uses_by_node:
                uses_by_node[p].append(i)
            else:
                uses_by_node[p] = [i]
                
    # 3. Simulation
    fast_mem = set()
    main_mem = set(inputs) 
    operations = []
    
    # Pointers for next use
    use_ptr = {n: 0 for n in nodes}
    
    def get_next_use(n, current_index):
        lst = uses_by_node.get(n, [])
        ptr = use_ptr[n]
        while ptr < len(lst) and lst[ptr] <= current_index:
            ptr += 1
        use_ptr[n] = ptr
        if ptr < len(lst):
            return lst[ptr]
        return float('inf')

    for i, v in enumerate(topo_order):
        if time.time() > deadline:
            break
            
        # Ensure parents are in fast_mem
        for p in parents.get(v, []):
            if p not in fast_mem:
                if p in main_mem:
                    operations.append(('load', p))
                    fast_mem.add(p)
                else:
                    # If parent is not in main_mem, we have a problem.
                    # This implies the parent was deleted without being stored, or never stored.
                    # In a correct DAG pebbling, if a parent is needed, it must be available.
                    # If it's not in main_mem, it might have been computed and deleted.
                    # We cannot recover it. This leads to invalid state.
                    # To avoid crashing, we skip or mark error. 
                    # But for now, assume well-formed execution keeps parents in main_mem if needed.
                    pass
        
        # Compute v
        operations.append(('compute', v))
        fast_mem.add(v)
        
        # Post-compute action
        if v in outputs:
            # Must finish in main memory. Store immediately to ensure it's safe.
            operations.append(('store', v))
            fast_mem.remove(v)
            main_mem.add(v)
        else:
            # Intermediate node
            # Evict if over budget
            while len(fast_mem) > fast_memory_budget:
                if not fast_mem:
                    break
                # Select victim: farthest next use or no future use
                best_victim = None
                best_score = -1
                
                for u in fast_mem:
                    nu = get_next_use(u, i)
                    if nu == float('inf'):
                        score = float('inf')
                    else:
                        score = nu - i
                        
                    if score > best_score:
                        best_score = score
                        best_victim = u
                        
                if best_victim is not None:
                    if best_score == float('inf'):
                        operations.append(('delete', best_victim))
                    else:
                        operations.append(('store', best_victim))
                        main_mem.add(best_victim)
                    fast_mem.remove(best_victim)

    # Final Check: Ensure all outputs are in main memory
    # If an output is still in fast_mem (shouldn't happen if we store immediately, 
    # but in case of timeout or logic error), store it.
    for v in outputs:
        if v in fast_mem:
            operations.append(('store', v))
            fast_mem.remove(v)
            main_mem.add(v)
            
    return operations
