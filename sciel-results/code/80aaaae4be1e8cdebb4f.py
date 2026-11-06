import sys
from collections import deque

def pebble(dag, fast_memory_budget):
    n = len(dag)
    if n == 0:
        return []
    
    # Build graph
    children = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i, node in enumerate(dag):
        parents = node.get('parents', [])
        in_degree[i] = len(parents)
        for p in parents:
            children[p].append(i)
            
    is_sink = [len(children[i]) == 0 for i in range(n)]
    is_input = [len(dag[i].get('parents', [])) == 0 for i in range(n)]
    
    # Topological Sort
    q = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    indeg = in_degree[:]
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
                
    if len(topo_order) != n:
        topo_order = list(range(n))
        
    topo_index = {node: idx for idx, node in enumerate(topo_order)}
    
    # Pre-compute need_in_memory_until
    need_in_memory_until = [0] * n
    for u in range(n):
        if is_sink[u]:
            need_in_memory_until[u] = topo_index[u]
        else:
            max_child_idx = 0
            for v in children[u]:
                if topo_index[v] > max_child_idx:
                    max_child_idx = topo_index[v]
            need_in_memory_until[u] = max_child_idx
            
    memory = set()
    ops = []
    
    for v in topo_order:
        parents = dag[v].get('parents', [])
        
        if not parents:
            continue
            
        # Load parents
        for p in parents:
            if p not in memory:
                memory.add(p)
                ops.append(['load', p])
                
        # Evict if necessary
        if len(memory) >= fast_memory_budget:
            parent_set = set(parents)
            candidates = [u for u in memory if u not in parent_set]
            
            if candidates:
                # Evict node with furthest need_in_memory_until
                u_to_evict = max(candidates, key=lambda x: need_in_memory_until[x])
                
                # If it's an input, or if it's not needed in the future, just delete.
                # If it's needed in the future, store it first.
                if is_input[u_to_evict] or need_in_memory_until[u_to_evict] < topo_index[v]:
                    memory.remove(u_to_evict)
                    ops.append(['delete', u_to_evict])
                else:
                    memory.remove(u_to_evict)
                    ops.append(['store', u_to_evict])
                    ops.append(['delete', u_to_evict])
                    
        # Compute
        memory.add(v)
        ops.append(['compute', v])
        
        # If sink, store and delete
        if is_sink[v]:
            ops.append(['store', v])
            memory.remove(v)
            ops.append(['delete', v])
            
    return ops
