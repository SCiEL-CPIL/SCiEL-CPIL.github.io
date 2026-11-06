import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 55.0  # 55 seconds to be safe
    
    # Parse DAG
    if isinstance(dag, list):
        dag_dict = {}
        for item in dag:
            if isinstance(item, tuple):
                v, parents = item
                dag_dict[v] = parents
            else:
                v, parents = item
                dag_dict[v] = parents
    else:
        dag_dict = dag

    all_vertices = set(dag_dict.keys())
    parent_set = set()
    for v, parents in dag_dict.items():
        for p in parents:
            parent_set.add(p)
    
    inputs = [v for v in all_vertices if v not in parent_set]
    outputs = [v for v in all_vertices if v not in parent_set]
    
    fast_mem = set(inputs)
    slow_mem = set()
    ops = []
    
    remaining_children = {v: 0 for v in all_vertices}
    for v, parents in dag_dict.items():
        for p in parents:
            remaining_children[p] += 1
            
    def ensure_blue(v):
        nonlocal fast_mem, slow_mem, ops, remaining_children
        
        if time.time() - start_time > deadline:
            return
        
        if v in fast_mem:
            return
            
        if v in slow_mem:
            ops.append(['load', v])
            fast_mem.add(v)
            slow_mem.remove(v)
            return
            
        parents = dag_dict[v]
        for p in parents:
            ensure_blue(p)
            
        if len(fast_mem) >= fast_memory_budget:
            evict_candidate = None
            for p in parents:
                if remaining_children[p] == 0:
                    evict_candidate = p
                    break
            
            if evict_candidate is None:
                for u in fast_mem:
                    if u != v:
                        if remaining_children[u] == 0:
                            evict_candidate = u
                            break
            
            if evict_candidate is not None:
                ops.append(['delete', evict_candidate])
                fast_mem.remove(evict_candidate)
                if evict_candidate in slow_mem:
                    slow_mem.remove(evict_candidate)
            else:
                stored = False
                for p in parents:
                    if p in fast_mem:
                        ops.append(['store', p])
                        fast_mem.remove(p)
                        slow_mem.add(p)
                        stored = True
                        break
                
                if not stored:
                    for u in fast_mem:
                        if u != v:
                            ops.append(['store', u])
                            fast_mem.remove(u)
                            slow_mem.add(u)
                            break
        
        ops.append(['compute', v])
        fast_mem.add(v)
        
        for p in parents:
            remaining_children[p] -= 1

    for v in outputs:
        ensure_blue(v)
        
    return ops
