# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55.0
    max_ops = 20000
    
    n = len(dag)
    if n == 0:
        return []
        
    is_input = [len(dag[i]) == 0 for i in range(n)]
    
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in dag[i]:
            children[p].append(i)
            
    is_output = [len(children[i]) == 0 for i in range(n)]
    
    next_direct_use = [float('inf')] * n
    for i in range(n):
        if children[i]:
            next_direct_use[i] = min(children[i])
    
    in_memory = [False] * n
    operations = []
    memory_set = set()
    op_count = 0
    
    def add_op(op, v):
        nonlocal op_count
        if op_count >= max_ops:
            raise TimeoutError("Max ops exceeded")
        operations.append([op, v])
        op_count += 1

    def evict_one(exclude_set):
        candidates = []
        for v in memory_set:
            if v in exclude_set:
                continue
            if is_output[v]:
                continue
            candidates.append(v)
            
        if not candidates:
            return None
            
        def get_sort_key(v):
            is_inp = is_input[v]
            next_use = next_direct_use[v]
            return (is_inp, -next_use)
            
        candidates.sort(key=get_sort_key)
        
        v_to_evict = candidates[0]
        add_op("delete", v_to_evict)
        in_memory[v_to_evict] = False
        memory_set.remove(v_to_evict)
        return v_to_evict

    def ensure_in_memory(v):
        if in_memory[v]:
            return
        
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
            
        if is_input[v]:
            add_op("load", v)
            in_memory[v] = True
            memory_set.add(v)
        else:
            for p in dag[v]:
                ensure_in_memory(p)
            
            add_op("compute", v)
            in_memory[v] = True
            memory_set.add(v)
            
            if is_output[v]:
                add_op("store", v)
                in_memory[v] = False
                memory_set.remove(v)
            else:
                if len(memory_set) > fast_memory_budget:
                    # Exclude parents of v and v itself
                    exclude = set(dag[v])
                    exclude.add(v)
                    evict_one(exclude)

    for v in range(n):
        if time.time() > deadline or op_count >= max_ops:
            break
        ensure_in_memory(v)

    return operations
# EVOLVE-BLOCK-END
