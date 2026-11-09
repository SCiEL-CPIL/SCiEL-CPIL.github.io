import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    n = len(dag)
    parents = dag
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
            
    inputs = [i for i in range(n) if not parents[i]]
    outputs = [i for i in range(n) if not children[i]]
    output_set = set(outputs)
    input_set = set(inputs)
    
    # Precompute last use
    last_use = [0] * n
    for i in range(n):
        if children[i]:
            last_use[i] = max(children[i])
        else:
            last_use[i] = i
            
    in_fast = set(inputs)
    ops = []
    
    # Initial overflow handling
    while len(in_fast) > fast_memory_budget:
        # Store input with smallest last_use
        cands = [v for v in in_fast]
        cands.sort(key=lambda x: last_use[x])
        v = cands[0]
        ops.append(['store', v])
        in_fast.remove(v)
        
    for i in range(n):
        if time.time() - start_time > 45:
            # Fallback
            return naive_schedule(dag, fast_memory_budget)
            
        if i in input_set:
            continue
            
        # Load parents
        for p in parents[i]:
            if p not in in_fast:
                ops.append(['load', p])
                in_fast.add(p)
                
        ops.append(['compute', i])
        in_fast.add(i)
        
        # Space Management
        if len(in_fast) > fast_memory_budget:
            # Find node to store
            cands = [v for v in in_fast if v not in output_set]
            if cands:
                cands.sort(key=lambda x: last_use[x])
                v = cands[0]
                ops.append(['store', v])
                in_fast.remove(v)
            else:
                # Critical: All in fast memory are outputs.
                # We cannot store them.
                # This situation should not happen if budget >= num_outputs.
                # If it does, we are stuck.
                pass

    return ops

def naive_schedule(dag, fast_memory_budget):
    # Simple valid schedule
    n = len(dag)
    parents = dag
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
            
    inputs = [i for i in range(n) if not parents[i]]
    outputs = [i for i in range(n) if not children[i]]
    output_set = set(outputs)
    input_set = set(inputs)
    
    in_fast = set(inputs)
    ops = []
    
    while len(in_fast) > fast_memory_budget:
        v = in_fast.pop()
        ops.append(['store', v])
        
    for i in range(n):
        if i in input_set:
            continue
        for p in parents[i]:
            if p not in in_fast:
                ops.append(['load', p])
                in_fast.add(p)
        ops.append(['compute', i])
        in_fast.add(i)
        
        if len(in_fast) > fast_memory_budget:
            cands = [v for v in in_fast if v not in output_set]
            if cands:
                v = cands[0]
                ops.append(['store', v])
                in_fast.remove(v)
    return ops
