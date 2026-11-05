import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0
    
    if not dag:
        return []
        
    n = len(dag)
    parents = dag
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
            
    is_input = [len(parents[i]) == 0 for i in range(n)]
    is_output = [len(children[i]) == 0 for i in range(n)]
    
    fast_set = set()
    computed_set = set(i for i in range(n) if is_input[i])
    operations = []
    max_ops = 20000
    
    def add_op(op, v):
        if len(operations) < max_ops:
            operations.append([op, v])
            
    def evict_if_needed(current_idx):
        while len(fast_set) > fast_memory_budget:
            if len(operations) >= max_ops:
                break
            candidates = [v for v in fast_set if not is_output[v]]
            if not candidates:
                candidates = list(fast_set)
            
            if not candidates:
                break
                
            def score(v):
                rem = 0
                for c in children[v]:
                    if c > current_idx:
                        rem += 1
                return (rem, v)
                
            victim = min(candidates, key=score)
            fast_set.remove(victim)
            add_op("store", victim)
            
    for i in range(n):
        if time.time() > deadline or len(operations) >= max_ops:
            break
            
        if i in fast_set:
            continue
            
        if i in computed_set:
            add_op("load", i)
            fast_set.add(i)
        else:
            for p in parents[i]:
                if p not in fast_set:
                    if p in computed_set:
                        add_op("load", p)
                        fast_set.add(p)
            
            add_op("compute", i)
            fast_set.add(i)
            computed_set.add(i)
            
        evict_if_needed(i)
        
    return operations
