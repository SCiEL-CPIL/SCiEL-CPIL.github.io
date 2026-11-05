import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    nodes = list(dag.keys())
    N = len(nodes)
    
    children = {i: [] for i in nodes}
    for u, parents in dag.items():
        for p in parents:
            children[p].append(u)
            
    for u in children:
        children[u].sort()
        
    inputs = set(u for u in nodes if len(dag[u]) == 0)
    outputs = set(u for u in nodes if len(children[u]) == 0)
    
    fast_mem = set()
    slow_mem = set()
    child_idx = {u: 0 for u in nodes}
    
    ops = []
    transfers = 0
    
    def get_next_use(u):
        if child_idx[u] < len(children[u]):
            return children[u][child_idx[u]]
        return float('inf')
        
    for u in range(N):
        if time.time() - start_time > 85:
            break
            
        if u in inputs:
            if u not in fast_mem:
                ops.append(['load', u])
                transfers += 1
                fast_mem.add(u)
                if u in slow_mem:
                    slow_mem.remove(u)
        else:
            for p in dag[u]:
                if p not in fast_mem:
                    if p in slow_mem:
                        ops.append(['load', p])
                        transfers += 1
                        fast_mem.add(p)
                        slow_mem.remove(p)
                    else:
                        ops.append(['load', p])
                        transfers += 1
                        fast_mem.add(p)
            
            ops.append(['compute', u])
            fast_mem.add(u)
            if u in slow_mem:
                slow_mem.remove(u)
            
        for p in dag[u]:
            child_idx[p] += 1
            
        while len(fast_mem) > fast_memory_budget:
            max_next = -1
            victim = -1
            for v in fast_mem:
                nu = get_next_use(v)
                if nu > max_next:
                    max_next = nu
                    victim = v
            
            if victim == -1:
                break
                
            if max_next == float('inf'):
                if victim in outputs:
                    ops.append(['store', victim])
                    transfers += 1
                    fast_mem.remove(victim)
                    slow_mem.add(victim)
                else:
                    ops.append(['delete', victim])
                    fast_mem.remove(victim)
            else:
                ops.append(['store', victim])
                transfers += 1
                fast_mem.remove(victim)
                slow_mem.add(victim)
                
    for u in outputs:
        if u in fast_mem:
            ops.append(['store', u])
            transfers += 1
            fast_mem.remove(u)
            slow_mem.add(u)
            
    return ops
