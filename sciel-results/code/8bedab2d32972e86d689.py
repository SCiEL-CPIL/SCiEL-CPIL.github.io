import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Parse DAG
    nodes = []
    parents_map = {}
    
    if isinstance(dag, dict):
        for node, parents in dag.items():
            nodes.append(node)
            parents_map[node] = parents
    elif isinstance(dag, list):
        for item in dag:
            if isinstance(item, tuple) and len(item) == 2:
                node, parents = item
                nodes.append(node)
                parents_map[node] = parents
            elif isinstance(item, dict):
                node = item.get('node')
                parents = item.get('parents', [])
                nodes.append(node)
                parents_map[node] = parents

    children_map = {n: [] for n in nodes}
    for n in nodes:
        for p in parents_map.get(n, []):
            children_map[p].append(n)
            
    is_input = set()
    is_output = set()
    for n in nodes:
        if not parents_map.get(n, []):
            is_input.add(n)
        if not children_map[n]:
            is_output.add(n)
            
    in_degree = {n: len(parents_map.get(n, [])) for n in nodes}
    queue = [n for n in nodes if in_degree[n] == 0]
    topo_order = []
    while queue:
        queue.sort()
        n = queue.pop(0)
        topo_order.append(n)
        for c in children_map[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    red = set()
    ops = []
    max_ops = 19000
    
    def is_deadline_exceeded():
        return time.time() > deadline

    def evict(victim):
        if victim in red:
            ops.append(['store', victim])
            red.remove(victim)
            
    def load(v):
        ops.append(['load', v])
        red.add(v)
        
    def compute(v):
        ops.append(['compute', v])
        red.add(v)
        
    def pebble_vertex(v):
        if is_deadline_exceeded() or len(ops) >= max_ops:
            return False
        if v in red:
            return True
            
        if v in is_input:
            if len(red) == fast_memory_budget:
                victim = None
                for u in list(red):
                    if u not in parents_map.get(v, []):
                        victim = u
                        break
                if victim is None:
                    victim = min(red, key=lambda u: len(children_map[u]))
                evict(victim)
            load(v)
            return True
            
        for p in parents_map.get(v, []):
            if p not in red:
                if not pebble_vertex(p):
                    return False
                if is_deadline_exceeded() or len(ops) >= max_ops:
                    return False
        
        if v not in red:
            if len(red) == fast_memory_budget:
                victim = None
                for u in list(red):
                    if u not in parents_map.get(v, []):
                        victim = u
                        break
                if victim is None:
                    victim = min(red, key=lambda u: len(children_map[u]))
                evict(victim)
            compute(v)
            
        if v in is_output:
            if v in red:
                evict(v)
            return True
            
        return True

    for v in topo_order:
        if not pebble_vertex(v):
            break
            
    return ops
