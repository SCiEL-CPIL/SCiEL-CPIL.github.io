# EVOLVE-BLOCK-START
import sys
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 45.0
    
    parents_map = dag['parents']
    outputs = set(dag['outputs'])
    all_vertices = list(dag['vertices'])
    
    children_map = defaultdict(list)
    for v in all_vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    in_degree = {v: len(parents_map.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    topo_pos = {v: i for i, v in enumerate(topo)}
    
    remaining_uses = {v: len(children_map.get(v, [])) for v in all_vertices}
    
    ops = []
    fast_mem = set()
    transfers = 0
    
    for idx, v in enumerate(topo):
        if time.time() > deadline:
            break
        
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                ops.append(['load', p])
                fast_mem.add(p)
                transfers += 1
        
        ops.append(['compute', v])
        fast_mem.add(v)
        
        for p in parents_map.get(v, []):
            remaining_uses[p] -= 1
        
        for p in parents_map.get(v, []):
            if remaining_uses[p] == 0 and p not in outputs and p in fast_mem:
                ops.append(['store', p])
                fast_mem.discard(p)
                transfers += 1
        
        while len(fast_mem) > fast_memory_budget:
            best_to_evict = None
            best_next_need = -1
            for fm_v in fast_mem:
                if fm_v in outputs:
                    continue
                next_need = float('inf')
                for c in children_map.get(fm_v, []):
                    pos = topo_pos.get(c, float('inf'))
                    if pos > idx and pos < next_need:
                        next_need = pos
                if best_to_evict is None or next_need > best_next_need:
                    best_to_evict = fm_v
                    best_next_need = next_need
            if best_to_evict is not None:
                ops.append(['store', best_to_evict])
                fast_mem.discard(best_to_evict)
                transfers += 1
            else:
                break
    
    for v in topo:
        if time.time() > deadline:
            break
        if v in outputs and v in fast_mem:
            ops.append(['store', v])
            fast_mem.discard(v)
            transfers += 1
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
