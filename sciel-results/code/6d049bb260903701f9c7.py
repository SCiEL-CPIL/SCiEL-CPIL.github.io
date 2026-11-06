import sys
import time
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    sys.setrecursionlimit(10000)
    start = time.time()
    deadline = start + 85.0

    # 1. Parse DAG
    vertices = []
    parents_map = {}
    children_map = defaultdict(list)
    inputs = set()
    outputs = set()

    if isinstance(dag, dict):
        for v, ps in dag.items():
            if v not in parents_map:
                vertices.append(v)
                parents_map[v] = ps
                for p in ps:
                    children_map[p].append(v)
                    if p not in parents_map:
                        parents_map[p] = []
                        vertices.append(p)
            else:
                # Already seen, but ensure children are updated?
                # If dag is {v: [p1, p2]}, and we see v again, we should merge?
                # Unlikely in generated instances.
                pass
        for v in vertices:
            if not parents_map[v]:
                inputs.add(v)
            if not children_map[v]:
                outputs.add(v)
    elif isinstance(dag, list):
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v = item[0]
                ps = item[1]
                if v not in parents_map:
                    vertices.append(v)
                    parents_map[v] = ps
                    for p in ps:
                        children_map[p].append(v)
                        if p not in parents_map:
                            parents_map[p] = []
                            vertices.append(p)
        for v in vertices:
            if not parents_map[v]:
                inputs.add(v)
            if not children_map[v]:
                outputs.add(v)
    else:
        return []

    vertices = list(set(vertices))
    if not vertices:
        return []

    # 2. Topological Sort
    indeg = {v: len(parents_map[v]) for v in vertices}
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    indeg_temp = dict(indeg)
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children_map[v]:
            indeg_temp[c] -= 1
            if indeg_temp[c] == 0:
                q.append(c)
        
    if len(topo) != len(vertices):
        remaining = [v for v in vertices if v not in set(topo)]
        topo.extend(remaining)

    topo_index = {v: i for i, v in enumerate(topo)}
    
    remaining_children = {v: 0 for v in vertices}
    for v in vertices:
        for c in children_map[v]:
            if topo_index[c] > topo_index[v]:
                remaining_children[v] += 1

    budget = max(1, int(fast_memory_budget))
    
    fast = set()
    in_slow = set(inputs)
    transfers = 0
    ops = []
    
    depth = {v: 0 for v in vertices}
    for v in topo:
        if parents_map[v]:
            depth[v] = max(depth[p] for p in parents_map[v]) + 1
        else:
            depth[v] = 0

    def maybe_evict():
        nonlocal transfers
        while len(fast) > budget:
            if not fast:
                break
            
            best_v = None
            best_score = None
            
            for v in fast:
                rc = remaining_children[v]
                if rc == 0:
                    score = (0, 0, 0)
                else:
                    in_slow_flag = 0 if v in in_slow else 1
                    score = (1, in_slow_flag, depth[v])
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_v = v
            
            if best_v is None:
                break
            
            fast.remove(best_v)
            
            if remaining_children[best_v] > 0 or best_v in outputs:
                ops.append(['store', best_v])
                transfers += 1
                in_slow.add(best_v)

    def ensure(v):
        nonlocal transfers
        if v in fast:
            return
        
        if v in inputs:
            ops.append(['load', v])
            transfers += 1
            fast.add(v)
            maybe_evict()
        else:
            if v in in_slow:
                ops.append(['load', v])
                transfers += 1
                fast.add(v)
                maybe_evict()
            else:
                for p in parents_map[v]:
                    ensure(p)
                ops.append(['compute', v])
                fast.add(v)
                in_slow.add(v)
                maybe_evict()

    for v in topo:
        if time.time() > deadline:
            break
        ensure(v)
        
    for o in outputs:
        if o not in fast:
            ensure(o)
            
    return ops
