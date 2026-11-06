import time
import heapq
from collections import deque

def _count_transfers(ops):
    c = 0
    for op in ops:
        if op and op[0] in ('load', 'store'):
            c += 1
    return c

def _extract(dag):
    def get(d, k, default=None):
        if isinstance(d, dict):
            return d.get(k, default)
        return getattr(d, k, default)

    vertices = get(dag, 'vertices')
    parents = get(dag, 'parents')
    inputs = get(dag, 'inputs')
    outputs = get(dag, 'outputs')

    if vertices is None and isinstance(dag, dict):
        vertices = list(dag.keys())
        parents = {}
        for k, v in dag.items():
            if isinstance(v, (list, tuple, set)):
                parents[k] = list(v)
            else:
                parents[k] = []

    if vertices is None:
        return None
    if parents is None:
        parents = {}
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []
    return vertices, parents, inputs, outputs

def _verify(ops, vertices, parents, inputs, outputs, budget):
    vset = set(vertices)
    in_mem = set()
    for v in inputs:
        if v in vset:
            in_mem.add(v)
    for op in ops:
        if not op or len(op) < 2:
            return False
        name, v = op[0], op[1]
        if v not in vset:
            return False
        if name == 'load':
            if v not in in_mem:
                in_mem.add(v)
        elif name == 'store':
            if v not in in_mem:
                return False
            in_mem.discard(v)
        elif name == 'delete':
            if v not in in_mem:
                return False
            in_mem.discard(v)
        elif name == 'compute':
            for p in parents.get(v, []):
                if p not in in_mem:
                    return False
            in_mem.add(v)
        else:
            return False
        if len(in_mem) > budget:
            return False
    for v in outputs:
        if v in vset and v not in in_mem:
            return False
    return True

def _greedy_last_use_schedule(dag, fast_memory_budget, deadline):
    ex = _extract(dag)
    if ex is None:
        return None
    vertices, parents, inputs, outputs = ex
    n = len(vertices)
    if n == 0:
        return []
    vindex = {v: i for i, v in enumerate(vertices)}

    children = {v: [] for v in vertices}
    indeg = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in vindex:
                children[p].append(v)
                indeg[v] += 1

    indeg2 = dict(indeg)
    q = deque([v for v in vertices if indeg2[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    for v in vertices:
        if v not in topo:
            topo.append(v)
    pos = {v: i for i, v in enumerate(topo)}
    
    last_use = {}
    for v in vertices:
        max_child_pos = -1
        for c in children[v]:
            if pos.get(c, -1) > max_child_pos:
                max_child_pos = pos[c]
        last_use[v] = max_child_pos

    outputs_set = set(v for v in outputs if v in vindex)
    
    try:
        budget = max(0, int(fast_memory_budget))
    except Exception:
        budget = 0

    ops = []
    in_mem = set()
    for v in inputs:
        if v in vindex:
            in_mem.add(v)
            
    heap_non_out = []
    heap_out = []
    
    def push_heap(v):
        # We want to evict MAX last_use.
        # Min-heap of -last_use.
        lu = -last_use[v]
        if v in outputs_set:
            heapq.heappush(heap_out, (lu, v))
        else:
            heapq.heappush(heap_non_out, (lu, v))
            
    for v in in_mem:
        push_heap(v)
        
    for v in topo:
        if time.time() > deadline:
            break
            
        ps = [p for p in parents.get(v, []) if p in vindex]
        for p in ps:
            if p not in in_mem:
                ops.append(['load', p])
                in_mem.add(p)
                push_heap(p)
                
        ops.append(['compute', v])
        if v not in in_mem:
            in_mem.add(v)
            push_heap(v)
            
        while len(in_mem) > budget:
            victim = None
            # Try to evict non-output first
            while heap_non_out:
                lu, cand = heapq.heappop(heap_non_out)
                if cand in in_mem:
                    victim = cand
                    break
            if victim is None:
                while heap_out:
                    lu, cand = heapq.heappop(heap_out)
                    if cand in in_mem:
                        victim = cand
                        break
                        
            if victim is None:
                break # Should not happen
                
            ops.append(['store', victim])
            in_mem.discard(victim)
            
    for v in outputs:
        if v in vindex and v not in in_mem:
            ops.append(['load', v])
            in_mem.add(v)
            
    return ops

def pebble(dag, fast_memory_budget):
    start = time.time()
    deadline = start + 80.0
    
    my_ops = None
    try:
        m = _greedy_last_use_schedule(dag, fast_memory_budget, deadline)
        if m is not None:
            ex = _extract(dag)
            if ex is not None:
                vertices, parents, inputs, outputs = ex
                try:
                    budget = max(0, int(fast_memory_budget))
                except Exception:
                    budget = 0
                if _verify(m, vertices, parents, inputs, outputs, budget):
                    my_ops = m
    except Exception:
        my_ops = None
        
    if my_ops is not None:
        return my_ops
    return []
