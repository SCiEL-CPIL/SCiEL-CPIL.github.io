"""Communication-minimizing DAG pebbling with smart memory management."""
import time
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    
    all_vertices = set(parents.keys())
    for v in parents:
        for p in parents[v]:
            all_vertices.add(p)
    if outputs:
        all_vertices.update(outputs)
    
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    inputs = set(v for v in all_vertices if not parents.get(v, []))
    
    # Topological order
    in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for c in children[u]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(all_vertices):
        topo_order = list(all_vertices)
    
    # Use count: number of children (total uses)
    use_count = {v: len(children.get(v, [])) for v in all_vertices}
    
    # State
    fast = set()      # vertices in fast memory (red)
    slow = set(inputs)  # vertices in slow memory (blue), available to load
    
    ops = []
    
    def ensure_red(target):
        """Ensure target is in fast memory. Mutates fast, slow, ops."""
        nonlocal fast, slow, ops
        
        if target in fast:
            return True
        
        if time.time() > DEADLINE:
            return False
        
        if target in slow:
            ops.append(['load', target])
            fast.add(target)
            slow.discard(target)
            return True
        
        # Need to recompute
        target_parents = parents.get(target, [])
        for p in target_parents:
            if not ensure_red(p):
                return False
        
        if time.time() > DEADLINE:
            return False
        
        # Make room in fast memory if needed
        while len(fast) >= fast_memory_budget and target not in fast:
            # Delete a vertex that's not an output and has low future use
            best_del = None
            best_score = float('inf')
            for cand in fast:
                if cand == target:
                    continue
                if cand in outputs:
                    score = 1000 + use_count.get(cand, 0)
                else:
                    score = use_count.get(cand, 0)
                if score < best_score:
                    best_score = score
                    best_del = cand
            
            if best_del is None:
                break
            
            ops.append(['delete', best_del])
            fast.discard(best_del)
        
        ops.append(['compute', target])
        fast.add(target)
        return True
    
    # Process in topological order
    for v in topo_order:
        if time.time() > DEADLINE:
            break
        
        v_parents = parents.get(v, [])
        for p in v_parents:
            if not ensure_red(p):
                break
        
        if time.time() > DEADLINE:
            break
        
        # Make room for v if needed
        while len(fast) >= fast_memory_budget and v not in fast:
            best_del = None
            best_score = float('inf')
            for cand in fast:
                if cand == v:
                    continue
                if cand in outputs:
                    score = 1000 + use_count.get(cand, 0)
                else:
                    score = use_count.get(cand, 0)
                if score < best_score:
                    best_score = score
                    best_del = cand
            
            if best_del is None:
                break
            
            ops.append(['delete', best_del])
            fast.discard(best_del)
        
        ops.append(['compute', v])
        fast.add(v)
    
    # Store all outputs
    for v in outputs:
        if time.time() > DEADLINE:
            break
        if v in fast:
            ops.append(['store', v])
            fast.discard(v)
            slow.add(v)
        else:
            if ensure_red(v):
                ops.append(['store', v])
                fast.discard(v)
                slow.add(v)
    
    # Truncate if needed
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
