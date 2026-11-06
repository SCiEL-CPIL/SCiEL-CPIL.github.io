"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict
import heapq

def _topological_sort(vertices, parents):
    """
    Returns a list of vertices in topological order.
    vertices: list of vertex IDs
    parents: dict mapping vertex -> list of parent IDs
    """
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in in_degree:
                    in_degree[v] += 1
                    children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(order) != len(vertices):
        return vertices
    return order

def _validate_schedule(dag, ops, fast_memory_budget):
    """
    Validates a schedule by simulating the state machine.
    Returns True if valid, False otherwise.
    """
    vertices = set(dag['vertices'])
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    blue = set(inputs)
    red = set()
    computed = set()
    
    for op, v in ops:
        if v not in vertices:
            return False
        
        if op == 'load':
            if v not in blue or v in red:
                return False
            if len(red) >= fast_memory_budget:
                return False
            red.add(v)
            blue.remove(v)
            
        elif op == 'store':
            if v not in red or v in blue:
                return False
            red.remove(v)
            blue.add(v)
            
        elif op == 'compute':
            if v in red:
                pass # Recompute allowed
            
            for p in parents.get(v, []):
                if p not in red:
                    return False
            
            if v not in red:
                if len(red) >= fast_memory_budget:
                    return False
                red.add(v)
            computed.add(v)
            
        elif op == 'delete':
            if v not in red:
                return False
            red.remove(v)
            
        else:
            return False
            
    for v in outputs:
        if v not in blue:
            return False
            
    return True

def _count_transfers(ops):
    """Count the number of load and store operations (transfers)."""
    count = 0
    for op, v in ops:
        if op == 'load' or op == 'store':
            count += 1
    return count

def _compute_depths(vertices, parents):
    """Compute longest path depth from sources for each vertex."""
    order = _topological_sort(vertices, parents)
    depth = {v: 0 for v in vertices}
    for v in order:
        for p in parents.get(v, []):
            if depth[p] + 1 > depth[v]:
                depth[v] = depth[p] + 1
    return depth, order

def _compute_next_use(vertices, parents, order):
    """
    For each vertex, compute the index in topological order of its next use (child).
    This helps decide what to evict - evict vertices whose next use is furthest away.
    """
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Map vertex to its position in topological order
    pos = {v: i for i, v in enumerate(order)}
    
    # For each vertex, find the minimum position among its children (next use)
    next_use = {}
    for v in vertices:
        if children[v]:
            next_use[v] = min(pos[c] for c in children[v])
        else:
            next_use[v] = float('inf')
    
    return next_use, children

def _greedy_schedule_with_lookahead(dag, fast_memory_budget, order, next_use, children):
    """
    Greedy schedule that uses lookahead to decide evictions.
    When we need to evict, we evict the vertex whose next use is furthest in the future.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    ops = []
    in_memory = set()
    remaining_consumers = {v: len(children[v]) for v in vertices}
    
    for v in order:
        if v in inputs:
            if v not in in_memory:
                if len(in_memory) >= fast_memory_budget:
                    # Need to evict: choose vertex with furthest next use
                    candidates = [(next_use[u], u) for u in in_memory if remaining_consumers[u] == 0]
                    if candidates:
                        # Evict the one with furthest next use (largest next_use value)
                        candidates.sort(key=lambda x: -x[0])
                        victim = candidates[0][1]
                        if victim in outputs:
                            ops.append(['store', victim])
                        else:
                            ops.append(['delete', victim])
                        in_memory.remove(victim)
                    else:
                        return None
                ops.append(['load', v])
                in_memory.add(v)
        else:
            # Ensure parents are in memory
            for p in parents[v]:
                if p not in in_memory:
                    if p in inputs:
                        if len(in_memory) >= fast_memory_budget:
                            candidates = [(next_use[u], u) for u in in_memory if remaining_consumers[u] == 0]
                            if candidates:
                                candidates.sort(key=lambda x: -x[0])
                                victim = candidates[0][1]
                                if victim in outputs:
                                    ops.append(['store', victim])
                                else:
                                    ops.append(['delete', victim])
                                in_memory.remove(victim)
                            else:
                                return None
                        ops.append(['load', p])
                        in_memory.add(p)
                    else:
                        return None
            
            if len(in_memory) >= fast_memory_budget:
                candidates = [(next_use[u], u) for u in in_memory if remaining_consumers[u] == 0]
                if candidates:
                    candidates.sort(key=lambda x: -x[0])
                    victim = candidates[0][1]
                    if victim in outputs:
                        ops.append(['store', victim])
                    else:
                        ops.append(['delete', victim])
                    in_memory.remove(victim)
                else:
                    return None
            
            ops.append(['compute', v])
            in_memory.add(v)
            
            for p in parents[v]:
                remaining_consumers[p] -= 1
    
    # Final cleanup
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
    
    return ops

def _layered_schedule(dag, fast_memory_budget):
    """
    Layered pebbling: process by depth layers.
    Keep current and next layer in memory.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    depth, order = _compute_depths(vertices, parents)
    
    layers = defaultdict(list)
    for v in order:
        layers[depth[v]].append(v)
    
    max_depth = max(layers.keys()) if layers else 0
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    ops = []
    in_memory = set()
    remaining_consumers = {v: len(children[v]) for v in vertices}
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    for d in range(max_depth + 1):
        if d not in layers:
            continue
        for v in layers[d]:
            if v in inputs:
                if v not in in_memory:
                    if len(in_memory) >= fast_memory_budget:
                        # Evict from previous layers
                        candidates = [(u, remaining_consumers[u]) for u in in_memory if remaining_consumers[u] == 0]
                        if candidates:
                            # Evict one with 0 remaining consumers
                            victim = candidates[0][0]
                            if victim in outputs:
                                ops.append(['store', victim])
                            else:
                                ops.append(['delete', victim])
                            in_memory.remove(victim)
                        else:
                            return None
                    ops.append(['load', v])
                    in_memory.add(v)
            else:
                for p in parents[v]:
                    if p not in in_memory:
                        if p in inputs:
                            if len(in_memory) >= fast_memory_budget:
                                candidates = [(u, remaining_consumers[u]) for u in in_memory if remaining_consumers[u] == 0]
                                if candidates:
                                    victim = candidates[0][0]
                                    if victim in outputs:
                                        ops.append(['store', victim])
                                    else:
                                        ops.append(['delete', victim])
                                    in_memory.remove(victim)
                                else:
                                    return None
                            ops.append(['load', p])
                            in_memory.add(p)
                        else:
                            return None
                
                if len(in_memory) >= fast_memory_budget:
                    candidates = [(u, remaining_consumers[u]) for u in in_memory if remaining_consumers[u] == 0]
                    if candidates:
                        victim = candidates[0][0]
                        if victim in outputs:
                            ops.append(['store', victim])
                        else:
                            ops.append(['delete', victim])
                        in_memory.remove(victim)
                    else:
                        return None
                
                ops.append(['compute', v])
                in_memory.add(v)
                
                for p in parents[v]:
                    remaining_consumers[p] -= 1
    
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
    
    return ops

def _bottom_up_schedule(dag, fast_memory_budget):
    """
    Bottom-up pebbling strategy.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    remaining_consumers = {v: len(children[v]) for v in vertices}
    
    ops = []
    in_memory = set()
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    for v in reversed(order):
        if v in inputs:
            continue
        
        for p in parents[v]:
            if p not in in_memory:
                if p in inputs:
                    if len(in_memory) >= fast_memory_budget:
                        freed = False
                        for u in list(in_memory):
                            if remaining_consumers[u] == 0:
                                if u in outputs:
                                    ops.append(['store', u])
                                else:
                                    ops.append(['delete', u])
                                in_memory.remove(u)
                                freed = True
                                break
                        if not freed:
                            return None
                    ops.append(['load', p])
                    in_memory.add(p)
                else:
                    return None
        
        if len(in_memory) >= fast_memory_budget:
            freed = False
            for u in list(in_memory):
                if remaining_consumers[u] == 0:
                    if u in outputs:
                        ops.append(['store', u])
                    else:
                        ops.append(['delete', u])
                    in_memory.remove(u)
                    freed = True
                    break
            if not freed:
                return None
        
        ops.append(['compute', v])
        in_memory.add(v)
        
        for p in parents[v]:
            remaining_consumers[p] -= 1
            if remaining_consumers[p] == 0:
                if p in in_memory:
                    if p in outputs:
                        ops.append(['store', p])
                    else:
                        ops.append(['delete', p])
                    in_memory.remove(p)
    
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
    
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0
    
    try:
        best_schedule = None
        best_transfers = float('inf')
        
        vertices = dag['vertices']
        parents = dag['parents']
        outputs = set(dag['outputs'])
        
        depth, order = _compute_depths(vertices, parents)
        next_use, children = _compute_next_use(vertices, parents, order)
        
        # Strategy 1: Greedy with lookahead
        if time.time() < deadline:
            schedule = _greedy_schedule_with_lookahead(dag, fast_memory_budget, order, next_use, children)
            if schedule is not None and _validate_schedule(dag, schedule, fast_memory_budget):
                t = _count_transfers(schedule)
                if t < best_transfers:
                    best_transfers = t
                    best_schedule = schedule
        
        # Strategy 2: Layered
        if time.time() < deadline:
            schedule = _layered_schedule(dag, fast_memory_budget)
            if schedule is not None and _validate_schedule(dag, schedule, fast_memory_budget):
                t = _count_transfers(schedule)
                if t < best_transfers:
                    best_transfers = t
                    best_schedule = schedule
        
        # Strategy 3: Bottom-up
        if time.time() < deadline:
            schedule = _bottom_up_schedule(dag, fast_memory_budget)
            if schedule is not None and _validate_schedule(dag, schedule, fast_memory_budget):
                t = _count_transfers(schedule)
                if t < best_transfers:
                    best_transfers = t
                    best_schedule = schedule
        
        if best_schedule is not None:
            return best_schedule
        
        return []
        
    except Exception:
        return []

# EVOLVE-BLOCK-END
