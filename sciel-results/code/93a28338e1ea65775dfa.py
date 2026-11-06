"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict

def _topological_sort(vertices, parents):
    """Returns a list of vertices in topological order."""
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
    """Validates a schedule by simulating the state machine."""
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
    """Count the number of load and store operations."""
    return sum(1 for op, v in ops if op in ('load', 'store'))

def _greedy_schedule(dag, fast_memory_budget):
    """
    Greedy pebbling schedule that minimizes transfers.
    Uses a priority-based eviction strategy.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    vertex_to_idx = {v: i for i, v in enumerate(order)}
    
    # Precompute children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # For each vertex, compute the latest index of its children (for eviction priority)
    latest_child_idx = {}
    for v in vertices:
        if children[v]:
            latest_child_idx[v] = max(vertex_to_idx[c] for c in children[v])
        else:
            latest_child_idx[v] = vertex_to_idx[v]  # leaf
    
    ops = []
    in_memory = set()
    memory_full_count = 0
    
    # Track which vertices have been computed
    computed = set()
    
    # Process vertices in topological order
    for v in order:
        # Check if v is an input (no parents)
        if not parents.get(v, []):
            # Input vertex: must be loaded
            if v not in in_memory:
                if len(in_memory) >= fast_memory_budget:
                    # Need to evict something
                    # Find the vertex with the latest last-use that is not needed soon
                    candidates = []
                    for u in in_memory:
                        if u != v:
                            # u can be evicted if all its children are already computed
                            # or if we're willing to recompute
                            all_children_computed = all(c in computed for c in children[u])
                            candidates.append((latest_child_idx.get(u, -1), u, all_children_computed))
                    
                    if candidates:
                        # Prefer to evict vertices whose children are all computed (safe to delete)
                        safe_candidates = [c for c in candidates if c[2]]
                        if safe_candidates:
                            # Evict the one with the earliest last use
                            safe_candidates.sort(key=lambda x: x[0])
                            _, evict_v, _ = safe_candidates[0]
                        else:
                            # Must evict and recompute later
                            candidates.sort(key=lambda x: x[0])
                            _, evict_v, _ = candidates[0]
                        
                        if evict_v in outputs:
                            ops.append(['store', evict_v])
                        else:
                            ops.append(['delete', evict_v])
                        in_memory.remove(evict_v)
                    
                    if len(in_memory) >= fast_memory_budget:
                        return None  # Can't make progress
                
                ops.append(['load', v])
                in_memory.add(v)
            continue
        
        # Non-input vertex: compute it
        # Ensure all parents are in memory
        for p in parents[v]:
            if p not in in_memory:
                # Need to load or recompute p
                # Check if p is an input
                if not parents.get(p, []):
                    # Input: load it
                    if len(in_memory) >= fast_memory_budget:
                        # Evict
                        candidates = []
                        for u in in_memory:
                            if u != p:
                                all_children_computed = all(c in computed for c in children[u])
                                candidates.append((latest_child_idx.get(u, -1), u, all_children_computed))
                        
                        if candidates:
                            safe_candidates = [c for c in candidates if c[2]]
                            if safe_candidates:
                                safe_candidates.sort(key=lambda x: x[0])
                                _, evict_v, _ = safe_candidates[0]
                            else:
                                candidates.sort(key=lambda x: x[0])
                                _, evict_v, _ = candidates[0]
                            
                            if evict_v in outputs:
                                ops.append(['store', evict_v])
                            else:
                                ops.append(['delete', evict_v])
                            in_memory.remove(evict_v)
                        
                        if len(in_memory) >= fast_memory_budget:
                            return None
                    
                    ops.append(['load', p])
                    in_memory.add(p)
                else:
                    # Non-input: need to recompute
                    # This is complex; for now, try to find space
                    if len(in_memory) >= fast_memory_budget:
                        # Try to evict
                        candidates = []
                        for u in in_memory:
                            if u != p and u not in parents[v]:
                                all_children_computed = all(c in computed for c in children[u])
                                candidates.append((latest_child_idx.get(u, -1), u, all_children_computed))
                        
                        if candidates:
                            safe_candidates = [c for c in candidates if c[2]]
                            if safe_candidates:
                                safe_candidates.sort(key=lambda x: x[0])
                                _, evict_v, _ = safe_candidates[0]
                            else:
                                candidates.sort(key=lambda x: x[0])
                                _, evict_v, _ = candidates[0]
                            
                            if evict_v in outputs:
                                ops.append(['store', evict_v])
                            else:
                                ops.append(['delete', evict_v])
                            in_memory.remove(evict_v)
                        
                        if len(in_memory) >= fast_memory_budget:
                            return None
                    
                    # Recompute p: need its parents in memory
                    for gp in parents.get(p, []):
                        if gp not in in_memory:
                            if len(in_memory) >= fast_memory_budget:
                                return None
                            # Load gp if input, else this gets complex
                            if not parents.get(gp, []):
                                ops.append(['load', gp])
                                in_memory.add(gp)
                            else:
                                return None  # Too complex for this path
                    
                    ops.append(['compute', p])
                    in_memory.add(p)
        
        # Now compute v
        if v not in in_memory:
            if len(in_memory) >= fast_memory_budget:
                # Evict
                candidates = []
                for u in in_memory:
                    if u != v:
                        all_children_computed = all(c in computed for c in children[u])
                        candidates.append((latest_child_idx.get(u, -1), u, all_children_computed))
                
                if candidates:
                    safe_candidates = [c for c in candidates if c[2]]
                    if safe_candidates:
                        safe_candidates.sort(key=lambda x: x[0])
                        _, evict_v, _ = safe_candidates[0]
                    else:
                        candidates.sort(key=lambda x: x[0])
                        _, evict_v, _ = candidates[0]
                    
                    if evict_v in outputs:
                        ops.append(['store', evict_v])
                    else:
                        ops.append(['delete', evict_v])
                    in_memory.remove(evict_v)
                
                if len(in_memory) >= fast_memory_budget:
                    return None
            
            ops.append(['compute', v])
            in_memory.add(v)
        
        computed.add(v)
        
        # After computing v, we can evict parents that are no longer needed
        for p in parents[v]:
            remaining = [c for c in children[p] if c not in computed]
            if not remaining and p not in outputs:
                # p is no longer needed, evict it
                if p in in_memory:
                    ops.append(['delete', p])
                    in_memory.remove(p)
            elif not remaining and p in outputs:
                # p is an output, store it
                if p in in_memory:
                    ops.append(['store', p])
                    in_memory.remove(p)
    
    # Final cleanup: store outputs, delete others
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
    
    return ops

def _layered_schedule(dag, fast_memory_budget):
    """
    Schedule optimized for layered DAGs.
    Processes layer by layer, keeping only the current and next layer in memory.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    vertex_to_idx = {v: i for i, v in enumerate(order)}
    
    # Precompute children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Determine depth/level for each vertex
    level = {}
    for v in order:
        if not parents.get(v, []):
            level[v] = 0
        else:
            level[v] = max(level[p] for p in parents[v]) + 1
    
    # Group vertices by level
    max_level = max(level.values()) if level else 0
    levels = defaultdict(list)
    for v in vertices:
        levels[level[v]].append(v)
    
    ops = []
    in_memory = set()
    computed = set()
    
    # Process level by level
    for lvl in range(max_level + 1):
        # Load inputs for this level if needed
        for v in levels[lvl]:
            if not parents.get(v, []):
                # Input at this level
                if v not in in_memory:
                    if len(in_memory) >= fast_memory_budget:
                        # Evict vertices from previous levels that are no longer needed
                        evicted = False
                        for u in list(in_memory):
                            if u != v and all(c in computed for c in children[u]):
                                if u in outputs:
                                    ops.append(['store', u])
                                else:
                                    ops.append(['delete', u])
                                in_memory.remove(u)
                                evicted = True
                                break
                        
                        if len(in_memory) >= fast_memory_budget:
                            return None
                    
                    ops.append(['load', v])
                    in_memory.add(v)
            
            # Compute v
            if v not in in_memory:
                # Ensure parents are in memory
                for p in parents[v]:
                    if p not in in_memory:
                        if not parents.get(p, []):
                            # Input
                            if len(in_memory) >= fast_memory_budget:
                                # Evict
                                evicted = False
                                for u in list(in_memory):
                                    if u != p and all(c in computed for c in children[u]):
                                        if u in outputs:
                                            ops.append(['store', u])
                                        else:
                                            ops.append(['delete', u])
                                        in_memory.remove(u)
                                        evicted = True
                                        break
                                
                                if len(in_memory) >= fast_memory_budget:
                                    return None
                            
                            ops.append(['load', p])
                            in_memory.add(p)
                        else:
                            # Non-input parent, need to recompute
                            # For layered DAGs, parents should be from previous level
                            # Try to load/recompute
                            if len(in_memory) >= fast_memory_budget:
                                evicted = False
                                for u in list(in_memory):
                                    if u != p and u not in parents[v] and all(c in computed for c in children[u]):
                                        if u in outputs:
                                            ops.append(['store', u])
                                        else:
                                            ops.append(['delete', u])
                                        in_memory.remove(u)
                                        evicted = True
                                        break
                                
                                if len(in_memory) >= fast_memory_budget:
                                    return None
                            
                            # Recompute p
                            for gp in parents.get(p, []):
                                if gp not in in_memory:
                                    if not parents.get(gp, []):
                                        if len(in_memory) >= fast_memory_budget:
                                            return None
                                        ops.append(['load', gp])
                                        in_memory.add(gp)
                                    else:
                                        return None
                            
                            ops.append(['compute', p])
                            in_memory.add(p)
                
                if len(in_memory) >= fast_memory_budget:
                    # Evict
                    evicted = False
                    for u in list(in_memory):
                        if u != v and all(c in computed for c in children[u]):
                            if u in outputs:
                                ops.append(['store', u])
                            else:
                                ops.append(['delete', u])
                            in_memory.remove(u)
                            evicted = True
                            break
                    
                    if len(in_memory) >= fast_memory_budget:
                        return None
                
                ops.append(['compute', v])
                in_memory.add(v)
            
            computed.add(v)
        
        # After processing this level, evict vertices from previous levels that are no longer needed
        for u in list(in_memory):
            if u not in levels[lvl] and u not in levels[lvl + 1] if lvl + 1 <= max_level else u not in levels[lvl]:
                # Check if u is still needed
                future_children = [c for c in children[u] if c not in computed]
                if not future_children:
                    if u in outputs:
                        ops.append(['store', u])
                    else:
                        ops.append(['delete', u])
                    in_memory.remove(u)
    
    # Final cleanup
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
    
    return ops

def _simple_schedule(dag, fast_memory_budget):
    """
    A robust simple schedule:
    1. Load all inputs.
    2. Compute vertices in topological order.
    3. When memory is full, delete vertices that are no longer needed.
    4. Store outputs when they are no longer needed as parents, or at the end.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    # Precompute children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    ops = []
    in_memory = set()
    
    # Count remaining children for each vertex
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # 1. Load all inputs
    for v in order:
        if not parents.get(v, []):
            if len(in_memory) >= fast_memory_budget:
                return None
            ops.append(['load', v])
            in_memory.add(v)
            
    # 2. Compute in topological order
    for v in order:
        if not parents.get(v, []):
            continue
            
        if len(in_memory) >= fast_memory_budget:
            freed = False
            for u in list(in_memory):
                if u == v:
                    continue
                if remaining_children[u] == 0:
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
            remaining_children[p] -= 1
            
    # 3. Final cleanup
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
            
    return ops

def _count_transfers(ops):
    """Count the number of load and store operations."""
    return sum(1 for op, v in ops if op in ('load', 'store'))

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    try:
        # Try multiple strategies and pick the best valid one
        schedules = []
        
        # Strategy 1: Simple schedule
        s1 = _simple_schedule(dag, fast_memory_budget)
        if s1 is not None and _validate_schedule(dag, s1, fast_memory_budget):
            schedules.append(s1)
        
        # Check time
        if time.time() - start_time > deadline - 20:
            if schedules:
                return min(schedules, key=_count_transfers)
            return s1 if s1 else []
        
        # Strategy 2: Greedy schedule
        s2 = _greedy_schedule(dag, fast_memory_budget)
        if s2 is not None and _validate_schedule(dag, s2, fast_memory_budget):
            schedules.append(s2)
        
        # Check time
        if time.time() - start_time > deadline - 15:
            if schedules:
                return min(schedules, key=_count_transfers)
            return s1 if s1 else []
        
        # Strategy 3: Layered schedule
        s3 = _layered_schedule(dag, fast_memory_budget)
        if s3 is not None and _validate_schedule(dag, s3, fast_memory_budget):
            schedules.append(s3)
        
        # Check time
        if time.time() - start_time > deadline - 10:
            if schedules:
                return min(schedules, key=_count_transfers)
            return s1 if s1 else []
        
        # Pick the best schedule
        if schedules:
            best = min(schedules, key=_count_transfers)
            return best
        else:
            # Fallback
            return s1 if s1 else []
            
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

# EVOLVE-BLOCK-END
