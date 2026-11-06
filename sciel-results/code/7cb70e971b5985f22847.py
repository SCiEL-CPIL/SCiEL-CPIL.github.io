# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

# Deadline guard: 70s of the 90s wall-clock limit, leaving safety margin
_DEADLINE_SECONDS = 70.0


def _toposort_vertices(dag):
    """Extract vertices and build adjacency from the dag structure."""
    if 'vertices' in dag:
        vertices = list(dag['vertices'])
    elif 'nodes' in dag:
        vertices = list(dag['nodes'])
    else:
        all_v = set()
        parents = dag.get('parents', {})
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        vertices = sorted(all_v)

    if 'parents' in dag:
        parents = {v: list(ps) for v, ps in dag['parents'].items()}
    elif 'edges' in dag:
        parents = {}
        for edge in dag['edges']:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                p, c = edge
                parents.setdefault(c, []).append(p)
    else:
        parents = {}

    if 'outputs' in dag:
        outputs = set(dag['outputs'])
    elif 'output' in dag:
        outputs = set(dag['output']) if isinstance(dag['output'], (list, set, tuple)) else {dag['output']}
    else:
        children = {}
        for v, ps in parents.items():
            for p in ps:
                children.setdefault(p, []).append(v)
        outputs = {v for v in vertices if v not in children}

    if 'inputs' in dag:
        inputs = set(dag['inputs'])
    elif 'blue' in dag:
        inputs = set(dag['blue'])
    elif 'blue_vertices' in dag:
        inputs = set(dag['blue_vertices'])
    else:
        inputs = {v for v in vertices if not parents.get(v)}

    return vertices, parents, outputs, inputs


def _validate_ops(ops, dag, fast_memory_budget):
    """Validate that the operation sequence is legal."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    blue = set(inputs)
    fast_mem = set()
    available = set(inputs)
    
    for i, (op, v) in enumerate(ops):
        if op == 'load':
            if v not in inputs:
                return False
            if v in fast_mem:
                return False
            if len(fast_mem) >= fast_memory_budget:
                return False
            fast_mem.add(v)
            available.add(v)
        elif op == 'store':
            if v not in fast_mem and v not in blue:
                return False
            if v in fast_mem:
                fast_mem.remove(v)
            blue.add(v)
        elif op == 'compute':
            ps = parents.get(v, [])
            for p in ps:
                if p not in available:
                    return False
            if len(fast_mem) >= fast_memory_budget:
                return False
            fast_mem.add(v)
            available.add(v)
        elif op == 'delete':
            if v not in fast_mem:
                return False
            fast_mem.remove(v)
            if v not in blue:
                available.discard(v)
        else:
            return False
    
    if not outputs.issubset(blue):
        return False
    
    return True


def _count_transfers(ops):
    """Count load+store operations (transfers)."""
    return sum(1 for op, v in ops if op in ('load', 'store'))


def _exact_dijkstra_pebble(dag, fast_memory_budget, time_limit=15.0):
    """Exact Dijkstra search over memory configurations to minimize transfers."""
    start_time = time.time()
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    init_blue = frozenset(inputs)
    init_fast = frozenset()
    init_state = (init_blue, init_fast)
    
    def is_goal(state):
        blue, fast = state
        return outputs.issubset(blue)
    
    if is_goal(init_state):
        return []
    
    best_cost = {init_state: 0}
    parent = {}
    pq = [(0, 0, init_state)]
    
    max_states = 200000
    states_explored = 0
    
    while pq:
        if time.time() - start_time > time_limit:
            break
        if states_explored >= max_states:
            break
            
        cost, ops_count, state = heapq.heappop(pq)
        
        if cost > best_cost.get(state, float('inf')):
            continue
        
        states_explored += 1
        
        if is_goal(state):
            path = []
            cur = state
            while cur != init_state:
                prev, op, v = parent[cur]
                path.append((op, v))
                cur = prev
            path.reverse()
            return path
        
        blue, fast = state
        available = blue | fast
        
        # Generate transitions
        # 1. load(v) for v in inputs not in fast
        for v in inputs:
            if v not in fast:
                new_fast = fast | {v}
                if len(new_fast) <= fast_memory_budget:
                    new_state = (blue, new_fast)
                    new_cost = cost + 1
                    if new_cost < best_cost.get(new_state, float('inf')):
                        best_cost[new_state] = new_cost
                        parent[new_state] = (state, 'load', v)
                        heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
        
        # 2. store(v) for v in fast
        for v in fast:
            new_fast = fast - {v}
            new_blue = blue | {v}
            new_state = (new_blue, new_fast)
            new_cost = cost + 1
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent[new_state] = (state, 'store', v)
                heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
        
        # 3. compute(v) for v not in inputs, not in fast, all parents available
        for v in vertices:
            if v in inputs or v in fast:
                continue
            ps = parents.get(v, [])
            if all(p in available for p in ps):
                if len(fast) < fast_memory_budget:
                    new_fast = fast | {v}
                    new_state = (blue, new_fast)
                    new_cost = cost
                    if new_cost < best_cost.get(new_state, float('inf')):
                        best_cost[new_state] = new_cost
                        parent[new_state] = (state, 'compute', v)
                        heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
        
        # 4. delete(v) for v in fast
        for v in fast:
            new_fast = fast - {v}
            new_state = (blue, new_fast)
            new_cost = cost
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent[new_state] = (state, 'delete', v)
                heapq.heappush(pq, (new_cost, ops_count + 1, new_state))
    
    return None


def _greedy_with_lookahead(dag, fast_memory_budget):
    """Greedy pebbling with lookahead: process in topological order, but make
    smarter eviction decisions by looking ahead to future needs."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Compute for each vertex: how many more children need it (future use count)
    # and the depth (for prioritization)
    depth = {}
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in ps) + 1
    
    # For each vertex, compute its "usefulness score": 
    # higher = more useful to keep in fast memory
    # A vertex is useful if it has many remaining children or is an output
    def usefulness(v, remaining_children):
        score = 0
        if v in outputs:
            score += 1000  # Outputs are very valuable
        score += remaining_children.get(v, 0) * 10
        # Prefer keeping vertices with higher depth (closer to outputs)
        score += depth.get(v, 0)
        return score
    
    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    
    # Track remaining children for each vertex
    remaining_children = {v: len(children.get(v, [])) for v in vertices}
    
    # Process in topological order
    for v in vertices:
        if v in inputs:
            if v not in available:
                # Need space
                if len(fast_mem) >= fast_memory_budget:
                    # Find least useful item to evict
                    best_evict = None
                    best_score = float('inf')
                    for fm in fast_mem:
                        s = usefulness(fm, remaining_children)
                        if s < best_score:
                            best_score = s
                            best_evict = fm
                    if best_evict is not None:
                        ops.append(('delete', best_evict))
                        fast_mem.discard(best_evict)
                        if best_evict not in blue:
                            available.discard(best_evict)
                
                if len(fast_mem) < fast_memory_budget:
                    ops.append(('load', v))
                    fast_mem.add(v)
                    available.add(v)
        else:
            # Compute v
            ps = parents.get(v, [])
            # All parents must be available
            for p in ps:
                if p not in available:
                    # Shouldn't happen in topological order, but handle it
                    pass
            
            # Need space for result
            if len(fast_mem) >= fast_memory_budget:
                best_evict = None
                best_score = float('inf')
                for fm in fast_mem:
                    s = usefulness(fm, remaining_children)
                    if s < best_score:
                        best_score = s
                        best_evict = fm
                if best_evict is not None:
                    ops.append(('delete', best_evict))
                    fast_mem.discard(best_evict)
                    if best_evict not in blue:
                        available.discard(best_evict)
            
            if len(fast_mem) < fast_memory_budget:
                ops.append(('compute', v))
                fast_mem.add(v)
                available.add(v)
        
        # Update remaining children for all parents
        for p in parents.get(v, []):
            remaining_children[p] = remaining_children.get(p, 0) - 1
        
        # Store outputs immediately
        if v in outputs and v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
    
    # Cleanup
    for v in list(fast_mem):
        ops.append(('delete', v))
    
    return ops


def _layered_optimized(dag, fast_memory_budget):
    """Layer-by-layer pebbling optimized for layered DAGs.
    
    For layered DAGs, we can process one layer at a time:
    - Load inputs for the current layer
    - Compute the current layer
    - Store outputs from this layer
    - Delete intermediate results that won't be needed again
    
    This minimizes the peak fast memory usage and transfers.
    """
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    # Compute depth
    depth = {}
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in ps) + 1
    
    max_depth = max(depth.values()) if depth else 0
    
    # Group by depth
    layers = defaultdict(list)
    for v in vertices:
        layers[depth[v]].append(v)
    
    # For each layer, determine which inputs are needed
    # An input at layer d is needed if it's a parent of any vertex at layer d+1 or later
    # Actually, inputs are at layer 0. We need to track which inputs are needed for future layers.
    
    # Build: for each input, which layers need it
    input_needs = defaultdict(set)  # input -> set of layers that need it
    for v in vertices:
        for p in parents.get(v, []):
            if p in inputs:
                input_needs[p].add(depth[v])
    
    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    
    # Track which inputs have been loaded and are still needed
    loaded_inputs = set()
    
    for d in range(max_depth + 1):
        layer_vertices = layers[d]
        
        # Determine which inputs are needed for this layer or future layers
        needed_inputs = set()
        for v in layer_vertices:
            for p in parents.get(v, []):
                if p in inputs:
                    needed_inputs.add(p)
        
        # Also include inputs needed for future layers (they should stay in fast mem)
        for inp in inputs:
            if d in input_needs.get(inp, set()):
                needed_inputs.add(inp)
        
        # Load needed inputs that aren't in fast mem
        for inp in needed_inputs:
            if inp not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    # Evict an input that's no longer needed
                    evicted = False
                    for fm in list(fast_mem):
                        if fm in inputs:
                            # Check if fm is still needed
                            still_needed = False
                            for future_d in range(d, max_depth + 1):
                                if fm in input_needs and future_d in input_needs[fm]:
                                    still_needed = True
                                    break
                            if not still_needed:
                                ops.append(('delete', fm))
                                fast_mem.discard(fm)
                                available.discard(fm)
                                evicted = True
                                break
                    if not evicted and fast_mem:
                        # Evict any non-input or least useful
                        fm = next(iter(fast_mem))
                        ops.append(('delete', fm))
                        fast_mem.discard(fm)
                        if fm not in blue:
                            available.discard(fm)
                
                if len(fast_mem) < fast_memory_budget:
                    ops.append(('load', inp))
                    fast_mem.add(inp)
                    available.add(inp)
                    loaded_inputs.add(inp)
        
        # Compute vertices in this layer
        for v in layer_vertices:
            if v in inputs:
                continue
            ps = parents.get(v, [])
            # All parents should be available
            if len(fast_mem) >= fast_memory_budget:
                # Evict something
                evicted = False
                for fm in list(fast_mem):
                    # Prefer evicting inputs that are no longer needed
                    if fm in inputs:
                        still_needed = False
                        for future_d in range(d, max_depth + 1):
                            if fm in input_needs and future_d in input_needs[fm]:
                                still_needed = True
                                break
                        if not still_needed:
                            ops.append(('delete', fm))
                            fast_mem.discard(fm)
                            available.discard(fm)
                            evicted = True
                            break
                if not evicted and fast_mem:
                    fm = next(iter(fast_mem))
                    ops.append(('delete', fm))
                    fast_mem.discard(fm)
                    if fm not in blue:
                        available.discard(fm)
            
            if len(fast_mem) < fast_memory_budget:
                ops.append(('compute', v))
                fast_mem.add(v)
                available.add(v)
        
        # Store outputs in this layer
        for v in layer_vertices:
            if v in outputs and v in fast_mem:
                ops.append(('store', v))
                fast_mem.discard(v)
                blue.add(v)
        
        # Delete intermediate results from previous layers that are no longer needed
        # A vertex at depth d-1 is no longer needed if all its children are at depth d or later
        # and all those children have been processed
        # For simplicity, delete non-output, non-input vertices from previous layers
        # that are no longer parents of any unprocessed vertex
        for v in list(fast_mem):
            if v in inputs or v in outputs:
                continue
            # Check if v is still a parent of any unprocessed vertex
            still_parent = False
            for child in children.get(v, []):
                if depth.get(child, 0) >= d:
                    still_parent = True
                    break
            if not still_parent:
                ops.append(('delete', v))
                fast_mem.discard(v)
                if v not in blue:
                    available.discard(v)
    
    # Final cleanup
    for v in list(fast_mem):
        ops.append(('delete', v))
    
    return ops


def _naive_baseline(dag, fast_memory_budget):
    """Generate a simple naive schedule as fallback."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    ops = []
    for v in vertices:
        if v in inputs:
            ops.append(('load', v))
        else:
            ops.append(('compute', v))
    for v in outputs:
        ops.append(('store', v))
    return ops


def pebble(dag, fast_memory_budget):
    """Main entry point: find communication-minimizing pebbling schedule."""
    start_time = time.time()
    
    def time_remaining():
        return _DEADLINE_SECONDS - (time.time() - start_time)
    
    if time_remaining() < 2.0:
        # Emergency fallback
        return _naive_baseline(dag, fast_memory_budget)
    
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    n_vertices = len(vertices)
    
    best_ops = None
    best_transfers = float('inf')
    
    def consider(ops):
        nonlocal best_ops, best_transfers
        if ops is None:
            return
        if len(ops) > 20000:
            return
        if not _validate_ops(ops, dag, fast_memory_budget):
            return
        t = _count_transfers(ops)
        if t < best_transfers:
            best_transfers = t
            best_ops = ops
    
    # Try exact Dijkstra for small instances
    if n_vertices <= 14 and time_remaining() > 10.0:
        result = _exact_dijkstra_pebble(dag, fast_memory_budget, time_limit=min(10.0, time_remaining() - 5.0))
        consider(result)
    
    # Try greedy with lookahead
    if time_remaining() > 5.0:
        result = _greedy_with_lookahead(dag, fast_memory_budget)
        consider(result)
    
    # Try layered optimized
    if time_remaining() > 5.0:
        result = _layered_optimized(dag, fast_memory_budget)
        consider(result)
    
    # Try naive as last resort
    if best_ops is None:
        result = _naive_baseline(dag, fast_memory_budget)
        consider(result)
    
    if best_ops is not None:
        return best_ops
    
    return _naive_baseline(dag, fast_memory_budget)

# EVOLVE-BLOCK-END
