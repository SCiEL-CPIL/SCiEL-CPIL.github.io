"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        has_baseline = True
    except ImportError:
        has_baseline = False

    try:
        if isinstance(dag, dict):
            parents = dict(dag.get('parents', {}))
            outputs = list(dag.get('outputs', []))
            vertices = dag.get('vertices', None)
            inputs = dag.get('inputs', None)
        else:
            parents = dict(getattr(dag, 'parents', {}))
            outputs = list(getattr(dag, 'outputs', []))
            vertices = getattr(dag, 'vertices', None)
            inputs = getattr(dag, 'inputs', None)
    except Exception:
        if has_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        raise

    if vertices is not None:
        all_vertices = set(vertices)
    else:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        for o in outputs:
            all_vertices.add(o)

    if inputs is not None:
        input_vertices = set(inputs)
    else:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    # If the DAG is small enough, use Dijkstra for optimal solution
    if len(all_vertices) <= 20:
        return pebble_dijkstra(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)

    # Otherwise, use improved greedy with priority-based computation
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    # Compute topological order and remaining children count
    in_deg = {v: len(parents.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_deg[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    if len(topo_order) < len(all_vertices):
        for v in all_vertices:
            if v not in topo_order:
                topo_order.append(v)

    # Precompute next-use layer for eviction scoring
    layer = {}
    for v in input_vertices:
        layer[v] = 0
    queue = deque(list(input_vertices))
    while queue:
        v = queue.popleft()
        for c in children[v]:
            if c not in layer:
                all_p_done = all(p in layer for p in parents.get(c, []))
                if all_p_done:
                    layer[c] = max(layer[p] for p in parents.get(c, [])) + 1
                    queue.append(c)
    for v in all_vertices:
        if v not in layer:
            layer[v] = 0
    max_L = max(layers.keys()) if (layers := {v: layer[v] for v in all_vertices}) else 0

    # Precompute next use layers
    next_use_layers = {v: sorted(layer[c] for c in children[v]) for v in all_vertices}

    # State
    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []
    computed = set()  # vertices that have been computed at least once

    # Track remaining children for each vertex (to know when a vertex is "ready")
    remaining_parents = {v: len(parents.get(v, [])) for v in all_vertices}
    # A vertex is ready to compute if all its parents are in fast_mem
    # We maintain a set of ready vertices

    # Initialize: inputs can be loaded, non-inputs need parents in fast_mem
    # We'll use a loop: at each step, pick the best action

    # Priority for computing a vertex: prefer vertices with many children (high fan-out)
    # and vertices whose parents are about to be evicted

    # We'll use a different approach: simulate the pebbling process
    # At each step:
    #   1. If fast_mem is full and we need to compute/load, evict first
    #   2. Find all "ready" vertices (all parents in fast_mem)
    #   3. Among ready vertices, pick the best one to compute
    #   4. If no ready vertices, load a parent that is needed

    # Actually, let's use a cleaner approach:
    # Process vertices in topological order, but use a priority queue for ordering
    # within the topological constraints.

    # Key insight: we can compute a vertex as soon as all its parents are in fast_mem.
    # We want to compute vertices that will be used many times soon.

    # Let's use a simulation with explicit state tracking:
    # - fast_mem: set of red (fast) vertices
    # - slow_mem: set of black (slow) vertices  
    # - We need to ensure all outputs end up in slow_mem

    # Greedy strategy:
    # 1. Maintain a set of "available" vertices to compute (all parents in fast_mem)
    # 2. Also consider loading parents that are in slow_mem
    # 3. Pick the action that minimizes expected future transfers

    # For simplicity and speed, let's use a heuristic:
    # - Compute vertices in order of decreasing "urgency" (next use layer)
    # - Evict vertices with lowest "value" (no future uses, or far future uses)

    # Let's track which vertices still need to be computed
    to_compute = set(all_vertices - input_vertices)
    to_compute -= set(outputs)  # outputs will be computed as part of their computation
    # Actually, all non-input vertices need to be computed
    to_compute = set(all_vertices - input_vertices)

    # We'll process in a loop until all outputs are in slow_mem
    max_steps = len(all_vertices) * 10 + 100
    step = 0
    
    while step < max_steps and not time_check():
        step += 1
        
        # Check if done: all outputs in slow_mem
        if all(o in slow_mem for o in outputs):
            break
        
        # Find ready vertices: all parents in fast_mem
        ready = set()
        for v in to_compute:
            if v in fast_mem or v in slow_mem:
                continue
            ps = parents.get(v, [])
            if all(p in fast_mem for p in ps):
                ready.add(v)
        
        if ready:
            # Pick the best vertex to compute
            # Priority: high fan-out, low next-use-layer (used soon)
            best_v = None
            best_key = None
            for v in ready:
                # Score: prefer vertices with many children and early next use
                fanout = len(children[v])
                # Next use: minimum layer of children
                if children[v]:
                    min_child_layer = min(layer[c] for c in children[v])
                else:
                    min_child_layer = max_L + 1
                # Higher priority if high fanout and early use
                key = (fanout, -min_child_layer)
                if best_key is None or key > best_key:
                    best_key = key
                    best_v = v
            
            # Evict if needed
            while len(fast_mem) >= fast_memory_budget and best_v not in fast_mem:
                if time_check():
                    break
                # Find best vertex to evict
                best_evict = None
                best_evict_score = float('inf')
                for u in fast_mem:
                    if u in outputs:
                        # Must store outputs
                        score = -1.0  # Prefer to evict/store outputs
                    else:
                        # Check future uses
                        future_uses = [nl for nl in next_use_layers[u] if nl > layer.get(best_v, 0)]
                        if not future_uses:
                            score = 0.0  # No future uses, evict
                        else:
                            min_future = min(future_uses)
                            dist = min_future - layer.get(best_v, 0)
                            if u in slow_mem:
                                score = 1.0 + dist * 0.1
                            else:
                                score = 10.0 + dist * 0.1
                    
                    if score < best_evict_score:
                        best_evict_score = score
                        best_evict = u
                
                if best_evict is None:
                    break
                
                if best_evict in outputs:
                    ops.append(['store', best_evict])
                    slow_mem.add(best_evict)
                    fast_mem.discard(best_evict)
                elif best_evict in slow_mem:
                    ops.append(['delete', best_evict])
                    fast_mem.discard(best_evict)
                else:
                    ops.append(['store', best_evict])
                    slow_mem.add(best_evict)
                    ops.append(['delete', best_evict])
                    fast_mem.discard(best_evict)
            
            # Compute the vertex
            ops.append(['compute', best_v])
            fast_mem.add(best_v)
            to_compute.discard(best_v)
            computed.add(best_v)
        
        else:
            # No ready vertices. Need to load a parent.
            # Find a vertex that has parents in slow_mem but not in fast_mem
            # Pick the parent that is needed most urgently
            
            # Find all vertices that need to be computed but aren't ready
            blocked = set()
            for v in to_compute:
                if v in fast_mem or v in slow_mem:
                    continue
                ps = parents.get(v, [])
                if not all(p in fast_mem for p in ps):
                    blocked.add(v)
            
            if not blocked:
                # Nothing to compute, but outputs not all in slow_mem?
                # Maybe outputs are in fast_mem, need to store them
                break
            
            # Find the most urgent blocked vertex
            best_blocked = None
            best_blocked_key = None
            for v in blocked:
                fanout = len(children[v])
                if children[v]:
                    min_child_layer = min(layer[c] for c in children[v])
                else:
                    min_child_layer = max_L + 1
                key = (fanout, -min_child_layer)
                if best_blocked_key is None or key > best_blocked_key:
                    best_blocked_key = key
                    best_blocked = v
            
            # Load a parent of best_blocked that is in slow_mem but not in fast_mem
            ps = parents.get(best_blocked, [])
            load_candidate = None
            for p in ps:
                if p in slow_mem and p not in fast_mem:
                    load_candidate = p
                    break
            
            if load_candidate is None:
                # All parents are either in fast_mem or not in slow_mem
                # This shouldn't happen if state is maintained correctly
                break
            
            # Evict if needed
            while len(fast_mem) >= fast_memory_budget:
                if time_check():
                    break
                best_evict = None
                best_evict_score = float('inf')
                for u in fast_mem:
                    if u in outputs:
                        score = -1.0
                    else:
                        future_uses = [nl for nl in next_use_layers[u] if nl > layer.get(load_candidate, 0)]
                        if not future_uses:
                            score = 0.0
                        else:
                            min_future = min(future_uses)
                            dist = min_future - layer.get(load_candidate, 0)
                            if u in slow_mem:
                                score = 1.0 + dist * 0.1
                            else:
                                score = 10.0 + dist * 0.1
                    
                    if score < best_evict_score:
                        best_evict_score = score
                        best_evict = u
                
                if best_evict is None:
                    break
                
                if best_evict in outputs:
                    ops.append(['store', best_evict])
                    slow_mem.add(best_evict)
                    fast_mem.discard(best_evict)
                elif best_evict in slow_mem:
                    ops.append(['delete', best_evict])
                    fast_mem.discard(best_evict)
                else:
                    ops.append(['store', best_evict])
                    slow_mem.add(best_evict)
                    ops.append(['delete', best_evict])
                    fast_mem.discard(best_evict)
            
            ops.append(['load', load_candidate])
            fast_mem.add(load_candidate)
        
        # Store any outputs that are in fast_mem
        for o in list(fast_mem):
            if o in outputs:
                ops.append(['store', o])
                slow_mem.add(o)
                fast_mem.discard(o)

    # Final: ensure all outputs are in slow_mem
    for o in outputs:
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

def pebble_dijkstra(dag, budget, all_vertices, input_vertices, outputs, parents, time_check):
    # State: (frozenset(fast_red), frozenset(black))
    # In red-blue pebbling:
    # - Red = in fast memory
    # - Blue/Black = in slow memory (inputs start blue, outputs must end blue)
    # - To compute a vertex, all parents must be RED
    # - To load, vertex must be BLUE and not RED
    # - To store, vertex must be RED and not BLUE
    # - To delete, vertex must be RED
    
    init_black = frozenset(input_vertices)
    init_fast = frozenset()
    
    dist = {}
    prev = {}
    
    start_state = (init_fast, init_black)
    dist[start_state] = 0
    
    pq = [(0, start_state)]
    
    while pq:
        if time_check():
            break
        cost, state = heapq.heappop(pq)
        fast, black = state
        
        if cost > dist.get(state, float('inf')):
            continue
            
        if all(o in black for o in outputs):
            ops = []
            curr = state
            while curr != start_state:
                p_state, op = prev[curr]
                ops.append(op)
                curr = p_state
            ops.reverse()
            return ops
            
        # Generate neighbors
        # 1. Compute: vertex not in fast or black, all parents in fast
        for v in all_vertices:
            if v in fast or v in black:
                continue
            ps = parents.get(v, [])
            if all(p in fast for p in ps):
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['compute', v])
                        heapq.heappush(pq, (new_cost, new_state))
                        
        # 2. Store: vertex in fast, not in black
        for v in fast:
            if v not in black:
                new_black = black | frozenset([v])
                new_state = (fast, new_black)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = (state, ['store', v])
                    heapq.heappush(pq, (new_cost, new_state))
                    
        # 3. Delete: vertex in fast
        for v in fast:
            new_fast = fast - frozenset([v])
            new_state = (new_fast, black)
            new_cost = cost
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                prev[new_state] = (state, ['delete', v])
                heapq.heappush(pq, (new_cost, new_state))
                
        # 4. Load: vertex in black, not in fast
        for v in black:
            if v not in fast:
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost + 1
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['load', v])
                        heapq.heappush(pq, (new_cost, new_state))

    return []

# EVOLVE-BLOCK-END
