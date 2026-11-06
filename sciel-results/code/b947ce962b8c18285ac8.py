"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 50.0

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

    # If the DAG is small, use Dijkstra for optimal solution
    if len(all_vertices) <= 10:
        result = pebble_dijkstra(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)
        if result:
            return result

    # Otherwise, use improved greedy
    # Compute layer for each vertex
    layer = {}
    for v in input_vertices:
        layer[v] = 0

    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

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

    layers = defaultdict(list)
    for v in all_vertices:
        layers[layer[v]].append(v)

    max_L = max(layers.keys()) if layers else 0

    # Precompute usage info for better eviction heuristics
    use_count = {v: len(children.get(v, [])) for v in all_vertices}
    # remaining_use: how many children still need this vertex (decreases as we process layers)
    remaining_use = {v: use_count[v] for v in all_vertices}
    next_use_layer = {}
    for v in all_vertices:
        next_use_layer[v] = max_L + 1
        for c in children.get(v, []):
            cl = layer.get(c, 0)
            if cl < next_use_layer[v]:
                next_use_layer[v] = cl

    # Compute subtree depth and size for cost estimation
    subtree_depth = {v: 0 for v in all_vertices}
    subtree_size = {v: 1 for v in all_vertices}
    for L in range(max_L, -1, -1):
        for v in layers.get(L, []):
            if not children.get(v):
                continue
            max_d = 0
            total_s = 1
            for c in children.get(v, []):
                if c in subtree_depth:
                    max_d = max(max_d, subtree_depth[c] + 1)
                    total_s += subtree_size.get(c, 1)
            subtree_depth[v] = max_d
            subtree_size[v] = total_s

    # Determine which vertices are outputs (sinks)
    output_set = set(outputs)

    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []
    
    # Track which vertices have been stored (blue)
    stored = set()

    def evict_one(L):
        """Evict one vertex from fast_mem to make room. Returns True if evicted."""
        best_evict = None
        best_score = float('inf')
        for u in fast_mem:
            if u in output_set:
                # Outputs should be stored, not deleted
                score = -100
            elif u in input_vertices:
                # Input vertices: reload is always possible, but if used many times, keep
                score = 100 + 10 * remaining_use.get(u, 0)
            else:
                nu = next_use_layer.get(u, max_L + 1)
                ru = remaining_use.get(u, 0)
                size = subtree_size.get(u, 1)
                depth = subtree_depth.get(u, 0)
                
                if nu > max_L or ru == 0:
                    # No future use, safe to evict
                    score = 1.0
                else:
                    # Value of keeping in fast memory
                    # If we evict and it's not stored, we'd need to recompute (cost ~ subtree_size)
                    # If it's stored, we'd need to load (cost 1)
                    # Urgency: how soon is it needed again?
                    # Reuse: how many times is it needed?
                    
                    # Cost if evicted and not stored: recompute cost
                    recompute_cost = size
                    # Cost if evicted and stored: load cost
                    load_cost = 1 if u in stored else recompute_cost
                    
                    # Value = benefit of keeping in fast vs. cost of evicting
                    # If kept: 0 extra cost
                    # If evicted: load_cost or recompute_cost
                    # But also consider: if we keep, we might not have room for something more valuable
                    
                    # Simple heuristic: value proportional to (recompute_cost - load_cost) * reuse
                    # Plus urgency bonus
                    value = 0.0
                    value += (recompute_cost - load_cost) * ru * 5.0
                    value += recompute_cost * 0.5
                    value += depth * 1.0
                    
                    dist_to_use = nu - L
                    if dist_to_use <= 0:
                        value += 1000.0
                    elif dist_to_use == 1:
                        value += 300.0
                    elif dist_to_use == 2:
                        value += 100.0
                    else:
                        value += max(0, 50.0 - dist_to_use * 5.0)
                    
                    score = value

            if score < best_score:
                best_score = score
                best_evict = u
        
        if best_evict is None:
            return False
        
        if best_evict in output_set and best_evict not in stored:
            ops.append(['store', best_evict])
            slow_mem.add(best_evict)
            stored.add(best_evict)
        else:
            ops.append(['delete', best_evict])
        
        fast_mem.discard(best_evict)
        return True

    for L in range(max_L + 1):
        if time_check():
            break
        lv = layers.get(L, [])
        if not lv:
            continue

        # Load needed parents just-in-time
        needed_parents = set()
        for v in lv:
            if v in input_vertices:
                continue
            for p in parents.get(v, []):
                needed_parents.add(p)

        # Sort needed parents by next_use_layer to load most urgent first
        for p in sorted(needed_parents, key=lambda x: (next_use_layer.get(x, max_L + 1), -remaining_use.get(x, 0))):
            if p not in fast_mem and p in slow_mem:
                # Need to make room if over budget
                while len(fast_mem) >= fast_memory_budget:
                    if not evict_one(L):
                        break
                    if time_check():
                        break
                
                if p not in fast_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)

        # Compute vertices in this layer
        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break
            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)
            # Decrement remaining_use for parents
            for p in parents.get(v, []):
                if p in remaining_use:
                    remaining_use[p] -= 1

        # Proactively store outputs that are no longer needed
        for o in list(fast_mem):
            if o in output_set and o not in stored:
                # Check if any future layer needs this output as a parent
                future_needed = False
                for c in children.get(o, []):
                    if layer.get(c, -1) > L:
                        future_needed = True
                        break
                if not future_needed:
                    ops.append(['store', o])
                    slow_mem.add(o)
                    stored.add(o)
                    fast_mem.discard(o)

        # Evict if still over budget after proactive stores
        while len(fast_mem) > fast_memory_budget and not time_check():
            if not evict_one(L):
                break

    # Store any remaining outputs
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            stored.add(o)
            fast_mem.discard(o)
        elif o not in stored and o not in slow_mem:
            # Output was deleted without being stored - this shouldn't happen
            # but if it does, we need to recompute. For now, skip.
            pass

    # Post-processing: remove redundant operations
    # Remove consecutive duplicate loads/stores
    cleaned_ops = []
    for op in ops:
        if cleaned_ops and cleaned_ops[-1] == op:
            continue
        cleaned_ops.append(op)
    ops = cleaned_ops

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

def pebble_dijkstra(dag, budget, all_vertices, input_vertices, outputs, parents, time_check):
    # State: (frozenset(fast), frozenset(black))
    # Initial: fast = set(), black = input_vertices
    init_black = frozenset(input_vertices)
    init_fast = frozenset()
    
    # Dijkstra
    dist = {}
    prev = {} # state -> (prev_state, op)
    
    start_state = (init_fast, init_black)
    dist[start_state] = 0
    
    pq = [(0, start_state)]
    
    while pq:
        if time_check():
            return []
        cost, state = heapq.heappop(pq)
        fast, black = state
        
        if cost > dist.get(state, float('inf')):
            continue
            
        # Check if goal reached
        if all(o in black for o in outputs):
            # Reconstruct path
            ops = []
            curr = state
            while curr != start_state:
                p_state, op = prev[curr]
                ops.append(op)
                curr = p_state
            ops.reverse()
            return ops
            
        # Generate neighbors
        # 1. Compute
        for v in all_vertices:
            if v in fast or v in black:
                continue
            ps = parents.get(v, [])
            if all(p in fast or p in black for p in ps):
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['compute', v])
                        heapq.heappush(pq, (new_cost, new_state))
                        
        # 2. Store
        for v in fast:
            if v not in black:
                new_black = black | frozenset([v])
                new_state = (fast, new_black)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = (state, ['store', v])
                    heapq.heappush(pq, (new_cost, new_state))
                    
        # 3. Delete
        for v in fast:
            new_fast = fast - frozenset([v])
            new_state = (new_fast, black)
            new_cost = cost
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                prev[new_state] = (state, ['delete', v])
                heapq.heappush(pq, (new_cost, new_state))
                
        # 4. Load
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
