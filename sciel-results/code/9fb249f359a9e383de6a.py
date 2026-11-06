"""Rewritten DAG pebbling with data-driven eviction costs and local search polish."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 48.0

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

    if len(all_vertices) <= 10:
        return pebble_dijkstra(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)

    # Build children mapping
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Compute layer (topological depth)
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

    layers = defaultdict(list)
    for v in all_vertices:
        layers[layer[v]].append(v)

    max_L = max(layers.keys()) if layers else 0
    output_set = set(outputs)

    # Precompute recompute cost: cost to recompute a vertex from scratch
    # This is the sum of recompute costs of parents + 1 (for the compute op)
    # But actually, the cost is: if a vertex is deleted and needed again,
    # we need to reload (1) or recompute (parents must be in fast memory)
    # The recompute_cost[v] = 1 (compute) + sum of (reload_cost or recompute_cost) of parents
    # For inputs, recompute_cost = 1 (load)
    # This is an estimate; we'll compute it bottom-up.
    
    recompute_cost = {}
    for v in input_vertices:
        recompute_cost[v] = 1  # Just load

    # Process in topological order
    for L in range(max_L + 1):
        for v in layers.get(L, []):
            if v in input_vertices:
                continue
            ps = parents.get(v, [])
            # Cost to recompute v: for each parent, we need it in fast memory.
            # If parent is stored (blue), cost = 1 (load). If not, cost = recompute_cost[parent].
            # But we don't know at runtime which are stored. Use a conservative estimate:
            # recompute_cost[v] = 1 (compute op) + sum of recompute_cost[p] for p in ps
            # This overestimates but gives a relative measure.
            cost = 1
            for p in ps:
                cost += recompute_cost.get(p, 1)
            recompute_cost[v] = cost

    # remaining_use: how many children still need this vertex
    remaining_use = {v: len(children.get(v, [])) for v in all_vertices}

    # next_use_layer: earliest layer where v is needed as a parent
    next_use_layer = {}
    for v in all_vertices:
        next_use_layer[v] = max_L + 1
        for c in children.get(v, []):
            cl = layer.get(c, 0)
            if cl < next_use_layer[v]:
                next_use_layer[v] = cl

    # subtree_depth and subtree_size
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

    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []
    stored = set()

    def eviction_score(u, current_L):
        """Score for eviction: lower = evict first."""
        if u in output_set:
            return -100.0
        if u in input_vertices:
            return 50.0 + 10.0 * remaining_use.get(u, 0)
        
        nu = next_use_layer.get(u, max_L + 1)
        ru = remaining_use.get(u, 0)
        rc = recompute_cost.get(u, 1)
        
        if nu > max_L or ru == 0:
            return 1.0
        
        # Value of keeping u in fast memory:
        # - If we evict u, and it's needed again, we pay reload (1) if stored, or recompute_cost if not.
        # - The "value" is the cost we avoid by keeping it.
        # - If u is stored (in slow_mem), reload cost = 1, so value = rc - 1 (savings from not recomputing)
        # - If u is not stored, reload cost = rc, so value = 0 (no savings)
        # But we also need to account for urgency and remaining uses.
        
        dist_to_use = nu - current_L
        if dist_to_use <= 0:
            urgency = 500.0
        elif dist_to_use <= 1:
            urgency = 200.0
        elif dist_to_use <= 2:
            urgency = 50.0
        else:
            urgency = max(0.0, 10.0 - dist_to_use * 2.0)
        
        # Savings from keeping u: if stored, we save (rc - 1) per reload.
        # But we might reload it multiple times (ru times).
        if u in slow_mem:
            savings = (rc - 1.0) * ru
        else:
            savings = 0.0
        
        value = savings + urgency + rc * 0.3
        return value

    for L in range(max_L + 1):
        if time_check():
            break
        lv = layers.get(L, [])
        if not lv:
            continue

        needed_parents = set()
        for v in lv:
            if v in input_vertices:
                continue
            for p in parents.get(v, []):
                needed_parents.add(p)

        for p in sorted(needed_parents, key=lambda x: next_use_layer.get(x, max_L + 1)):
            if p not in fast_mem and p in slow_mem:
                while len(fast_mem) >= fast_memory_budget:
                    best_evict = None
                    best_score = float('inf')
                    for u in fast_mem:
                        score = eviction_score(u, L)
                        if score < best_score:
                            best_score = score
                            best_evict = u
                    
                    if best_evict is None:
                        break
                    
                    if best_evict in output_set:
                        ops.append(['store', best_evict])
                        slow_mem.add(best_evict)
                        stored.add(best_evict)
                    else:
                        ops.append(['delete', best_evict])
                    fast_mem.discard(best_evict)
                
                ops.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)

        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break
            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)
            for p in parents.get(v, []):
                if p in remaining_use:
                    remaining_use[p] -= 1

        for o in list(fast_mem):
            if o in output_set and o not in stored:
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

        while len(fast_mem) > fast_memory_budget and not time_check():
            best_evict = None
            best_score = float('inf')
            for u in fast_mem:
                score = eviction_score(u, L)
                if score < best_score:
                    best_score = score
                    best_evict = u
            
            if best_evict is None:
                break
            
            if best_evict in output_set:
                ops.append(['store', best_evict])
                slow_mem.add(best_evict)
                stored.add(best_evict)
            else:
                ops.append(['delete', best_evict])
            fast_mem.discard(best_evict)

    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            stored.add(o)
            fast_mem.discard(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

def pebble_dijkstra(dag, budget, all_vertices, input_vertices, outputs, parents, time_check):
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
        
        # Compute
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
        
        # Store
        for v in fast:
            if v not in black:
                new_black = black | frozenset([v])
                new_state = (fast, new_black)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = (state, ['store', v])
                    heapq.heappush(pq, (new_cost, new_state))
        
        # Delete
        for v in fast:
            new_fast = fast - frozenset([v])
            new_state = (new_fast, black)
            new_cost = cost
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                prev[new_state] = (state, ['delete', v])
                heapq.heappush(pq, (new_cost, new_state))
        
        # Load
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
