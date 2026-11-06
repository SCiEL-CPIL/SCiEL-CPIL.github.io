# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

_DEADLINE_SECONDS = 82.0


def _toposort_vertices(dag):
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
        parents = {k: list(v) for k, v in dag['parents'].items()}
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
    """Count load+store operations (the metric that matters)."""
    return sum(1 for op, v in ops if op in ('load', 'store'))


def _dijkstra_pebble_bitmask(dag, fast_memory_budget, time_limit=30.0):
    """Dijkstra over bitmask states for small/medium instances with A* heuristic."""
    start_time = time.time()
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    n = len(vertices)
    
    if n > 24:
        return None

    v_to_idx = {v: i for i, v in enumerate(vertices)}
    idx_to_v = {i: v for i, v in enumerate(vertices)}

    inputs_mask = 0
    for v in inputs:
        inputs_mask |= (1 << v_to_idx[v])
    
    outputs_mask = 0
    for v in outputs:
        outputs_mask |= (1 << v_to_idx[v])
    
    computed_mask = (~inputs_mask) & ((1 << n) - 1)

    parent_masks = []
    for v in vertices:
        ps = parents.get(v, [])
        mask = 0
        for p in ps:
            mask |= (1 << v_to_idx[p])
        parent_masks.append(mask)

    init_blue = inputs_mask
    init_fast = 0
    init_state = (init_blue, init_fast)

    def is_goal(blue, fast):
        return (blue & outputs_mask) == outputs_mask

    if is_goal(init_blue, init_fast):
        return []

    # A* heuristic: minimum additional loads/stores needed
    def heuristic(blue, fast):
        missing_outputs = outputs_mask & ~blue
        h = missing_outputs.bit_count()
        return h

    best_cost = {}
    best_cost[init_state] = 0
    parent_map = {}
    init_h = heuristic(init_blue, init_fast)
    pq = [(init_h, 0, init_state)]  # (f, g, state) where f = g + h

    max_states = 500000
    states_explored = 0
    best_found = None
    best_found_cost = float('inf')

    while pq:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if states_explored >= max_states:
            break

        f, g, state = heapq.heappop(pq)
        if g > best_cost.get(state, float('inf')):
            continue

        states_explored += 1
        blue, fast = state

        if is_goal(blue, fast):
            if g < best_found_cost:
                best_found_cost = g
                best_found = state
                break

        available = blue | fast

        # 1. load(v)
        load_candidates = inputs_mask & ~fast
        while load_candidates:
            bit = load_candidates & (-load_candidates)
            v_idx = bit.bit_length() - 1
            load_candidates ^= bit
            new_fast = fast | bit
            if new_fast.bit_count() <= fast_memory_budget:
                new_state = (blue, new_fast)
                new_g = g + 1
                if new_g < best_cost.get(new_state, float('inf')):
                    best_cost[new_state] = new_g
                    parent_map[new_state] = (state, 'load', idx_to_v[v_idx])
                    new_f = new_g + heuristic(blue, new_fast)
                    heapq.heappush(pq, (new_f, new_g, new_state))

        # 2. store(v)
        store_candidates = fast
        while store_candidates:
            bit = store_candidates & (-store_candidates)
            v_idx = bit.bit_length() - 1
            store_candidates ^= bit
            new_fast = fast & ~bit
            new_blue = blue | bit
            new_state = (new_blue, new_fast)
            new_g = g + 1
            if new_g < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_g
                parent_map[new_state] = (state, 'store', idx_to_v[v_idx])
                new_f = new_g + heuristic(new_blue, new_fast)
                heapq.heappush(pq, (new_f, new_g, new_state))

        # 3. compute(v)
        if fast.bit_count() < fast_memory_budget:
            compute_candidates = computed_mask & ~fast
            while compute_candidates:
                bit = compute_candidates & (-compute_candidates)
                v_idx = bit.bit_length() - 1
                compute_candidates ^= bit
                pm = parent_masks[v_idx]
                if (pm & ~available) == 0:
                    new_fast = fast | bit
                    new_state = (blue, new_fast)
                    new_g = g  # compute is free
                    if new_g < best_cost.get(new_state, float('inf')):
                        best_cost[new_state] = new_g
                        parent_map[new_state] = (state, 'compute', idx_to_v[v_idx])
                        new_f = new_g + heuristic(blue, new_fast)
                        heapq.heappush(pq, (new_f, new_g, new_state))

        # 4. delete(v)
        delete_candidates = fast
        while delete_candidates:
            bit = delete_candidates & (-delete_candidates)
            v_idx = bit.bit_length() - 1
            delete_candidates ^= bit
            new_fast = fast & ~bit
            new_state = (blue, new_fast)
            new_g = g  # delete is free
            if new_g < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_g
                parent_map[new_state] = (state, 'delete', idx_to_v[v_idx])
                new_f = new_g + heuristic(blue, new_fast)
                heapq.heappush(pq, (new_f, new_g, new_state))

    if best_found is not None:
        path = []
        cur = best_found
        while cur != init_state:
            prev, op, v = parent_map[cur]
            path.append((op, v))
            cur = prev
        path.reverse()
        return path

    return None


def _local_search_polish(ops, dag, fast_memory_budget, time_limit=5.0):
    """Reduce transfers by removing redundant operations and reordering."""
    start_time = time.time()
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    if not ops:
        return ops
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        new_ops = list(ops)
        
        # Pattern 1: delete(v) immediately followed by load(v) where v is an input
        i = 0
        while i < len(new_ops) - 1:
            op1, v1 = new_ops[i]
            op2, v2 = new_ops[i+1]
            if op1 == 'delete' and op2 == 'load' and v1 == v2 and v1 in inputs:
                del new_ops[i]
                del new_ops[i]
                improved = True
                continue
            i += 1
        
        # Pattern 2: load(v) immediately followed by delete(v) where v is an input
        i = 0
        while i < len(new_ops) - 1:
            op1, v1 = new_ops[i]
            op2, v2 = new_ops[i+1]
            if op1 == 'load' and op2 == 'delete' and v1 == v2 and v1 in inputs:
                v_used_later = False
                for j in range(i + 2, len(new_ops)):
                    opj, vj = new_ops[j]
                    if vj == v1 and opj in ('compute', 'store'):
                        if v1 in parents.get(vj, []):
                            v_used_later = True
                            break
                if not v_used_later:
                    del new_ops[i]
                    del new_ops[i]
                    improved = True
                    continue
            i += 1
        
        # Pattern 3: Remove trailing deletes that aren't needed
        last_store_idx = -1
        for i, (op, v) in enumerate(new_ops):
            if op == 'store':
                last_store_idx = i
        
        if last_store_idx >= 0 and last_store_idx < len(new_ops) - 1:
            new_ops = new_ops[:last_store_idx+1]
            improved = True
        
        # Pattern 4: Move stores earlier when possible to free memory
        new_ops2 = []
        fast_sim = set()
        blue_sim = set(inputs)
        available_sim = set(inputs)
        
        stored_outputs = set()
        
        for op, v in new_ops:
            if op == 'load':
                new_ops2.append((op, v))
                fast_sim.add(v)
                available_sim.add(v)
            elif op == 'store':
                new_ops2.append((op, v))
                if v in fast_sim:
                    fast_sim.discard(v)
                blue_sim.add(v)
                stored_outputs.add(v)
            elif op == 'compute':
                new_ops2.append((op, v))
                fast_sim.add(v)
                available_sim.add(v)
            elif op == 'delete':
                new_ops2.append((op, v))
                fast_sim.discard(v)
                if v not in blue_sim:
                    available_sim.discard(v)
        
        # Second pass: try to move stores earlier
        final_ops = []
        fast_sim2 = set()
        blue_sim2 = set(inputs)
        available_sim2 = set(inputs)
        stored_outputs2 = set()
        
        for op, v in new_ops2:
            if op == 'load':
                final_ops.append((op, v))
                fast_sim2.add(v)
                available_sim2.add(v)
            elif op == 'compute':
                final_ops.append((op, v))
                fast_sim2.add(v)
                available_sim2.add(v)
                if v in outputs and v not in stored_outputs2:
                    if len(fast_sim2) >= fast_memory_budget - 1:
                        final_ops.append(('store', v))
                        fast_sim2.discard(v)
                        blue_sim2.add(v)
                        stored_outputs2.add(v)
            elif op == 'store':
                if v not in stored_outputs2:
                    final_ops.append((op, v))
                    if v in fast_sim2:
                        fast_sim2.discard(v)
                    blue_sim2.add(v)
                    stored_outputs2.add(v)
            elif op == 'delete':
                final_ops.append((op, v))
                fast_sim2.discard(v)
                if v not in blue_sim2:
                    available_sim2.discard(v)
        
        if _validate_ops(final_ops, dag, fast_memory_budget):
            if _count_transfers(final_ops) < _count_transfers(new_ops2):
                new_ops = final_ops
                improved = True
        
        ops = new_ops
    
    return ops


def _greedy_pebble_improved(dag, fast_memory_budget):
    """Improved greedy with smarter eviction: prioritize keeping computed values over inputs."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    children = {}
    for v in vertices:
        for p in parents.get(v, []):
            children.setdefault(p, []).append(v)
    
    v_to_pos = {v: i for i, v in enumerate(vertices)}
    
    remaining_children = {}
    for v in vertices:
        count = 0
        for c in children.get(v, []):
            if v_to_pos.get(c, 0) > v_to_pos[v]:
                count += 1
        remaining_children[v] = count

    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    
    def get_eviction_priority(v):
        rc = remaining_children.get(v, 0)
        is_input = 1 if v in inputs else 0
        is_output = 1 if v in outputs else 0
        return (rc, is_input, -is_output)
    
    for v in vertices:
        if v in inputs:
            if v not in available:
                if len(fast_mem) >= fast_memory_budget:
                    best_evict = None
                    best_priority = None
                    for fm in fast_mem:
                        pr = get_eviction_priority(fm)
                        if best_priority is None or pr < best_priority:
                            best_priority = pr
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
            ps = parents.get(v, [])
            
            if len(fast_mem) >= fast_memory_budget:
                best_evict = None
                best_priority = None
                for fm in fast_mem:
                    pr = get_eviction_priority(fm)
                    if best_priority is None or pr < best_priority:
                        best_priority = pr
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
        
        for p in parents.get(v, []):
            remaining_children[p] = remaining_children.get(p, 0) - 1
        
        if v in outputs and v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
    
    for v in list(fast_mem):
        ops.append(('delete', v))
        fast_mem.discard(v)
    
    return ops


def _layered_greedy_pebble(dag, fast_memory_budget):
    """Layer-based greedy as fallback."""
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    
    depth = {}
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in ps) + 1
    
    max_depth = max(depth.values()) if depth else 0
    layers = {}
    for v in vertices:
        layers.setdefault(depth[v], []).append(v)
    
    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    remaining_outputs = set(outputs)
    
    for d in range(max_depth + 1):
        layer_vertices = layers.get(d, [])
        for v in layer_vertices:
            if v in inputs:
                if v not in fast_mem:
                    if len(fast_mem) >= fast_memory_budget:
                        to_delete = None
                        for fm in fast_mem:
                            if fm not in outputs and fm not in remaining_outputs:
                                to_delete = fm
                                break
                        if to_delete is None and fast_mem:
                            to_delete = next(iter(fast_mem))
                        if to_delete:
                            ops.append(('delete', to_delete))
                            fast_mem.discard(to_delete)
                            if to_delete not in blue:
                                available.discard(to_delete)
                    if len(fast_mem) < fast_memory_budget:
                        ops.append(('load', v))
                        fast_mem.add(v)
                        available.add(v)
            else:
                ps = parents.get(v, [])
                if len(fast_mem) >= fast_memory_budget:
                    to_delete = None
                    for fm in fast_mem:
                        if fm not in outputs and fm not in remaining_outputs:
                            to_delete = fm
                            break
                    if to_delete is None and fast_mem:
                        to_delete = next(iter(fast_mem))
                    if to_delete:
                        ops.append(('delete', to_delete))
                        fast_mem.discard(to_delete)
                        if to_delete not in blue:
                            available.discard(to_delete)
                if len(fast_mem) < fast_memory_budget:
                    ops.append(('compute', v))
                    fast_mem.add(v)
                    available.add(v)
        
        for v in layer_vertices:
            if v in outputs and v in fast_mem:
                ops.append(('store', v))
                fast_mem.discard(v)
                blue.add(v)
                remaining_outputs.discard(v)
    
    for v in remaining_outputs:
        if v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
    
    for v in list(fast_mem):
        ops.append(('delete', v))
    
    return ops


def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    try:
        vertices, parents, outputs, inputs = _toposort_vertices(dag)
        n_vertices = len(vertices)
        
        # For small/medium instances, try exact A* with bitmask
        if n_vertices <= 24:
            time_budget = min(40.0, 2.0 * n_vertices)
            remaining_time = _DEADLINE_SECONDS - (time.time() - start_time)
            time_budget = min(time_budget, remaining_time - 12.0)
            
            if time_budget > 2.0:
                result = _dijkstra_pebble_bitmask(dag, fast_memory_budget, time_limit=time_budget)
                if result is not None and len(result) <= 20000:
                    if _validate_ops(result, dag, fast_memory_budget):
                        remaining_for_polish = min(5.0, _DEADLINE_SECONDS - (time.time() - start_time) - 8.0)
                        if remaining_for_polish > 1.0:
                            polished = _local_search_polish(result, dag, fast_memory_budget, time_limit=remaining_for_polish)
                            if _validate_ops(polished, dag, fast_memory_budget) and len(polished) <= 20000:
                                return polished
                        return result
        
        # For larger instances or if A* failed, use improved greedy
        remaining_time = _DEADLINE_SECONDS - (time.time() - start_time)
        if remaining_time < 8.0:
            try:
                from autoresearch_suite.science_tcs_expansion import pebble_baseline
                return pebble_baseline(dag, fast_memory_budget)
            except Exception:
                ops = []
                for v in vertices:
                    if v in inputs:
                        ops.append(('load', v))
                    else:
                        ops.append(('compute', v))
                for v in outputs:
                    ops.append(('store', v))
                return ops
        
        # Try multiple greedy strategies and pick the best
        candidates = []
        
        result1 = _greedy_pebble_improved(dag, fast_memory_budget)
        if result1 and _validate_ops(result1, dag, fast_memory_budget) and len(result1) <= 20000:
            candidates.append(result1)
        
        result2 = _layered_greedy_pebble(dag, fast_memory_budget)
        if result2 and _validate_ops(result2, dag, fast_memory_budget) and len(result2) <= 20000:
            candidates.append(result2)
        
        best_ops = None
        best_transfers = float('inf')
        
        remaining_for_polish = min(5.0, _DEADLINE_SECONDS - (time.time() - start_time) - 8.0)
        polish_time_each = max(1.0, remaining_for_polish / max(1, len(candidates)))
        
        for cand in candidates:
            polished = _local_search_polish(cand, dag, fast_memory_budget, time_limit=polish_time_each)
            if _validate_ops(polished, dag, fast_memory_budget) and len(polished) <= 20000:
                transfers = _count_transfers(polished)
                if transfers < best_transfers:
                    best_transfers = transfers
                    best_ops = polished
        
        if best_ops is not None:
            return best_ops
        
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            ops = []
            for v in vertices:
                if v in inputs:
                    ops.append(('load', v))
                else:
                    ops.append(('compute', v))
            for v in outputs:
                ops.append(('store', v))
            return ops
    
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            try:
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
            except Exception:
                return []

# EVOLVE-BLOCK-END
