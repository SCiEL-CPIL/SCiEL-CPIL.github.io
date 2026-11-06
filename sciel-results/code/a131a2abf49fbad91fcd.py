import time
from collections import defaultdict, deque
import heapq
import sys

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 55  # Safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    if vertices is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
    
    # Local indices for needed vertices
    n_needed = len(needed_vertices)
    v_to_local = {v: i for i, v in enumerate(needed_vertices)}
    
    # Local parents/children
    local_parents = {i: [v_to_local[p] for p in parent_dict.get(needed_vertices[i], []) if p in needed_set] for i in range(n_needed)}
    local_children = {i: [v_to_local[c] for c in children.get(needed_vertices[i], []) if c in needed_set] for i in range(n_needed)}
    
    local_inputs = set(v_to_local[v] for v in inputs if v in needed_set)
    local_outputs = set(v_to_local[v] for v in outputs if v in needed_set)

    # Compute last use index for each local vertex
    last_child_idx = {}
    for i in range(n_needed):
        last_child_idx[i] = -1
        for c in local_children[i]:
            last_child_idx[i] = max(last_child_idx[i], c)

    # Compute future use counts for better eviction decisions
    future_use_counts = {}
    for i in range(n_needed):
        future_use_counts[i] = 0
        for c in local_children[i]:
            if c > i:
                future_use_counts[i] += 1

    # Strategy Selection
    # 1. Dijkstra for very small
    if fast_memory_budget <= 5 and n_needed <= 25:
        result = _dijkstra_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                                  fast_memory_budget, deadline, last_child_idx)
        if result is not None:
            return result

    # 2. Beam Search for medium
    if fast_memory_budget <= 10 and n_needed <= 150:
        result = _beam_search_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                                     fast_memory_budget, deadline, last_child_idx, beam_width=500)
        if result is not None:
            return result

    # 3. Enhanced Greedy for large
    return _greedy_pebble_enhanced(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                          fast_memory_budget, deadline, last_child_idx, future_use_counts)

def _dijkstra_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                     fast_memory_budget, deadline, last_child_idx):
    """
    Dijkstra-based optimal pebbling for small problems.
    State: (pos, frozenset of fast memory, frozenset of computed values)
    Action: load, store, compute, delete
    Cost: 1 for load/store, 0 for compute/delete
    """
    n = len(needed_vertices)
    if n == 0:
        return []
    
    init_fast = frozenset()
    init_computed = frozenset(local_inputs)
    init_state = (0, init_fast, init_computed)
    
    dist = {init_state: 0}
    parent = {}
    pq = [(0, 0, init_fast, init_computed)]
    
    best_final_cost = float('inf')
    best_final_state = None
    
    steps = 0
    max_steps = 200000
    
    while pq:
        if time.time() > deadline:
            break
        if steps >= max_steps:
            break
        steps += 1
        
        cost, pos, fast_mem, computed = heapq.heappop(pq)
        state = (pos, fast_mem, computed)
        
        if dist.get(state, float('inf')) < cost:
            continue
        
        if pos >= n:
            if all(o in fast_mem for o in local_outputs):
                if cost < best_final_cost:
                    best_final_cost = cost
                    best_final_state = state
                break
            continue
        
        v_local = pos
        
        # Action 1: Compute v_local
        parents_local = local_parents.get(v_local, [])
        if all(p in fast_mem for p in parents_local) and v_local not in computed:
            new_computed = computed | frozenset([v_local])
            new_fast = fast_mem | frozenset([v_local])
            if len(new_fast) <= fast_memory_budget:
                new_pos = pos + 1
                new_state = (new_pos, new_fast, new_computed)
                new_cost = cost
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'compute', v_local)
                    heapq.heappush(pq, (new_cost, new_pos, new_fast, new_computed))
        
        # Action 2: Load
        loadable = [u for u in range(n) if u not in fast_mem and (u in local_inputs or u in computed)]
        for u in loadable:
            if len(fast_mem) < fast_memory_budget:
                new_fast = fast_mem | frozenset([u])
                new_state = (pos, new_fast, computed)
                new_cost = cost + 1
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'load', u)
                    heapq.heappush(pq, (new_cost, pos, new_fast, computed))
            else:
                # Evict and Load
                for u_evict in fast_mem:
                    # Can we evict u_evict?
                    # If u_evict is input, delete is free.
                    # If u_evict is computed, store is cost 1.
                    if u_evict in local_inputs:
                        evict_cost = 0
                        evict_op = 'delete'
                    else:
                        evict_cost = 1
                        evict_op = 'store'
                    
                    new_fast = (fast_mem - frozenset([u_evict])) | frozenset([u])
                    new_state = (pos, new_fast, computed)
                    new_cost = cost + 1 + evict_cost
                    if dist.get(new_state, float('inf')) > new_cost:
                        dist[new_state] = new_cost
                        parent[new_state] = (state, f'load_evict_{u_evict}_{evict_op}', u)
                        heapq.heappush(pq, (new_cost, pos, new_fast, computed))
        
        # Action 3: Store
        for u in fast_mem:
            if u in computed and u not in local_inputs:
                new_fast = fast_mem - frozenset([u])
                new_state = (pos, new_fast, computed)
                new_cost = cost + 1
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'store', u)
                    heapq.heappush(pq, (new_cost, pos, new_fast, computed))
        
        # Action 4: Delete
        for u in fast_mem:
            if u in local_inputs:
                new_fast = fast_mem - frozenset([u])
                new_state = (pos, new_fast, computed)
                new_cost = cost
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'delete', u)
                    heapq.heappush(pq, (new_cost, pos, new_fast, computed))
            
            # Can we delete a computed value? Only if we never need it again.
            elif u in computed and last_child_idx[u] < pos:
                new_fast = fast_mem - frozenset([u])
                new_state = (pos, new_fast, computed)
                new_cost = cost
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'delete', u)
                    heapq.heappush(pq, (new_cost, pos, new_fast, computed))

    if best_final_state is None:
        return None
    
    ops = []
    state = best_final_state
    while state in parent:
        prev_state, action, vertex_local = parent[state]
        original_vertex = needed_vertices[vertex_local]
        
        if action == 'compute':
            ops.append(['compute', original_vertex])
        elif action == 'load':
            ops.append(['load', original_vertex])
        elif action.startswith('load_evict_'):
            parts = action.split('_')
            evict_local = int(parts[2])
            evict_op = parts[3]
            evict_vertex = needed_vertices[evict_local]
            ops.append([evict_op, evict_vertex])
            ops.append(['load', original_vertex])
        elif action == 'store':
            ops.append(['store', original_vertex])
        elif action == 'delete':
            ops.append(['delete', original_vertex])
        
        state = prev_state
    
    ops.reverse()
    
    if _verify_solution(ops, needed_vertices, local_parents, local_children, local_inputs, local_outputs, 
                        fast_memory_budget, needed_set=needed_set):
        return ops
    else:
        return None

def _beam_search_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                        fast_memory_budget, deadline, last_child_idx, beam_width=500):
    """
    Beam Search for medium problems.
    """
    n = len(needed_vertices)
    if n == 0:
        return []
    
    init_fast = frozenset()
    init_computed = frozenset(local_inputs)
    init_state = (0, init_fast, init_computed)
    
    # State: (pos, fast_mem, computed)
    # We keep a list of (cost, state, parent_info)
    # To save memory, we don't store full parent chain, but we can't reconstruct path easily.
    # For Beam Search, we usually just keep the best states.
    # To reconstruct, we need to store the path or use a map.
    # Given the size, storing the path in the state is expensive.
    # Alternative: Just find the min cost. If we can't reconstruct, we fail.
    # But we need to return the operations.
    # Let's store the operations list in the state? No, too big.
    # Let's store a pointer to the previous state?
    
    # Since beam width is small, we can store the full path for each beam entry.
    # Or, we can just run the greedy if we can't reconstruct.
    # Let's try to store the path.
    
    # State: (cost, pos, fast_mem, computed, path)
    # Path is a list of (op, vertex)
    
    current_beam = [(0, 0, init_fast, init_computed, [])]
    
    best_final_cost = float('inf')
    best_final_path = None
    
    max_steps = 10000
    steps = 0
    
    while current_beam:
        if time.time() > deadline:
            break
        if steps >= max_steps:
            break
        steps += 1
        
        next_beam = []
        
        for cost, pos, fast_mem, computed, path in current_beam:
            if pos >= n:
                if all(o in fast_mem for o in local_outputs):
                    if cost < best_final_cost:
                        best_final_cost = cost
                        best_final_path = path
                continue
            
            v_local = pos
            new_states = []
            
            # Generate successors
            
            # 1. Compute
            parents_local = local_parents.get(v_local, [])
            if all(p in fast_mem for p in parents_local) and v_local not in computed:
                new_computed = computed | frozenset([v_local])
                new_fast = fast_mem | frozenset([v_local])
                if len(new_fast) <= fast_memory_budget:
                    new_pos = pos + 1
                    new_path = path + [('compute', needed_vertices[v_local])]
                    new_states.append((cost, new_pos, new_fast, new_computed, new_path))
            
            # 2. Load
            loadable = [u for u in range(n) if u not in fast_mem and (u in local_inputs or u in computed)]
            for u in loadable:
                if len(fast_mem) < fast_memory_budget:
                    new_fast = fast_mem | frozenset([u])
                    new_path = path + [('load', needed_vertices[u])]
                    new_states.append((cost + 1, pos, new_fast, computed, new_path))
                else:
                    # Evict and Load
                    for u_evict in fast_mem:
                        if u_evict in local_inputs:
                            evict_cost = 0
                            evict_op = 'delete'
                        else:
                            evict_cost = 1
                            evict_op = 'store'
                        
                        new_fast = (fast_mem - frozenset([u_evict])) | frozenset([u])
                        new_path = path + [(evict_op, needed_vertices[u_evict]), ('load', needed_vertices[u])]
                        new_states.append((cost + 1 + evict_cost, pos, new_fast, computed, new_path))
            
            # 3. Store
            for u in fast_mem:
                if u in computed and u not in local_inputs:
                    new_fast = fast_mem - frozenset([u])
                    new_path = path + [('store', needed_vertices[u])]
                    new_states.append((cost + 1, pos, new_fast, computed, new_path))
            
            # 4. Delete
            for u in fast_mem:
                if u in local_inputs:
                    new_fast = fast_mem - frozenset([u])
                    new_path = path + [('delete', needed_vertices[u])]
                    new_states.append((cost, pos, new_fast, computed, new_path))
                elif u in computed and last_child_idx[u] < pos:
                    new_fast = fast_mem - frozenset([u])
                    new_path = path + [('delete', needed_vertices[u])]
                    new_states.append((cost, pos, new_fast, computed, new_path))
            
            next_beam.extend(new_states)
        
        # Prune next_beam to top K
        if not next_beam:
            break
            
        # Sort by cost, then by length of path (prefer shorter?)
        # We want min cost. Tie-break: maybe larger fast_mem?
        next_beam.sort(key=lambda x: (x[0], len(x[4])))
        
        current_beam = next_beam[:beam_width]
        
        # Early termination if we found a goal?
        # We can't stop immediately because a cheaper path might appear later in the beam.
        # But if the min cost in the beam is already > best_final_cost, we can prune.
        # Actually, we update best_final_cost when we see a goal.
        
    if best_final_path is None:
        return None
        
    # Convert path to list of lists
    ops = [[op, v] for op, v in best_final_path]
    
    if _verify_solution(ops, needed_vertices, local_parents, local_children, local_inputs, local_outputs, 
                        fast_memory_budget, needed_set=needed_set):
        return ops
    else:
        return None

def _verify_solution(ops, needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                     fast_memory_budget, needed_set):
    """Verify that a solution is valid."""
    in_fast = set()
    in_slow = set(local_inputs)
    computed = set(local_inputs)
    
    # Map local indices to original vertices for verification?
    # The ops contain original vertices.
    # We need to check against the DAG structure.
    # Let's use the original structure passed in?
    # Actually, the ops are generated from needed_vertices.
    # We need to check if parents are in fast mem.
    
    # Rebuild original parents/children for verification
    # This is expensive, but verification is only done once at the end.
    
    # Let's just use the local structure and map vertices back.
    v_to_local = {v: i for i, v in enumerate(needed_vertices)}
    
    for op, v in ops:
        if v not in v_to_local:
            return False
        v_local = v_to_local[v]
        
        if op == 'load':
            if v_local not in in_slow:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v_local)
        elif op == 'store':
            if v_local not in in_fast:
                return False
            in_fast.discard(v_local)
            in_slow.add(v_local)
        elif op == 'compute':
            parents_local = local_parents.get(v_local, [])
            if not all(p in in_fast for p in parents_local):
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v_local)
            computed.add(v_local)
        elif op == 'delete':
            if v_local not in in_fast:
                return False
            in_fast.discard(v_local)
    
    for o in local_outputs:
        if o not in in_fast:
            return False
    
    return True

def _greedy_pebble_enhanced(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                   fast_memory_budget, deadline, last_child_idx, future_use_counts):
    """Enhanced greedy pebbling with better eviction strategy."""
    n = len(needed_vertices)
    in_fast = set()
    in_slow = set(local_inputs)
    computed = set(local_inputs)
    operations = []

    def load(v):
        operations.append(['load', needed_vertices[v]])
        in_fast.add(v)
        
    def store(v):
        operations.append(['store', needed_vertices[v]])
        in_fast.discard(v)
        in_slow.add(v)
        
    def compute(v):
        operations.append(['compute', needed_vertices[v]])
        computed.add(v)
        in_fast.add(v)
        
    def delete(v):
        operations.append(['delete', needed_vertices[v]])
        in_fast.discard(v)

    def get_eviction_score(u, pos, parents_needed):
        """Calculate a score for how good it is to evict u. Higher score = better to evict."""
        score = 0.0
        u_last = last_child_idx.get(u, -1)
        
        # If no future uses, very high score
        if u_last < pos:
            score += 1000.0
        else:
            # Future uses count
            future_uses = future_use_counts.get(u, 0)
            score -= future_uses * 15.0
            
            # If it's a parent we need soon, lower score
            if u in parents_needed:
                score -= 50.0
        
        # If in slow, delete is free (better to evict)
        if u in in_slow:
            score += 80.0
        else:
            # If not in slow, store costs 1 (worse to evict)
            score -= 30.0
        
        # If it's an input, delete is always free
        if u in local_inputs:
            score += 20.0
            
        return score

    for pos in range(n):
        if time.time() > deadline:
            break
            
        v = pos
        parents = local_parents.get(v, [])

        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            
            if len(in_fast) >= fast_memory_budget:
                # Evict
                best_evict = None
                best_score = -float('inf')
                
                for u in list(in_fast):
                    if u == v: continue
                    if u in parents: continue # Don't evict parents we are loading
                    
                    score = get_eviction_score(u, pos, parents)
                    
                    if score > best_score:
                        best_score = score
                        best_evict = u

                if best_evict is not None:
                    if best_evict in in_slow:
                        delete(best_evict)
                    else:
                        store(best_evict)

            load(p)

        # Compute v
        if v not in in_fast:
            if v in local_inputs:
                if len(in_fast) >= fast_memory_budget:
                    # Evict
                    best_evict = None
                    best_score = -float('inf')
                    for u in list(in_fast):
                        if u == v: continue
                        score = get_eviction_score(u, pos, parents)
                        if score > best_score:
                            best_score = score
                            best_evict = u
                    if best_evict is not None:
                        if best_evict in in_slow:
                            delete(best_evict)
                        else:
                            store(best_evict)
                load(v)
            else:
                compute(v)

        # Proactive cleanup: evict items that are no longer needed
        for u in list(in_fast):
            u_last = last_child_idx.get(u, -1)
            if u_last < pos and u not in local_outputs:
                if u in in_slow:
                    delete(u)
                else:
                    store(u)

    # Ensure outputs in fast
    for v in local_outputs:
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                best_evict = None
                best_score = -float('inf')
                for u in list(in_fast):
                    if u in local_outputs: continue
                    score = get_eviction_score(u, n, set())
                    if score > best_score:
                        best_score = score
                        best_evict = u
                if best_evict is not None:
                    if best_evict in in_slow:
                        delete(best_evict)
                    else:
                        store(best_evict)
            load(v)

    return operations
