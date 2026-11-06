"""Rewritten program for Communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq

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

    # Compute last use index for each vertex
    last_child_idx = {}
    for v in needed_vertices:
        last_child_idx[v] = -1
        for child in children.get(v, []):
            if child in needed_set:
                last_child_idx[v] = max(last_child_idx[v], vertex_to_idx.get(child, -1))

    # Compute first use index for each vertex
    first_child_idx = {}
    for v in needed_vertices:
        first_child_idx[v] = num_vertices
        for child in children.get(v, []):
            if child in needed_set:
                first_child_idx[v] = min(first_child_idx[v], vertex_to_idx.get(child, num_vertices))

    # For small problems, use Dijkstra-based optimal search
    if fast_memory_budget <= 6 and len(needed_vertices) <= 30:
        result = _dijkstra_pebble(needed_vertices, parent_dict, children, inputs, outputs,
                                  fast_memory_budget, deadline, vertex_to_idx, needed_set,
                                  last_child_idx, first_child_idx)
        if result is not None:
            return result

    # Fallback to improved greedy
    return _greedy_pebble(needed_vertices, parent_dict, children, inputs, outputs,
                          fast_memory_budget, deadline, vertex_to_idx, needed_set,
                          last_child_idx, first_child_idx)

def _dijkstra_pebble(needed_vertices, parent_dict, children, inputs, outputs,
                     fast_memory_budget, deadline, vertex_to_idx, needed_set,
                     last_child_idx, first_child_idx):
    """
    Dijkstra-based optimal pebbling for small problems.
    State: (position, frozenset of fast memory, frozenset of computed values)
    Action: load, store, compute, delete
    Cost: 1 for load/store, 0 for compute/delete
    """
    n = len(needed_vertices)
    if n == 0:
        return []

    v_to_local = {v: i for i, v in enumerate(needed_vertices)}
    local_parents = {i: [v_to_local[p] for p in parent_dict.get(needed_vertices[i], []) if p in needed_set] for i in range(n)}
    local_children = {i: [] for i in range(n)}
    for i in range(n):
        for c in children.get(needed_vertices[i], []):
            if c in needed_set:
                local_children[i].append(v_to_local[c])

    local_inputs = set(v_to_local[v] for v in inputs if v in needed_set)
    local_outputs = set(v_to_local[v] for v in outputs if v in needed_set)

    # State: (pos, fast_mem_frozen, computed_frozen)
    # pos: next vertex index to process (0 to n)
    # We'll use a priority queue with (cost, state)
    
    # Initial state
    init_fast = frozenset()
    init_computed = frozenset(local_inputs)
    init_state = (0, init_fast, init_computed)
    
    # Dijkstra
    # dist[state] = min cost to reach state
    dist = {init_state: 0}
    # parent tracking for reconstruction
    parent = {}
    
    # Priority queue: (cost, pos, fast_mem, computed)
    pq = [(0, 0, init_fast, init_computed)]
    
    best_final_cost = float('inf')
    best_final_state = None
    
    steps = 0
    max_steps = 500000
    
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
        
        # Check if we've processed all vertices
        if pos >= n:
            # Check if all outputs are in fast memory
            all_outputs_in_fast = all(o in fast_mem for o in local_outputs)
            if all_outputs_in_fast:
                if cost < best_final_cost:
                    best_final_cost = cost
                    best_final_state = state
                # We can continue searching for potentially better paths,
                # but since Dijkstra pops in order, this is the best so far
                # We could break here, but let's continue for now
                # Actually, since we're using Dijkstra, the first time we pop a 
                # goal state is optimal. So we can break.
                break
            continue
        
        v_local = pos  # The vertex at position pos in needed_vertices
        
        # Generate successor states
        # Possible actions:
        # 1. Compute v_local if all parents are in fast_mem
        # 2. Load a vertex not in fast_mem but in computed or inputs
        # 3. Store a vertex in fast_mem that is in computed (moves to slow)
        # 4. Delete a vertex in fast_mem that is in slow (removes from fast, no cost)
        
        # Action 1: Compute v_local
        parents_local = local_parents.get(v_local, [])
        all_parents_in_fast = all(p in fast_mem for p in parents_local)
        
        if all_parents_in_fast and v_local not in computed:
            new_computed = computed | frozenset([v_local])
            new_fast = fast_mem | frozenset([v_local])
            if len(new_fast) <= fast_memory_budget:
                new_pos = pos + 1
                new_state = (new_pos, new_fast, new_computed)
                new_cost = cost  # compute is free
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'compute', v_local)
                    heapq.heappush(pq, (new_cost, new_pos, new_fast, new_computed))
        
        # Action 2: Load a vertex
        # A vertex can be loaded if it's in inputs or in computed (and not in fast_mem)
        loadable = []
        for u in range(n):
            if u not in fast_mem:
                if u in local_inputs or u in computed:
                    loadable.append(u)
        
        for u in loadable:
            if len(fast_mem) < fast_memory_budget:
                new_fast = fast_mem | frozenset([u])
                new_state = (pos, new_fast, computed)
                new_cost = cost + 1  # load costs 1
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'load', u)
                    heapq.heappush(pq, (new_cost, pos, new_fast, computed))
            else:
                # Need to evict one vertex first
                # Try evicting each vertex in fast_mem
                for u_evict in fast_mem:
                    # If u_evict is in slow, delete is free; otherwise store costs 1
                    if u_evict in local_inputs:
                        # Inputs are always available, so we can delete for free
                        evict_cost = 0
                    elif u_evict in computed:
                        # We need to store it
                        evict_cost = 1
                    else:
                        # Should not happen
                        evict_cost = 1
                    
                    new_fast = (fast_mem - frozenset([u_evict])) | frozenset([u])
                    new_state = (pos, new_fast, computed)
                    new_cost = cost + 1 + evict_cost  # load + evict
                    if dist.get(new_state, float('inf')) > new_cost:
                        dist[new_state] = new_cost
                        parent[new_state] = (state, f'load_evict_{u_evict}', u)
                        heapq.heappush(pq, (new_cost, pos, new_fast, computed))
        
        # Action 3: Store a vertex in fast_mem that is in computed
        for u in fast_mem:
            if u in computed and u not in local_inputs:
                new_fast = fast_mem - frozenset([u])
                new_state = (pos, new_fast, computed)
                new_cost = cost + 1  # store costs 1
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'store', u)
                    heapq.heappush(pq, (new_cost, pos, new_fast, computed))
        
        # Action 4: Delete a vertex in fast_mem that is in slow (inputs)
        for u in fast_mem:
            if u in local_inputs:
                new_fast = fast_mem - frozenset([u])
                new_state = (pos, new_fast, computed)
                new_cost = cost  # delete is free
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'delete', u)
                    heapq.heappush(pq, (new_cost, pos, new_fast, computed))
    
    if best_final_state is None:
        return None
    
    # Reconstruct path
    ops = []
    state = best_final_state
    while state in parent:
        prev_state, action, vertex_local = parent[state]
        # Map local vertex to original
        if 0 <= vertex_local < n:
            original_vertex = needed_vertices[vertex_local]
        else:
            original_vertex = vertex_local
        
        if action == 'compute':
            ops.append(['compute', original_vertex])
        elif action == 'load':
            ops.append(['load', original_vertex])
        elif action.startswith('load_evict_'):
            evict_local = int(action.split('_')[2])
            evict_vertex = needed_vertices[evict_local]
            if evict_local in local_inputs:
                ops.append(['delete', evict_vertex])
            else:
                ops.append(['store', evict_vertex])
            ops.append(['load', original_vertex])
        elif action == 'store':
            ops.append(['store', original_vertex])
        elif action == 'delete':
            ops.append(['delete', original_vertex])
        
        state = prev_state
    
    ops.reverse()
    
    # Verify the solution
    if _verify_solution(ops, needed_vertices, parent_dict, children, inputs, outputs, 
                        fast_memory_budget, needed_set):
        return ops
    else:
        return None

def _verify_solution(ops, needed_vertices, parent_dict, children, inputs, outputs,
                     fast_memory_budget, needed_set):
    """Verify that a solution is valid."""
    in_fast = set()
    in_slow = set(inputs)
    computed = set(inputs)
    
    for op, v in ops:
        if op == 'load':
            if v not in in_slow:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            in_slow.add(v)
        elif op == 'compute':
            parents = [p for p in parent_dict.get(v, []) if p in needed_set]
            if not all(p in in_fast for p in parents):
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
            computed.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
    
    # Check all outputs are in fast memory
    for o in outputs:
        if o in needed_set and o not in in_fast:
            return False
    
    return True

def _greedy_pebble(needed_vertices, parent_dict, children, inputs, outputs,
                   fast_memory_budget, deadline, vertex_to_idx, needed_set,
                   last_child_idx, first_child_idx):
    """Improved greedy pebbling with lookahead."""
    in_fast = set()
    in_slow = set(inputs)
    computed = set(inputs)
    operations = []

    def load(v):
        operations.append(['load', v])
        in_fast.add(v)
        
    def store(v):
        operations.append(['store', v])
        in_fast.discard(v)
        in_slow.add(v)
        
    def compute(v):
        operations.append(['compute', v])
        computed.add(v)
        in_fast.add(v)
        
    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)

    # Process vertices in topological order
    for v in needed_vertices:
        if time.time() > deadline:
            break

        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        v_idx = vertex_to_idx.get(v, -1)

        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            
            # Need to load parent p
            if len(in_fast) >= fast_memory_budget:
                # Evict: Find best candidate to remove using improved heuristic
                best_evict = None
                best_score = -float('inf')
                best_is_slow = False
                
                for u in list(in_fast):
                    # Score: how good is it to evict u?
                    # Higher score = better to evict
                    score = 0
                    
                    # 1. If u is no longer needed (last use in past), high score
                    u_last = last_child_idx.get(u, -1)
                    if u_last < v_idx:
                        score += 1000
                    
                    # 2. If u is a parent of v, penalize (we need it soon)
                    if u in parents:
                        score -= 500
                    
                    # 3. If u is in slow memory, we can delete for free (good)
                    if u in in_slow:
                        score += 100
                    else:
                        score -= 50  # Must store, costs a transfer
                    
                    # 4. Lookahead: if u is needed for many future vertices, penalize
                    future_uses = 0
                    for child in children.get(u, []):
                        if child in needed_set and vertex_to_idx.get(child, -1) > v_idx:
                            future_uses += 1
                    score -= future_uses * 10
                    
                    if score > best_score:
                        best_score = score
                        best_evict = u
                        best_is_slow = (u in in_slow)

                if best_evict is not None:
                    if best_is_slow:
                        delete(best_evict)
                    else:
                        store(best_evict)

            load(p)

        # Compute v
        if v not in in_fast:
            if v in inputs:
                # Input vertex, just ensure it's in fast
                if v not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        # Evict
                        best_evict = None
                        best_score = -float('inf')
                        best_is_slow = False
                        for u in list(in_fast):
                            u_last = last_child_idx.get(u, -1)
                            score = 0
                            if u_last < v_idx:
                                score += 1000
                            if u in in_slow:
                                score += 100
                            else:
                                score -= 50
                            if score > best_score:
                                best_score = score
                                best_evict = u
                                best_is_slow = (u in in_slow)
                        if best_evict is not None:
                            if best_is_slow:
                                delete(best_evict)
                            else:
                                store(best_evict)
                    load(v)
            else:
                compute(v)
        # If v is in in_fast, it was loaded as an input or computed earlier

        # Proactive cleanup: remove vertices that are no longer needed
        # This helps keep memory clean for future loads
        if len(in_fast) > fast_memory_budget * 0.5:  # Only clean up if memory is somewhat full
            for u in list(in_fast):
                u_last = last_child_idx.get(u, -1)
                if u_last < v_idx and u not in outputs:
                    # u is no longer needed
                    if u in in_slow:
                        delete(u)
                    else:
                        store(u)

    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                # Evict something
                best_evict = None
                best_score = -float('inf')
                best_is_slow = False
                for u in list(in_fast):
                    if u in outputs:
                        continue
                    u_last = last_child_idx.get(u, -1)
                    score = 0
                    if u_last < num_vertices:
                        score += 1000
                    if u in in_slow:
                        score += 100
                    else:
                        score -= 50
                    if score > best_score:
                        best_score = score
                        best_evict = u
                        best_is_slow = (u in in_slow)
                
                if best_evict is not None:
                    if best_is_slow:
                        delete(best_evict)
                    else:
                        store(best_evict)

            load(v)

    return operations
# EVOLVE-BLOCK-END
