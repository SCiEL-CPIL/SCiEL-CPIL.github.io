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

    # 3. Recursive divide-and-conquer for layered DAGs
    if n_needed <= 500:
        result = _recursive_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                                   fast_memory_budget, deadline, last_child_idx)
        if result is not None:
            return result

    # 4. Greedy for large
    return _greedy_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                          fast_memory_budget, deadline, last_child_idx)

def _dijkstra_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                     fast_memory_budget, deadline, last_child_idx):
    """
    Dijkstra-based optimal pebbling for small problems.
    State: (frozenset of fast memory, frozenset of computed values)
    We process vertices in topological order implicitly by only allowing compute when parents are available.
    Cost: 1 for load/store, 0 for compute/delete
    """
    n = len(needed_vertices)
    if n == 0:
        return []
    
    init_fast = frozenset()
    init_computed = frozenset(local_inputs)
    init_state = (init_fast, init_computed)
    
    dist = {init_state: 0}
    parent = {}
    pq = [(0, init_fast, init_computed)]
    
    best_final_cost = float('inf')
    best_final_state = None
    
    steps = 0
    max_steps = 100000
    
    while pq:
        if time.time() > deadline:
            break
        if steps >= max_steps:
            break
        steps += 1
        
        cost, fast_mem, computed = heapq.heappop(pq)
        state = (fast_mem, computed)
        
        if dist.get(state, float('inf')) < cost:
            continue
        
        # Check if we're done: all outputs in fast mem
        if all(o in fast_mem for o in local_outputs):
            if cost < best_final_cost:
                best_final_cost = cost
                best_final_state = state
            continue
        
        # Find which vertices can be computed next (all parents in fast mem, not yet computed)
        computable = []
        for v in range(n):
            if v in computed:
                continue
            parents_local = local_parents.get(v, [])
            if all(p in fast_mem for p in parents_local):
                computable.append(v)
        
        # Action 1: Compute any computable vertex
        for v in computable:
            new_computed = computed | frozenset([v])
            new_fast = fast_mem | frozenset([v])
            if len(new_fast) <= fast_memory_budget:
                new_state = (new_fast, new_computed)
                new_cost = cost
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'compute', v)
                    heapq.heappush(pq, (new_cost, new_fast, new_computed))
        
        # Action 2: Load a vertex from slow memory (inputs or computed)
        loadable = [u for u in range(n) if u not in fast_mem and (u in local_inputs or u in computed)]
        for u in loadable:
            if len(fast_mem) < fast_memory_budget:
                new_fast = fast_mem | frozenset([u])
                new_state = (new_fast, computed)
                new_cost = cost + 1
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'load', u)
                    heapq.heappush(pq, (new_cost, new_fast, new_computed))
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
                    new_state = (new_fast, computed)
                    new_cost = cost + 1 + evict_cost
                    if dist.get(new_state, float('inf')) > new_cost:
                        dist[new_state] = new_cost
                        parent[new_state] = (state, f'load_evict_{u_evict}_{evict_op}', u)
                        heapq.heappush(pq, (new_cost, new_fast, new_computed))
        
        # Action 3: Store a computed value (not input)
        for u in fast_mem:
            if u in computed and u not in local_inputs:
                new_fast = fast_mem - frozenset([u])
                new_state = (new_fast, computed)
                new_cost = cost + 1
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'store', u)
                    heapq.heappush(pq, (new_cost, new_fast, new_computed))
        
        # Action 4: Delete (free) - inputs always, computed if no future uses
        for u in fast_mem:
            if u in local_inputs:
                new_fast = fast_mem - frozenset([u])
                new_state = (new_fast, computed)
                new_cost = cost
                if dist.get(new_state, float('inf')) > new_cost:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'delete', u)
                    heapq.heappush(pq, (new_cost, new_fast, new_computed))
            elif u in computed:
                # Check if u has any future children not yet computed
                has_future_child = False
                for c in local_children.get(u, []):
                    if c not in computed:
                        has_future_child = True
                        break
                if not has_future_child:
                    new_fast = fast_mem - frozenset([u])
                    new_state = (new_fast, computed)
                    new_cost = cost
                    if dist.get(new_state, float('inf')) > new_cost:
                        dist[new_state] = new_cost
                        parent[new_state] = (state, 'delete', u)
                        heapq.heappush(pq, (new_cost, new_fast, new_computed))

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
    State: (fast_mem, computed) - no position tracking, just explore valid transitions.
    """
    n = len(needed_vertices)
    if n == 0:
        return []
    
    init_fast = frozenset()
    init_computed = frozenset(local_inputs)
    
    # State: (cost, fast_mem, computed, path)
    current_beam = [(0, init_fast, init_computed, [])]
    
    best_final_cost = float('inf')
    best_final_path = None
    
    max_steps = 5000
    steps = 0
    
    while current_beam:
        if time.time() > deadline:
            break
        if steps >= max_steps:
            break
        steps += 1
        
        next_beam = []
        
        for cost, fast_mem, computed, path in current_beam:
            # Check if done
            if all(o in fast_mem for o in local_outputs):
                if cost < best_final_cost:
                    best_final_cost = cost
                    best_final_path = path
                continue
            
            new_states = []
            
            # Find computable vertices
            computable = []
            for v in range(n):
                if v in computed:
                    continue
                parents_local = local_parents.get(v, [])
                if all(p in fast_mem for p in parents_local):
                    computable.append(v)
            
            # 1. Compute
            for v in computable:
                new_computed = computed | frozenset([v])
                new_fast = fast_mem | frozenset([v])
                if len(new_fast) <= fast_memory_budget:
                    new_path = path + [('compute', needed_vertices[v])]
                    new_states.append((cost, new_fast, new_computed, new_path))
            
            # 2. Load
            loadable = [u for u in range(n) if u not in fast_mem and (u in local_inputs or u in computed)]
            for u in loadable:
                if len(fast_mem) < fast_memory_budget:
                    new_fast = fast_mem | frozenset([u])
                    new_path = path + [('load', needed_vertices[u])]
                    new_states.append((cost + 1, new_fast, computed, new_path))
                else:
                    for u_evict in fast_mem:
                        if u_evict in local_inputs:
                            evict_cost = 0
                            evict_op = 'delete'
                        else:
                            evict_cost = 1
                            evict_op = 'store'
                        
                        new_fast = (fast_mem - frozenset([u_evict])) | frozenset([u])
                        new_path = path + [(evict_op, needed_vertices[u_evict]), ('load', needed_vertices[u])]
                        new_states.append((cost + 1 + evict_cost, new_fast, computed, new_path))
            
            # 3. Store
            for u in fast_mem:
                if u in computed and u not in local_inputs:
                    new_fast = fast_mem - frozenset([u])
                    new_path = path + [('store', needed_vertices[u])]
                    new_states.append((cost + 1, new_fast, computed, new_path))
            
            # 4. Delete
            for u in fast_mem:
                if u in local_inputs:
                    new_fast = fast_mem - frozenset([u])
                    new_path = path + [('delete', needed_vertices[u])]
                    new_states.append((cost, new_fast, computed, new_path))
                elif u in computed:
                    # Check if no future children need it
                    has_future_child = False
                    for c in local_children.get(u, []):
                        if c not in computed:
                            has_future_child = True
                            break
                    if not has_future_child:
                        new_fast = fast_mem - frozenset([u])
                        new_path = path + [('delete', needed_vertices[u])]
                        new_states.append((cost, new_fast, computed, new_path))
            
            next_beam.extend(new_states)
        
        if not next_beam:
            break
        
        # Deduplicate by (fast_mem, computed) keeping min cost
        best_by_state = {}
        for cost, fast_mem, computed, path in next_beam:
            key = (fast_mem, computed)
            if key not in best_by_state or cost < best_by_state[key][0]:
                best_by_state[key] = (cost, fast_mem, computed, path)
        
        next_beam = list(best_by_state.values())
        next_beam.sort(key=lambda x: x[0])
        
        current_beam = next_beam[:beam_width]
        
        # Prune: if min cost in beam >= best_final_cost, we can stop
        if current_beam and current_beam[0][0] >= best_final_cost:
            break
        
    if best_final_path is None:
        return None
        
    ops = [[op, v] for op, v in best_final_path]
    
    if _verify_solution(ops, needed_vertices, local_parents, local_children, local_inputs, local_outputs, 
                        fast_memory_budget, needed_set=needed_set):
        return ops
    else:
        return None

def _find_cut_vertices(local_parents, local_children, local_inputs, local_outputs, n):
    """
    Find good cut vertices that split the DAG into sub-problems.
    A cut vertex is one whose removal (or separation) divides the graph into
    components with limited interface.
    Returns a list of candidate cut vertices sorted by quality.
    """
    # Compute reachability
    def get_ancestors(v, parent_map):
        """Get all ancestors of v (including v itself)."""
        visited = set()
        stack = [v]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for p in parent_map.get(u, []):
                if p not in visited:
                    stack.append(p)
        return visited
    
    def get_descendants(v, child_map):
        """Get all descendants of v (including v itself)."""
        visited = set()
        stack = [v]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for c in child_map.get(u, []):
                if c not in visited:
                    stack.append(c)
        return visited
    
    # Compute for all vertices (limit to avoid O(n^2) for large n)
    if n <= 200:
        candidates = []
        for v in range(n):
            ancestors = get_ancestors(v, local_parents)
            descendants = get_descendants(v, local_children)
            
            # Interface: vertices that are both ancestors and descendants
            # (excluding v itself)
            interface = (ancestors & descendants) - {v}
            interface_size = len(interface)
            
            # Score: prefer small interface, and prefer vertices that are
            # "central" (have many ancestors and descendants)
            score = interface_size
            if score < n * 0.3:  # Only consider if interface is less than 30% of graph
                candidates.append((score, v, ancestors, descendants, interface))
        
        candidates.sort(key=lambda x: x[0])
        return candidates
    
    return []

def _recursive_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                      fast_memory_budget, deadline, last_child_idx):
    """
    Recursive divide-and-conquer pebbling.
    Find a cut vertex that splits the DAG, recursively pebble sub-DAGs,
    and synthesize the schedule.
    """
    n = len(needed_vertices)
    if n == 0:
        return []
    
    # Try to find a good cut vertex
    candidates = _find_cut_vertices(local_parents, local_children, local_inputs, local_outputs, n)
    
    if not candidates:
        # No good cut vertex found, fall back to greedy
        return _greedy_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                              fast_memory_budget, deadline, last_child_idx)
    
    # Try the best few cut vertices
    for score, cut_v, ancestors, descendants, interface in candidates[:3]:
        if time.time() > deadline:
            break
        
        # The cut vertex divides the graph into:
        # - Part 1: ancestors of cut_v (including cut_v) - these are "upstream"
        # - Part 2: descendants of cut_v (including cut_v) - these are "downstream"
        # - Interface: vertices that are both - these need to be in fast memory when switching
        
        upstream = ancestors - {cut_v}
        downstream = descendants - {cut_v}
        interface_set = interface  # already computed
        
        # Check if this is a good split
        if len(upstream) < 3 or len(downstream) < 3:
            continue
        
        # Use a modified greedy that prioritizes keeping interface vertices in fast memory
        result = _greedy_with_cut(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                                  fast_memory_budget, deadline, last_child_idx, cut_v, ancestors, descendants)
        if result is not None:
            return result
    
    # If no cut vertex worked well, fall back to greedy
    return _greedy_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                          fast_memory_budget, deadline, last_child_idx)

def _greedy_with_cut(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                     fast_memory_budget, deadline, last_child_idx, cut_v, before_cut, after_cut):
    """
    Greedy pebbling with awareness of a cut vertex.
    Prioritize keeping vertices that are needed on both sides of the cut in fast memory.
    """
    n = len(needed_vertices)
    in_fast = set()
    in_slow = set(local_inputs)
    computed = set(local_inputs)
    operations = []

    # Vertices that are "interface" - needed on both sides
    interface = before_cut & after_cut - {cut_v}
    
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

    def evict_one(protected=None):
        """Evict the worst vertex from fast memory."""
        if protected is None:
            protected = set()
        
        best_evict = None
        best_score = -float('inf')
        
        for u in list(in_fast):
            if u in protected:
                continue
            
            score = 0.0
            u_last = last_child_idx.get(u, -1)
            
            # Count future uses (children not yet computed)
            future_uses = 0
            for c in local_children.get(u, []):
                if c not in computed:
                    future_uses += 1
            
            # If no future uses, very good to evict
            if future_uses == 0:
                score += 1000
            else:
                score -= future_uses * 50
            
            # If in slow memory, delete is free (good)
            if u in in_slow:
                score += 100
            else:
                score -= 100
            
            # Prefer evicting inputs over computed
            if u in local_inputs:
                score += 50
            
            # CRITICAL: Never evict interface vertices or the cut vertex
            if u in interface or u == cut_v:
                score -= 10000
            
            if score > best_score:
                best_score = score
                best_evict = u

        if best_evict is not None:
            if best_evict in in_slow:
                delete(best_evict)
            else:
                store(best_evict)
            return best_evict
        return None

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
                # Evict one that's not a parent we need and not interface
                protected = set(parents) | interface | {cut_v}
                evict_one(protected)

            load(p)

        # Compute v
        if v not in in_fast:
            if v in local_inputs:
                if len(in_fast) >= fast_memory_budget:
                    protected = interface | {cut_v}
                    evict_one(protected)
                load(v)
            else:
                compute(v)

        # Proactive cleanup: evict vertices with no future uses (except interface)
        for u in list(in_fast):
            if u in local_outputs:
                continue
            if u in interface or u == cut_v:
                continue
            future_uses = 0
            for c in local_children.get(u, []):
                if c not in computed:
                    future_uses += 1
            if future_uses == 0:
                if u in in_slow:
                    delete(u)
                else:
                    store(u)

    # Ensure outputs in fast
    for v in local_outputs:
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                protected = set(local_outputs) | interface | {cut_v}
                evict_one(protected)
            load(v)

    return operations

def _verify_solution(ops, needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                     fast_memory_budget, needed_set):
    """Verify that a solution is valid."""
    in_fast = set()
    in_slow = set(local_inputs)
    computed = set(local_inputs)
    
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

def _greedy_pebble(needed_vertices, local_parents, local_children, local_inputs, local_outputs,
                   fast_memory_budget, deadline, last_child_idx):
    """Improved greedy pebbling with better eviction heuristic."""
    n = len(needed_vertices)
    in_fast = set()
    in_slow = set(local_inputs)
    computed = set(local_inputs)
    operations = []

    # Precompute: for each vertex, the number of children that are not yet computed
    # We'll track this dynamically

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

    def evict_one(protected=None):
        """Evict the worst vertex from fast memory."""
        if protected is None:
            protected = set()
        
        best_evict = None
        best_score = -float('inf')
        
        for u in list(in_fast):
            if u in protected:
                continue
            
            score = 0.0
            u_last = last_child_idx.get(u, -1)
            
            # Count future uses (children not yet computed)
            future_uses = 0
            for c in local_children.get(u, []):
                if c not in computed:
                    future_uses += 1
            
            # If no future uses, very good to evict
            if future_uses == 0:
                score += 1000
            else:
                # Cost of recomputing: roughly the number of future uses times some factor
                score -= future_uses * 50
            
            # If in slow memory, delete is free (good)
            if u in in_slow:
                score += 100
            else:
                # Store costs 1
                score -= 100
            
            # Prefer evicting inputs over computed (inputs are cheaper to reload)
            if u in local_inputs:
                score += 50
            
            if score > best_score:
                best_score = score
                best_evict = u

        if best_evict is not None:
            if best_evict in in_slow:
                delete(best_evict)
            else:
                store(best_evict)
            return best_evict
        return None

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
                # Evict one that's not a parent we need
                protected = set(parents)
                evict_one(protected)

            load(p)

        # Compute v
        if v not in in_fast:
            if v in local_inputs:
                if len(in_fast) >= fast_memory_budget:
                    protected = set()
                    evict_one(protected)
                load(v)
            else:
                compute(v)

        # Proactive cleanup: evict vertices with no future uses
        for u in list(in_fast):
            if u in local_outputs:
                continue
            future_uses = 0
            for c in local_children.get(u, []):
                if c not in computed:
                    future_uses += 1
            if future_uses == 0:
                if u in in_slow:
                    delete(u)
                else:
                    store(u)

    # Ensure outputs in fast
    for v in local_outputs:
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                protected = set(local_outputs)
                evict_one(protected)
            load(v)

    return operations
