"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq
from collections import defaultdict, deque

def _parse_dag(dag):
    """Parse various possible DAG formats into a unified structure."""
    if isinstance(dag, dict):
        parents = dag.get('parents', None)
        outputs = dag.get('outputs', None)
        inputs = dag.get('inputs', None)
        vertices = dag.get('vertices', None)
        n = dag.get('n', None)
        
        if parents is None:
            if isinstance(dag, list):
                parents = [list(p) for p in dag]
                n = len(dag)
                if outputs is None:
                    outputs = [n - 1]
                if inputs is None:
                    inputs = [i for i in range(n) if not parents[i]]
            else:
                return None, None, None, None, None
        else:
            if isinstance(parents, list):
                parents = [list(p) for p in parents]
                n = len(parents)
            elif isinstance(parents, dict):
                n = max(parents.keys()) + 1 if parents else 0
                parents_list = [list(parents.get(i, [])) for i in range(n)]
                parents = parents_list
            
            if outputs is None:
                children = [[] for _ in range(n)]
                for v in range(n):
                    for p in parents[v]:
                        children[p].append(v)
                outputs = [v for v in range(n) if not children[v]]
            
            if inputs is None:
                inputs = [v for v in range(n) if not parents[v]]
        
        return parents, outputs, inputs, vertices, n
    
    elif isinstance(dag, list):
        parents = [list(p) if isinstance(p, (list, tuple)) else [] for p in dag]
        n = len(dag)
        outputs = [n - 1] if n > 0 else []
        inputs = [i for i in range(n) if not parents[i]]
        return parents, outputs, inputs, None, n
    
    return None, None, None, None, None


def _build_children_and_last_use(parents, n):
    """Build children list and last-use index for each vertex."""
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # last_use[v] = maximum index of a child of v, or -1 if no children
    last_use = [-1] * n
    for v in range(n):
        if children[v]:
            last_use[v] = max(children[v])
    
    return children, last_use


def _astar_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    A* search over memory states for communication-minimizing pebbling.
    State: (frozenset(red_pebbled), next_vertex_to_process)
    Goal: all outputs stored, all non-output vertices unpebbled.
    """
    n = len(parents)
    if n == 0:
        return []
    
    children, last_use = _build_children_and_last_use(parents, n)
    is_output = set(outputs)
    is_input = [len(parents[v]) == 0 for v in range(n)]
    
    # Precompute: for each vertex, whether it's an input
    # Precompute: outputs that are also inputs (need to be loaded and stored)
    input_outputs = set(v for v in outputs if is_input[v])
    computed_outputs = set(v for v in outputs if not is_input[v])
    
    # All outputs that must be stored
    all_outputs = set(outputs)
    
    # Heuristic: minimum transfers remaining
    # For each output not yet stored, at least 1 store needed
    # For each input that must be loaded (and isn't currently red), at least 1 load needed
    # But this is a lower bound; we use it as h
    
    def heuristic(red_set, stored_outputs):
        """Lower bound on remaining transfers."""
        h = 0
        # Each output not yet stored needs at least 1 store
        for o in all_outputs:
            if o not in stored_outputs:
                h += 1
        # Each input-output that's not yet loaded needs at least 1 load
        # (but if it's already in red_set, no load needed)
        for io in input_outputs:
            if io not in stored_outputs and io not in red_set:
                h += 1
        return h
    
    # State: (red_set_frozen, stored_outputs_frozen)
    # We track progress by which outputs have been stored
    # A* priority: g (transfers so far) + h (estimated remaining)
    
    # Initial state: no red pebbles, no stored outputs
    initial_red = frozenset()
    initial_stored = frozenset()
    
    # We need to track: for each vertex, whether it's been "processed" (all its children processed)
    # Actually, let's use a different approach: track the set of vertices that have been "completed"
    # A vertex is completed when it's no longer needed (last_use passed) OR it's stored
    
    # Let's use: (red_set, stored_set, completed_set) where completed means the vertex's value
    # is no longer needed for any future computation
    # Actually, this is getting complex. Let me use a simpler state representation.
    
    # State: (red_set, stored_set)
    # Transitions:
    # 1. load v: if v is input, v not in red, |red| < budget -> add v to red, cost 1
    # 2. compute v: if all parents in red, v not in red, |red| < budget -> add v to red, cost 0
    # 3. store v: if v in red and v in outputs -> remove v from red, add to stored, cost 1
    # 4. delete v: if v in red and v not in outputs (or v's last_use < current position) -> remove from red, cost 0
    
    # The issue: we need to know when a vertex's last_use has passed to allow deletion.
    # Let's track the "current position" as the max index of any vertex in red or stored.
    
    # Actually, let me use a cleaner formulation:
    # State: (red_set, stored_set)
    # A vertex v can be deleted if:
    #   - v is in red
    #   - v is not an output (or if output, already stored)
    #   - No unprocessed child of v exists (i.e., all children of v have index > max(red|stored) or are already done)
    
    # This is still complex. Let me use a different approach:
    # Process vertices in topological order. At each step, we can:
    #   - Load an input
    #   - Compute a vertex whose parents are all in red
    #   - Store an output that's in red
    #   - Delete a non-output vertex that's no longer needed
    
    # For A*, state = (red_set, stored_set, next_compute_index)
    # where next_compute_index tracks which vertices we've "passed" in topological order
    
    # Actually, the cleanest approach for these layered DAGs:
    # Since vertices are topologically ordered, a vertex v's parents all have index < v.
    # So we can process in order 0, 1, ..., n-1.
    # At step v, we need to ensure all parents of v are in red, then compute v.
    # After computing v, we can delete parents that are no longer needed.
    
    # But A* needs to explore different choices of what to keep in red.
    # State: (red_set, stored_set) where we've processed up to some point.
    # Let's define: processed = max(red_set | stored_set) if non-empty, else -1
    # But this doesn't fully capture state.
    
    # Let me use: state = (red_set, stored_set)
    # And define the "frontier" as the set of vertices that can be acted upon next.
    # A vertex v can be:
    #   - loaded if is_input[v] and v not in red and v not in stored
    #   - computed if all parents in red and v not in red and v not in stored
    #   - stored if v in red and v in all_outputs and v not in stored
    #   - deleted if v in red and v not in all_outputs (or v in stored) and no future child needs it
    
    # For "no future child needs it": a child c of v needs v if c is not yet computed/stored
    # and c's index > all indices in red|stored... this is getting circular.
    
    # Simplification for layered DAGs: if all children of v have index > max_index_in_state,
    # then v can be deleted.
    
    def max_index(state_set):
        if not state_set:
            return -1
        return max(state_set)
    
    def can_delete(v, red_set, stored_set):
        """Check if vertex v can be deleted from red."""
        if v not in red_set:
            return False
        if v in all_outputs and v not in stored_set:
            return False  # Output not yet stored
        # Check if any child of v is still "active" (not yet processed)
        # A child c is active if c is not in stored and c's index > max(red_set | stored_set) 
        # OR c is in red_set (meaning it's been computed but not yet cleaned up)
        # Actually, simpler: if all children of v have index <= max(red|stored), v can be deleted
        # because all children have already been processed
        mi = max(max_index(red_set), max_index(stored_set))
        for c in children[v]:
            if c > mi and c not in stored_set:
                return False
        return True
    
    # A* search
    # State: (red_set, stored_set)
    # G-cost: number of transfers (loads + stores) so far
    # H-cost: lower bound on remaining transfers
    
    start = (frozenset(), frozenset())
    
    # For efficiency, we'll limit the search
    max_states = 50000
    states_explored = 0
    visited = {}  # state -> best g-cost seen
    
    # Priority queue: (g + h, g, red_set, stored_set, ops_list)
    # We store ops to reconstruct the path
    # But storing full ops lists is expensive. Let's store parent pointers instead.
    
    # Actually, for memory efficiency, let's use a different approach:
    # Store: (priority, g, red_set, stored_set, parent_index, action)
    # And keep a list of states to reconstruct paths.
    
    # Let me use a simpler BFS-like approach with pruning since the state space
    # for these DAGs might be manageable.
    
    # Alternative: use the greedy last-use strategy but with better eviction heuristics.
    # Given the complexity of full A*, let me implement a hybrid:
    # 1. Try A* for small DAGs (n <= 15 or so)
    # 2. Fall back to greedy for larger ones
    
    if n > 20:
        return None  # Too large for A*, fall back to greedy
    
    # A* implementation
    # State: (red_set, stored_set)
    # We'll use dictionaries for visited states
    
    pq = []
    start_state = (frozenset(), frozenset())
    h0 = heuristic(start_state[0], start_state[1])
    heapq.heappush(pq, (h0, 0, 0, start_state[0], start_state[1]))
    # (f, g, counter, red_set, stored_set)
    
    best_solution = None
    best_cost = float('inf')
    
    # Parent tracking for path reconstruction
    # state -> (parent_state, action)
    parent_map = {start_state: None}
    
    t_start = time.time()
    
    while pq:
        if time.time() > deadline:
            break
        if states_explored > max_states:
            break
        
        f, g, counter, red_set, stored_set = heapq.heappop(pq)
        state = (red_set, stored_set)
        
        states_explored += 1
        
        # Check if goal
        if stored_set == all_outputs:
            # All outputs stored
            best_solution = state
            best_cost = g
            break
        
        # If we've seen this state with lower g, skip
        if state in visited and visited[state] <= g:
            continue
        visited[state] = g
        
        # Generate successors
        mi = max(max_index(red_set), max_index(stored_set))
        
        # Possible actions:
        # 1. Load an input
        for v in range(n):
            if is_input[v] and v not in red_set and v not in stored_set:
                if len(red_set) < fast_memory_budget:
                    new_red = red_set | {v}
                    new_state = (new_red, stored_set)
                    new_g = g + 1
                    new_h = heuristic(new_red, stored_set)
                    heapq.heappush(pq, (new_g + new_h, new_g, states_explored, new_red, stored_set))
                    parent_map[new_state] = (state, ('load', v))
        
        # 2. Compute a vertex
        for v in range(n):
            if not is_input[v] and v not in red_set and v not in stored_set:
                # Check all parents in red
                if all(p in red_set for p in parents[v]):
                    if len(red_set) < fast_memory_budget:
                        new_red = red_set | {v}
                        new_state = (new_red, stored_set)
                        new_g = g  # compute is free
                        new_h = heuristic(new_red, stored_set)
                        heapq.heappush(pq, (new_g + new_h, new_g, states_explored, new_red, stored_set))
                        parent_map[new_state] = (state, ('compute', v))
        
        # 3. Store an output
        for v in all_outputs:
            if v in red_set and v not in stored_set:
                new_red = red_set - {v}
                new_stored = stored_set | {v}
                new_state = (new_red, new_stored)
                new_g = g + 1
                new_h = heuristic(new_red, new_stored)
                heapq.heappush(pq, (new_g + new_h, new_g, states_explored, new_red, new_stored))
                parent_map[new_state] = (state, ('store', v))
        
        # 4. Delete a non-essential vertex
        for v in red_set:
            if can_delete(v, red_set, stored_set):
                new_red = red_set - {v}
                new_state = (new_red, stored_set)
                new_g = g  # delete is free
                new_h = heuristic(new_red, stored_set)
                heapq.heappush(pq, (new_g + new_h, new_g, states_explored, new_red, stored_set))
                parent_map[new_state] = (state, ('delete', v))
    
    if best_solution is None:
        return None
    
    # Reconstruct path
    ops = []
    state = best_solution
    while parent_map[state] is not None:
        parent_state, action = parent_map[state]
        ops.append(list(action))
        state = parent_state
    ops.reverse()
    
    return ops


def _optimized_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Optimized pebbling using last-use analysis for efficient memory management.
    Processes vertices in topological order, evicting values that are no longer needed.
    """
    n = len(parents)
    if n == 0:
        return []
    
    children, last_use = _build_children_and_last_use(parents, n)
    is_output = set(outputs)
    
    # Precompute which vertices are inputs (no parents)
    is_input = [len(parents[v]) == 0 for v in range(n)]
    
    ops = []
    in_fast = set()
    # Track for each vertex in fast memory: its last use index
    # We'll use a list to track last_use for quick eviction
    
    def get_last_use(v):
        return last_use[v]
    
    for v in range(n):
        if time.time() > deadline:
            return None  # Time out
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                # Need to load p; evict if necessary
                while len(in_fast) >= fast_memory_budget:
                    # Find vertex with smallest last_use to evict
                    min_lu = float('inf')
                    evict_v = None
                    for u in in_fast:
                        lu = get_last_use(u)
                        if lu < min_lu:
                            min_lu = lu
                            evict_v = u
                    if evict_v is None or min_lu >= v:
                        # Can't evict safely (all needed at or after v)
                        # Try evicting outputs or any vertex
                        for u in in_fast:
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                break
                        else:
                            # All are outputs, evict the one with smallest last_use
                            for u in in_fast:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                break
                        if len(in_fast) >= fast_memory_budget:
                            return None  # Stuck
                    else:
                        ops.append(['delete', evict_v])
                        in_fast.discard(evict_v)
                
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', p])
                    in_fast.add(p)
        
        # Compute v if it has parents
        if parents[v]:
            if v in in_fast:
                # Already computed, just need to ensure it's available
                pass
            else:
                # Make room if needed
                while len(in_fast) >= fast_memory_budget:
                    min_lu = float('inf')
                    evict_v = None
                    for u in in_fast:
                        lu = get_last_use(u)
                        if lu < min_lu:
                            min_lu = lu
                            evict_v = u
                    if evict_v is None:
                        break
                    ops.append(['delete', evict_v])
                    in_fast.discard(evict_v)
                
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
                else:
                    return None  # Can't compute
        
        # If v is an input that's also an output, handle it
        if not parents[v] and v in is_output:
            if v not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    min_lu = float('inf')
                    evict_v = None
                    for u in in_fast:
                        lu = get_last_use(u)
                        if lu < min_lu:
                            min_lu = lu
                            evict_v = u
                    if evict_v is None:
                        break
                    ops.append(['delete', evict_v])
                    in_fast.discard(evict_v)
                
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', v])
                    in_fast.add(v)
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
        
        # If v is an output, store it
        if v in is_output and parents[v]:
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
        
        # Evict parents that are no longer needed after v
        for p in parents[v]:
            if p in in_fast and p not in is_output:
                if last_use[p] == v:
                    # p is no longer needed
                    ops.append(['delete', p])
                    in_fast.discard(p)
    
    return ops


def _validate_ops(ops, parents, outputs, fast_memory_budget):
    """Validate that operations are legal and complete."""
    n = len(parents)
    in_fast = set()
    stored = set()
    
    for op, v in ops:
        if op == 'load':
            if v in in_fast:
                continue
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            stored.add(v)
        elif op == 'compute':
            for p in parents[v]:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
        else:
            return False
    
    for o in outputs:
        if o not in stored:
            return False
    
    return True


def _count_transfers(ops):
    """Count the number of transfers (load + store operations)."""
    return sum(1 for op, v in ops if op in ('load', 'store'))


def _simple_pebbling(parents, outputs, fast_memory_budget, deadline):
    """Simple fallback pebbling strategy."""
    n = len(parents)
    if n == 0:
        return []
    
    children, last_use = _build_children_and_last_use(parents, n)
    is_output = set(outputs)
    
    ops = []
    in_fast = set()
    
    for v in range(n):
        if time.time() > deadline:
            return None
        
        # Load parents if needed
        for p in parents[v]:
            if p not in in_fast:
                # Evict if needed
                while len(in_fast) >= fast_memory_budget:
                    # Evict vertex with smallest last_use
                    best = None
                    best_lu = float('inf')
                    for u in in_fast:
                        if u not in is_output:
                            lu = last_use[u]
                            if lu < best_lu:
                                best_lu = lu
                                best = u
                    if best is None:
                        for u in in_fast:
                            lu = last_use[u]
                            if lu < best_lu:
                                best_lu = lu
                                best = u
                    if best is not None:
                        ops.append(['delete', best])
                        in_fast.discard(best)
                    else:
                        break
                
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', p])
                    in_fast.add(p)
        
        # Compute if needed
        if parents[v] and v not in in_fast:
            while len(in_fast) >= fast_memory_budget:
                best = None
                best_lu = float('inf')
                for u in in_fast:
                    lu = last_use[u]
                    if lu < best_lu:
                        best_lu = lu
                        best = u
                if best is not None:
                    ops.append(['delete', best])
                    in_fast.discard(best)
                else:
                    break
            
            if len(in_fast) < fast_memory_budget:
                ops.append(['compute', v])
                in_fast.add(v)
        
        # Handle input-output vertices
        if not parents[v] and v in is_output:
            if v not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    best = None
                    best_lu = float('inf')
                    for u in in_fast:
                        lu = last_use[u]
                        if lu < best_lu:
                            best_lu = lu
                            best = u
                    if best is not None:
                        ops.append(['delete', best])
                        in_fast.discard(best)
                    else:
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', v])
                    in_fast.add(v)
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
        
        # Store outputs
        if v in is_output and parents[v] and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
        
        # Clean up
        for p in parents[v]:
            if p in in_fast and p not in is_output and last_use[p] == v:
                ops.append(['delete', p])
                in_fast.discard(p)
    
    return ops


def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 80.0  # Safety margin before 90s kill
    
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        result = pebble_baseline(dag, fast_memory_budget)
        if result is not None:
            return result
    except Exception:
        pass
    
    parents, outputs, inputs, vertices, n = _parse_dag(dag)
    
    if parents is None or n is None:
        return []
    
    if outputs is None:
        outputs = [n - 1]
    
    if fast_memory_budget is None:
        fast_memory_budget = n
    if fast_memory_budget <= 0:
        fast_memory_budget = 1
    
    # Try A* pebbling first (for small DAGs)
    try:
        ops = _astar_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    # Try optimized pebbling
    try:
        ops = _optimized_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    # Try simple pebbling
    try:
        ops = _simple_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    # Last resort: very simple approach
    try:
        ops = []
        in_fast = set()
        children, last_use = _build_children_and_last_use(parents, n)
        is_output = set(outputs)
        
        for v in range(n):
            if time.time() > deadline:
                break
            
            for p in parents[v]:
                if p not in in_fast:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', p])
                        in_fast.add(p)
            
            if parents[v]:
                while len(in_fast) >= fast_memory_budget:
                    evicted = False
                    for u in list(in_fast):
                        if u not in is_output:
                            ops.append(['delete', u])
                            in_fast.discard(u)
                            evicted = True
                            break
                    if not evicted:
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
            
            if v in is_output:
                if v not in in_fast and not parents[v]:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', v])
                        in_fast.add(v)
                
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
            
            for p in parents[v]:
                if p in in_fast and p not in is_output and last_use[p] == v:
                    ops.append(['delete', p])
                    in_fast.discard(p)
        
        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    return []

# EVOLVE-BLOCK-END
