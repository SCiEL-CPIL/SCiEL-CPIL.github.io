import sys
import heapq
from functools import lru_cache

# EVOLVE-BLOCK-START

def pebble(dag, fast_memory_budget):
    """
    Solve DAG pebbling with A* search over memory configurations.
    
    dag: dict with keys:
        'n': number of vertices (0-indexed)
        'parents': list of lists, parents[i] = list of parent vertex indices of i
        'outputs': list of vertex indices that must be blue at the end
    
    fast_memory_budget: int, maximum number of red (fast memory) slots
    
    Returns: list of [operation, vertex] where operation is in {'load', 'store', 'compute', 'delete'}
    """
    n = dag['n']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    budget = fast_memory_budget
    
    if n == 0:
        return []
    
    # Topological order: vertices 0..n-1 are already in topological order
    # (parents[i] < i for all i, or at least parents come before children)
    
    # Determine which vertices need to be computed (those on paths to outputs)
    # We'll use the full graph
    
    # State: (loaded_mask, computed_mask) where:
    #   loaded_mask: vertices currently in fast memory (red)
    #   computed_mask: vertices whose values are currently available (blue or in fast mem)
    # Actually, let's think more carefully.
    
    # In pebbling:
    # - Blue pebble = value is on slow disk (permanent, no cost to keep)
    # - Red pebble = value is in fast memory (costs 1 slot)
    # - No pebble = value not available
    #
    # Operations:
    # - load(v): move blue pebble to red (needs v to be blue)
    # - store(v): move red pebble to blue (needs v to be red)
    # - compute(v): compute v from parents (needs all parents to have red pebbles, 
    #              result gets a red pebble, needs free fast memory slot)
    # - delete(v): remove red pebble (frees a slot)
    #
    # Goal: all output vertices have blue pebbles at the end.
    # Start: all input vertices (no parents) have blue pebbles.
    
    # State representation: (blue_mask, red_mask)
    # blue_mask: vertices with blue pebbles
    # red_mask: vertices with red pebbles
    # Constraint: |red_mask| <= budget
    # Initially: blue_mask has all inputs, red_mask = 0
    # Goal: all outputs in blue_mask
    
    # Let's identify input vertices (no parents)
    inputs = [i for i in range(n) if not parents[i]]
    initial_blue = 0
    for v in inputs:
        initial_blue |= (1 << v)
    initial_red = 0
    
    goal_mask = 0
    for v in outputs:
        goal_mask |= (1 << v)
    
    # A* search
    # State: (blue_mask, red_mask)
    # Cost: number of operations so far
    # Heuristic: minimum additional operations needed
    
    # Precompute: for each vertex, what's the minimum "work" to get it blue?
    # Actually, let's use a simpler heuristic: count of outputs not yet blue,
    # times minimum operations per output (at least 1 store or 1 compute+store)
    
    # Better heuristic: for each output not yet blue, we need at least 1 operation.
    # And if it's not computed yet, we need at least 1 compute + 1 store = 2 ops.
    
    # Let's use: heuristic = number of outputs that are not blue
    # (each needs at least a store or already computed and needs store)
    
    # Actually, a vertex can be blue without being in outputs. Let's think...
    # An output vertex needs to be blue. If it's not blue, we need at least 1 op.
    # If it's not even computed (not in blue or red), we need at least 2 ops (compute + store).
    
    # For the heuristic, let's be optimistic: 1 op per non-blue output.
    
    def heuristic(blue_mask, red_mask):
        h = 0
        for v in outputs:
            if not (blue_mask & (1 << v)):
                # Not blue. If not in red either, needs compute + store (2 ops min)
                # If in red, needs store (1 op min)
                if red_mask & (1 << v):
                    h += 1
                else:
                    h += 2
        return h
    
    # Priority queue: (f = g + h, g, blue_mask, red_mask, operations_so_far)
    # We need to track the path, so let's store parent pointers or reconstruct.
    
    # Given the potentially large state space, let's use A* with careful pruning.
    # n can be up to... let's assume reasonable size (say < 30 vertices for the bitmask approach)
    
    if n > 25:
        # Fallback to a simpler greedy approach for large n
        return _greedy_pebble(dag, fast_memory_budget)
    
    # A* implementation
    # State: (blue_mask, red_mask)
    # Use dict for best g-cost seen
    
    start_state = (initial_blue, initial_red)
    
    # Check if already at goal
    if (initial_blue & goal_mask) == goal_mask:
        return []
    
    open_set = []
    h0 = heuristic(initial_blue, initial_red)
    heapq.heappush(open_set, (h0, 0, initial_blue, initial_red))
    
    # g_score: state -> best cost
    g_score = {start_state: 0}
    parent = {}  # state -> (prev_state, operation, vertex)
    
    max_ops = 20000
    visited = set()
    op_count = 0
    
    while open_set:
        f, g, blue, red = heapq.heappop(open_set)
        state = (blue, red)
        
        if state in visited:
            continue
        visited.add(state)
        
        # Check goal
        if (blue & goal_mask) == goal_mask:
            # Reconstruct path
            ops = []
            cur = state
            while cur != start_state:
                prev, op, v = parent[cur]
                ops.append([op, v])
                cur = prev
            ops.reverse()
            if len(ops) <= max_ops:
                return ops
            else:
                break  # Path too long
        
        # Generate successors
        red_count = bin(red).count('1')
        
        # For each vertex v, try possible operations
        
        for v in range(n):
            v_bit = 1 << v
            
            # 1. load(v): blue -> red. Requires v is blue, v is NOT red, red_count < budget
            if (blue & v_bit) and not (red & v_bit) and red_count < budget:
                new_blue = blue & ~v_bit
                new_red = red | v_bit
                new_state = (new_blue, new_red)
                new_g = g + 1
                if new_g < g_score.get(new_state, float('inf')):
                    g_score[new_state] = new_g
                    h = heuristic(new_blue, new_red)
                    parent[new_state] = (state, 'load', v)
                    heapq.heappush(open_set, (new_g + h, new_g, new_blue, new_red))
            
            # 2. store(v): red -> blue. Requires v is red
            if red & v_bit:
                new_blue = blue | v_bit
                new_red = red & ~v_bit
                new_state = (new_blue, new_red)
                new_g = g + 1
                if new_g < g_score.get(new_state, float('inf')):
                    g_score[new_state] = new_g
                    h = heuristic(new_blue, new_red)
                    parent[new_state] = (state, 'store', v)
                    heapq.heappush(open_set, (new_g + h, new_g, new_blue, new_red))
            
            # 3. compute(v): compute v from parents. Requires all parents are red,
            #    v is not already red, red_count < budget (since result is red)
            #    Also v should not already be blue or red (recomputation allowed but let's allow it)
            if not (red & v_bit):
                # Check all parents are red
                all_parents_red = True
                for p in parents[v]:
                    if not (red & (1 << p)):
                        all_parents_red = False
                        break
                if all_parents_red and red_count < budget:
                    new_blue = blue  # blue unchanged
                    new_red = red | v_bit
                    new_state = (new_blue, new_red)
                    new_g = g + 1
                    if new_g < g_score.get(new_state, float('inf')):
                        g_score[new_state] = new_g
                        h = heuristic(new_blue, new_red)
                        parent[new_state] = (state, 'compute', v)
                        heapq.heappush(open_set, (new_g + h, new_g, new_blue, new_red))
            
            # 4. delete(v): remove red pebble. Requires v is red
            if red & v_bit:
                new_blue = blue
                new_red = red & ~v_bit
                new_state = (new_blue, new_red)
                new_g = g + 1
                if new_g < g_score.get(new_state, float('inf')):
                    g_score[new_state] = new_g
                    h = heuristic(new_blue, new_red)
                    parent[new_state] = (state, 'delete', v)
                    heapq.heappush(open_set, (new_g + h, new_g, new_blue, new_red))
        
        # Safety: limit exploration
        op_count += 1
        if op_count > 500000:
            break
    
    # If A* didn't find a solution, fall back to greedy
    return _greedy_pebble(dag, fast_memory_budget)


def _greedy_pebble(dag, fast_memory_budget):
    """Greedy fallback: compute in topological order, store outputs, delete intermediates."""
    n = dag['n']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    budget = fast_memory_budget
    
    # Topological order: assume 0..n-1
    blue = set()
    red = set()
    ops = []
    
    # Initially, inputs are blue
    for v in range(n):
        if not parents[v]:
            blue.add(v)
    
    # Process vertices in topological order
    computed = set()
    
    for v in range(n):
        # Load parents into red if they're blue
        for p in parents[v]:
            if p not in red:
                if p in blue:
                    ops.append(['load', p])
                    blue.remove(p)
                    red.add(p)
        
        # Compute v (all parents should be red now)
        # But we need budget space
        # Delete some red vertices that aren't needed anymore if no space
        while len(red) >= budget and red:
            # Find a red vertex that is not a parent of any uncomputed vertex
            # and not an output
            to_delete = None
            for r in list(red):
                if r not in outputs:
                    # Check if r is a parent of any uncomputed vertex
                    is_parent_of_uncomputed = False
                    for u in range(v, n):
                        if r in parents[u] and u not in computed:
                            is_parent_of_uncomputed = True
                            break
                    if not is_parent_of_uncomputed:
                        to_delete = r
                        break
            if to_delete is None:
                # Just delete any non-output red
                for r in list(red):
                    if r not in outputs:
                        to_delete = r
                        break
            if to_delete is None:
                to_delete = next(iter(red))
            ops.append(['delete', to_delete])
            red.remove(to_delete)
        
        ops.append(['compute', v])
        red.add(v)
        computed.add(v)
        
        # If v is an output, store it
        if v in outputs:
            ops.append(['store', v])
            red.remove(v)
            blue.add(v)
        else:
            # Keep in red or delete? Delete to free space if possible
            # Actually, keep it for now; we'll manage space as needed
            pass
    
    # At the end, ensure all outputs are blue
    for v in outputs:
        if v in red:
            ops.append(['store', v])
            red.remove(v)
            blue.add(v)
        elif v not in blue:
            # Shouldn't happen, but handle
            pass
    
    # Trim to max 20000 ops
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops


# EVOLVE-BLOCK-END
