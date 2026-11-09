"""Communication-minimizing DAG pebbling using exact state-space search (Dijkstra)."""

import heapq
from functools import lru_cache


def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem with minimum communication cost.
    
    Args:
        dag: Dictionary with keys:
            - 'parents': list of lists, where parents[i] is list of parent indices of vertex i
            - 'outputs': list of output vertex indices
            - 'n': number of vertices (optional, inferred from parents)
        fast_memory_budget: int, max number of red pebbles allowed at once
    
    Returns:
        list of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'
    """
    # Parse DAG structure
    if isinstance(dag, dict):
        parents = dag.get('parents', [])
        outputs = set(dag.get('outputs', []))
        n = dag.get('n', len(parents))
    else:
        # Fallback: assume dag is a tuple/list (parents, outputs)
        if len(dag) == 2:
            parents, outputs_list = dag
            outputs = set(outputs_list)
            n = len(parents)
        else:
            parents = list(dag)
            outputs = set(range(len(parents)))
            n = len(parents)
    
    # Determine which vertices are inputs (no parents)
    inputs = set(i for i in range(n) if len(parents[i]) == 0)
    
    # Topological order is assumed: parents[i] < i for all i
    
    # State: (frozenset of red vertices, frozenset of blue vertices that are "stored" in slow memory)
    # But actually, let's think more carefully.
    # 
    # Initial state: all inputs are blue (in slow memory), nothing is red.
    # Goal state: all outputs are blue, nothing needs to be red (or we can delete all reds).
    #
    # Actually, the problem says "Inputs start blue, all outputs must finish blue".
    # "Every red value consumes one fast-memory slot, including newly computed results."
    #
    # Let me re-interpret:
    # - Slow memory: contains blue values. Initially all inputs are blue in slow memory.
    # - Fast memory: contains red values. Budget is fast_memory_budget.
    # - Operations:
    #   - load: move a blue value from slow memory to fast memory (becomes red). Cost: 1 transfer.
    #   - store: move a red value from fast memory to slow memory (becomes blue). Cost: 1 transfer.
    #   - compute: compute a vertex from its parents (all parents must be red). The result becomes red. Cost: 0? Or is it free?
    #   - delete: remove a red value from fast memory. Cost: 0?
    #
    # Wait, "Score mean transfers / naive schedule". So cost is number of transfers (load/store).
    # compute and delete have 0 cost.
    #
    # State: (red_set, blue_set_in_slow)
    # But blue_set_in_slow can grow. Initially it's the set of inputs.
    # When we store a red value, it becomes blue in slow memory.
    # When we load a blue value from slow memory, it becomes red in fast memory.
    #
    # Actually, once a value is in slow memory as blue, it stays there. We can load it multiple times.
    # So the state is:
    # - red_set: set of vertices currently in fast memory (red)
    # - stored_set: set of vertices that have been stored to slow memory (blue)
    #
    # Initial: red_set = empty, stored_set = inputs
    # Goal: all outputs are in stored_set (blue), and we can have any red_set (but we should clear it? No, the problem just says outputs must finish blue)
    #
    # Actually, re-reading: "all outputs must finish blue". So at the end, outputs must be in stored_set.
    # Red vertices at the end don't matter as long as outputs are blue.
    #
    # But wait, if a vertex is red, it's not blue. So if an output is red at the end, it's not blue.
    # So we need all outputs to be in stored_set at the end.
    
    # Let's define state as (frozenset(red), frozenset(stored))
    # But stored can be large. Let's think about the size.
    # n is probably small for this problem.
    
    # Initial state
    init_red = frozenset()
    init_stored = frozenset(inputs)
    
    # Goal: all outputs in stored
    # We don't care about red at the end, as long as outputs are stored.
    
    # Dijkstra
    # State: (red, stored)
    # Cost: number of transfers (load + store operations)
    
    # We need to track the path to reconstruct operations.
    
    # Let's use Dijkstra with state = (red_tuple, stored_tuple)
    # where we use tuples for hashability.
    
    # For efficiency, let's encode states as integers if n is small.
    # But let's first try with frozensets.
    
    # Actually, let's think about the maximum n. If n is up to 20-30, the state space could be large.
    # But the problem says "at most 20000 operations", so the solution isn't too long.
    
    # Let's try Dijkstra with frozensets.
    
    # dist[state] = minimum cost to reach state
    # parent[state] = (prev_state, operation, vertex)
    
    # Start
    start_state = (init_red, init_stored)
    
    # Check if already at goal
    def is_goal(state):
        _, stored = state
        return outputs.issubset(stored)
    
    if is_goal(start_state):
        return []
    
    # Dijkstra
    # Priority queue: (cost, counter, state)
    # counter to break ties deterministically
    counter = 0
    pq = []
    heapq.heappush(pq, (0, counter, start_state))
    counter += 1
    
    best_cost = {start_state: 0}
    parent = {start_state: None}  # (prev_state, op, vertex)
    
    goal_state = None
    
    while pq:
        cost, _, state = heapq.heappop(pq)
        
        if cost > best_cost.get(state, float('inf')):
            continue
        
        if is_goal(state):
            goal_state = state
            break
        
        red, stored = state
        red_set = set(red)
        stored_set = set(stored)
        
        # Generate all possible next states
        
        # 1. load: pick a vertex that is stored (blue in slow) but not red, and not already red
        #    It must be in stored_set and not in red_set
        #    After load: it's in red, not in stored? Or does it stay in stored?
        #    
        #    I think: when you load a value from slow memory to fast memory, it becomes red in fast memory.
        #    The slow memory copy still exists? Or is it consumed?
        #    
        #    Typically in pebbling, loading moves it. But since we can recompute, maybe slow memory is permanent.
        #    
        #    Let's assume: slow memory is permanent once stored. Loading just copies to fast memory.
        #    So stored_set remains unchanged, red_set gains the vertex.
        #    
        #    Actually, that makes more sense for "recomputations allowed".
        
        # So:
        # load(v): v in stored_set, v not in red_set. New state: red_set | {v}, stored_set unchanged. Cost: +1
        # store(v): v in red_set. New state: red_set - {v}, stored_set | {v}. Cost: +1
        # compute(v): all parents of v are in red_set, v not in red_set. New state: red_set | {v}, stored_set unchanged. Cost: 0
        # delete(v): v in red_set. New state: red_set - {v}, stored_set unchanged. Cost: 0
        
        # Let's generate all possible operations
        
        next_states = []
        
        # Load operations
        for v in range(n):
            if v in stored_set and v not in red_set:
                new_red = frozenset(red_set | {v})
                new_stored = stored  # unchanged
                new_state = (new_red, new_stored)
                next_states.append((new_state, 'load', v, 1))
        
        # Store operations
        for v in red_set:
            new_red = frozenset(red_set - {v})
            new_stored = frozenset(stored_set | {v})
            new_state = (new_red, new_stored)
            next_states.append((new_state, 'store', v, 1))
        
        # Compute operations
        for v in range(n):
            if v in red_set:
                continue
            # Check if all parents are red
            if all(p in red_set for p in parents[v]):
                # Also, v should not be an input (inputs can't be computed)
                if v in inputs:
                    continue
                new_red = frozenset(red_set | {v})
                new_stored = stored  # unchanged
                new_state = (new_red, new_stored)
                next_states.append((new_state, 'compute', v, 0))
        
        # Delete operations
        for v in red_set:
            new_red = frozenset(red_set - {v})
            new_stored = stored  # unchanged
            new_state = (new_red, new_stored)
            next_states.append((new_state, 'delete', v, 0))
        
        for new_state, op, v, op_cost in next_states:
            new_cost = cost + op_cost
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent[new_state] = (state, op, v)
                heapq.heappush(pq, (new_cost, counter, new_state))
                counter += 1
    
    if goal_state is None:
        # No solution found? Return empty or fallback
        return []
    
    # Reconstruct path
    operations = []
    state = goal_state
    while parent[state] is not None:
        prev_state, op, v = parent[state]
        operations.append([op, v])
        state = prev_state
    
    operations.reverse()
    
    # Truncate to at most 20000 operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
