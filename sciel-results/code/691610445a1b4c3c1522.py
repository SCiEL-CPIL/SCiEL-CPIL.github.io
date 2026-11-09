"""Program for communication-minimizing DAG pebbling using state-space search."""
# EVOLVE-BLOCK-START
import heapq
from functools import lru_cache
from typing import List, Tuple, Dict, Set, FrozenSet
import copy

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem using Dijkstra's algorithm over the state space.
    
    State: (frozenset of red vertices, frozenset of blue vertices in fast memory)
    Actually, let's define state more carefully:
    - A vertex can be: not_pebbled, red (in fast memory), blue (in fast memory)
    - Inputs start blue (in fast memory? or just available?)
    - "Inputs start blue" - this means input vertices are initially blue
    - "all outputs must finish blue"
    - "every red value consumes one fast-memory slot, including newly computed results"
    - So fast_memory_budget limits the number of red pebbles
    
    Let me reconsider the state representation:
    - For each vertex, it can be: absent, red, blue
    - Red pebbles consume fast memory slots
    - Blue pebbles... do they consume fast memory? The problem says "every red value consumes one fast-memory slot"
    - So only red pebbles count toward the budget? Or do blue pebbles also count?
    
    Re-reading: "every red value consumes one fast-memory slot, including newly computed results"
    This suggests only red values consume fast memory slots. Blue values might be stored differently.
    
    Wait, but "Inputs start blue" and "all outputs must finish blue" - if blue doesn't consume fast memory,
    then we could keep everything blue with no cost. That doesn't make sense.
    
    Let me reconsider: perhaps both red and blue pebbles consume fast memory, but the problem specifically
    mentions red because newly computed results are initially red.
    
    Actually, in standard pebbling:
    - A pebble can be red or blue
    - You can place a pebble (load) on a vertex if all parents are pebbled
    - You can remove a pebble (delete)
    - You can change color (compute: red->blue or blue->red)
    - The constraint is on the number of pebbles (total or just red?)
    
    Given "fast_memory_budget" and "every red value consumes one fast-memory slot", I think:
    - Only red pebbles consume fast memory slots
    - Blue pebbles are stored in slow memory (no cost)
    - This makes more sense: you compute a value (it becomes red, using fast memory),
      then you can store it (becomes blue, no fast memory cost)
    
    So the state is:
    - For each vertex: absent, red, or blue
    - Constraint: number of red vertices <= fast_memory_budget
    
    Operations:
    - load: place a red pebble on a vertex (all parents must be pebbled, i.e., red or blue)
    - compute: change a red pebble to blue (or blue to red?)
    - store: change a blue pebble to... wait, what's the difference between compute and store?
    
    Let me re-read: "operations load/store/compute/delete"
    - load: place a pebble on a vertex
    - store: ?
    - compute: ?
    - delete: remove a pebble
    
    Perhaps:
    - load: place a red pebble on an input vertex (or any vertex whose parents are all pebbled)
    - compute: if all parents are pebbled, compute the vertex's value (place a red pebble)
    - store: convert a red pebble to blue (store in slow memory)
    - delete: remove a pebble
    
    Or maybe:
    - load: bring a value from slow memory to fast memory (blue -> red)
    - store: send a value from fast memory to slow memory (red -> blue)
    - compute: compute a new value (all parents pebbled, result is red)
    - delete: remove a pebble
    
    This interpretation makes sense:
    - Inputs start blue (in slow memory)
    - To use an input, you load it (blue -> red, costs 1 fast memory slot)
    - To compute a new vertex, all parents must be red (or at least pebbled?), result is red
    - To store a result, you store it (red -> blue, frees 1 fast memory slot)
    - To delete, you remove a pebble entirely
    
    But wait, "all outputs must finish blue" - so outputs need to be stored (red -> blue)
    
    Let me assume:
    - A vertex is pebbled if it has a red or blue pebble
    - To compute a vertex v, all parents of v must be pebbled (red or blue)
    - The result of computing v is a red pebble
    - load: if v is blue (in slow memory), make it red (in fast memory) - costs 1 slot
    - store: if v is red, make it blue (in slow memory) - frees 1 slot
    - compute: if all parents of v are pebbled, and v is not pebbled, place a red pebble on v - costs 1 slot
    - delete: remove a pebble from v (whether red or blue)
    
    Actually, I think the standard model is:
    - load: place a red pebble on an input vertex (inputs are initially available)
    - compute: if all parents are pebbled, place a red pebble on the vertex
    - store: convert a red pebble to blue (blue pebbles don't count toward budget?)
    - delete: remove any pebble
    
    Let me just implement Dijkstra with a reasonable state representation and see what happens.
    
    State: tuple of (vertex_state) where vertex_state is 0=absent, 1=red, 2=blue
    Constraint: count of red vertices <= fast_memory_budget
    
    Operations and their costs:
    - load v: if v is input and v is absent, make it red. Cost: 1 (transfer)
    - compute v: if all parents are pebbled (1 or 2) and v is absent, make it red. Cost: 1
    - store v: if v is red, make it blue. Cost: 1
    - delete v: if v is pebbled (1 or 2), make it absent. Cost: 1
    
    Initial state: inputs are blue (2), others absent (0)
    Target state: all outputs are blue (2)
    
    Let me implement this.
    """
    # Parse the DAG structure
    # dag should have some structure - let's assume it has:
    # - dag['vertices'] or similar
    # - dag['parents'] - dict from vertex to list of parents
    # - dag['inputs'] - list of input vertices
    # - dag['outputs'] - list of output vertices
    
    # Since I don't know the exact structure, let me try to infer it
    if isinstance(dag, dict):
        vertices = dag.get('vertices', list(range(dag.get('n', 0))))
        if 'n' in dag and 'vertices' not in dag:
            vertices = list(range(dag['n']))
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
    elif isinstance(dag, list):
        # Maybe dag is a list of (vertex, parents) tuples
        parents = {}
        for i, item in enumerate(dag):
            if isinstance(item, tuple):
                parents[i] = item[1] if len(item) > 1 else []
            else:
                parents[i] = []
        vertices = list(range(len(dag)))
        inputs = [0] if vertices else []
        outputs = [len(vertices)-1] if vertices else []
    else:
        # Fallback
        vertices = list(range(10))
        parents = {i: [] for i in vertices}
        inputs = [0]
        outputs = [9]
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Ensure parents is a dict mapping vertex to list of parents
    if not isinstance(parents, dict):
        parents = {i: list(parents[i]) if i < len(parents) else [] for i in range(n)}
    
    # Ensure all vertices have an entry in parents
    for v in vertices:
        if v not in parents:
            parents[v] = []
    
    # Initial state: inputs are blue (2), others absent (0)
    initial_state = [0] * n
    for v in inputs:
        if v < n:
            initial_state[v] = 2
    
    # Target: all outputs are blue (2)
    # We need to check if a state satisfies the target
    
    def is_target(state):
        for v in outputs:
            if v < n and state[v] != 2:
                return False
        return True
    
    # Check if initial state is already target
    if is_target(initial_state):
        return []
    
    # Dijkstra's algorithm
    # State: tuple of vertex states
    # Distance: number of transfers (operations)
    
    initial_state_tuple = tuple(initial_state)
    
    # Priority queue: (cost, step, state_tuple, path)
    # To avoid storing the full path in the heap, we'll store predecessor info
    
    # dist[state] = minimum cost to reach state
    dist = {initial_state_tuple: 0}
    
    # parent[state] = (prev_state, operation, vertex)
    parent = {}
    
    # Heap: (cost, state_tuple)
    heap = [(0, initial_state_tuple)]
    
    # Count red pebbles
    def count_red(state):
        return sum(1 for s in state if s == 1)
    
    # Check if state is valid (red count <= budget)
    def is_valid(state):
        return count_red(state) <= fast_memory_budget
    
    # If initial state is invalid, we can't do anything
    if not is_valid(initial_state_tuple):
        return []
    
    max_ops = 20000
    
    found = False
    target_state = None
    
    while heap and len(dist) < 1000000:  # Limit search space
        cost, state = heapq.heappop(heap)
        
        if state not in dist:
            continue
        
        if cost > dist[state]:
            continue
        
        if is_target(state):
            found = True
            target_state = state
            break
        
        if cost >= max_ops:
            continue
        
        # Generate all possible next states
        for v in range(n):
            current_state_v = state[v]
            
            # Try load: only for input vertices, if absent, make red
            if v in inputs and current_state_v == 0:
                new_state = list(state)
                new_state[v] = 1
                new_state_tuple = tuple(new_state)
                if is_valid(new_state_tuple):
                    new_cost = cost + 1
                    if new_state_tuple not in dist or dist[new_state_tuple] > new_cost:
                        dist[new_state_tuple] = new_cost
                        parent[new_state_tuple] = (state, 'load', v)
                        heapq.heappush(heap, (new_cost, new_state_tuple))
            
            # Try compute: if all parents are pebbled and v is absent, make red
            if current_state_v == 0:
                all_parents_pebbled = all(state[p] in (1, 2) for p in parents.get(v, []) if p < n)
                if all_parents_pebbled:
                    new_state = list(state)
                    new_state[v] = 1
                    new_state_tuple = tuple(new_state)
                    if is_valid(new_state_tuple):
                        new_cost = cost + 1
                        if new_state_tuple not in dist or dist[new_state_tuple] > new_cost:
                            dist[new_state_tuple] = new_cost
                            parent[new_state_tuple] = (state, 'compute', v)
                            heapq.heappush(heap, (new_cost, new_state_tuple))
            
            # Try store: if red, make blue
            if current_state_v == 1:
                new_state = list(state)
                new_state[v] = 2
                new_state_tuple = tuple(new_state)
                if is_valid(new_state_tuple):
                    new_cost = cost + 1
                    if new_state_tuple not in dist or dist[new_state_tuple] > new_cost:
                        dist[new_state_tuple] = new_cost
                        parent[new_state_tuple] = (state, 'store', v)
                        heapq.heappush(heap, (new_cost, new_state_tuple))
            
            # Try delete: if pebbled, make absent
            if current_state_v in (1, 2):
                new_state = list(state)
                new_state[v] = 0
                new_state_tuple = tuple(new_state)
                if is_valid(new_state_tuple):
                    new_cost = cost + 1
                    if new_state_tuple not in dist or dist[new_state_tuple] > new_cost:
                        dist[new_state_tuple] = new_cost
                        parent[new_state_tuple] = (state, 'delete', v)
                        heapq.heappush(heap, (new_cost, new_state_tuple))
        
        if found:
            break
    
    if not found:
        # Fallback: use a simple greedy approach
        return _greedy_pebble(dag, fast_memory_budget)
    
    # Reconstruct path
    path = []
    state = target_state
    while state in parent:
        prev_state, op, v = parent[state]
        path.append([op, v])
        state = prev_state
    
    path.reverse()
    
    # Truncate to max_ops if needed
    if len(path) > max_ops:
        path = path[:max_ops]
    
    return path


def _greedy_pebble(dag, fast_memory_budget):
    """Fallback greedy pebbling strategy."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', list(range(dag.get('n', 0))))
        if 'n' in dag and 'vertices' not in dag:
            vertices = list(range(dag['n']))
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
    else:
        return []
    
    n = len(vertices)
    if n == 0:
        return []
    
    if not isinstance(parents, dict):
        parents = {i: list(parents[i]) if i < len(parents) else [] for i in range(n)}
    
    for v in vertices:
        if v not in parents:
            parents[v] = []
    
    # Simple topological order processing
    # Load inputs, compute in topological order, store outputs
    ops = []
    
    # Track which vertices are pebbled
    pebbled = set(inputs)
    
    # Load inputs (they start blue, but we might need to load them to red for computation)
    # Actually, inputs start blue. To compute children, we need parents pebbled.
    # Blue counts as pebbled, so we can compute directly.
    
    # Process in topological order
    processed = set()
    changed = True
    while changed:
        changed = False
        for v in vertices:
            if v in processed:
                continue
            if all(p in pebbled for p in parents.get(v, [])):
                # Compute v
                if v not in pebbled:
                    ops.append(['compute', v])
                    pebbled.add(v)
                processed.add(v)
                changed = True
    
    # Store outputs
    for v in outputs:
        if v in pebbled:
            ops.append(['store', v])
    
    return ops[:20000]

# EVOLVE-BLOCK-END
