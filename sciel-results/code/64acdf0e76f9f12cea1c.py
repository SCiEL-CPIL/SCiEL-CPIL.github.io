"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import heapq
import itertools
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize total memory transfers (load + store operations).
    
    The DAG is assumed to have:
    - 'vertices': list of vertex IDs
    - 'edges': list of (parent, child) tuples, or similar structure
    - 'inputs': list of input vertex IDs (start blue)
    - 'outputs': list of output vertex IDs (must finish blue)
    - Parents are topologically ordered before children
    
    We model this as a shortest-path problem in the state space of red-pebble configurations.
    States are tuples of which vertices are in fast memory (red).
    Initial state: all inputs are red (since they start blue, we need to load them? 
    Wait: "Inputs start blue" means they are in slow memory. To compute a vertex, its parents must be red.
    So we need to load inputs first.
    
    Actually, let's re-read: "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot"
    - Blue = in slow memory (or external)
    - Red = in fast memory
    - To compute a vertex, all its parents must be red.
    - Load: move a blue vertex to red (if it's an input or was previously stored)
    - Store: move a red vertex to blue
    - Compute: if all parents are red, compute the vertex (it becomes red)
    - Delete: remove a red vertex (doesn't count as transfer? The task says "mean transfers", typically load/store are transfers)
    
    Wait, "Score mean transfers / naive schedule". Transfers are likely load + store operations.
    Compute and delete are local operations and don't count as transfers.
    
    Let's define the state as a frozenset of red vertices.
    Initial state: empty set (all inputs are blue).
    Target state: empty set (all outputs are blue, so they must be stored, and all other vertices can be anything? 
    No, "all outputs must finish blue". The state at the end should have no outputs red. 
    Actually, if we store an output, it becomes blue. We can also delete it. 
    To minimize transfers, we want to minimize loads and stores.
    
    However, the state space can be huge. The problem says "three layered DAGs". 
    This suggests the DAGs are not too wide or deep, or have a specific structure that allows efficient search.
    "Layered DAGs" might mean the vertices can be divided into layers, and edges only go between adjacent layers.
    
    If the budget is small, the number of possible states is C(n, k) where k is the budget.
    With budget 10 and n=20, that's 184756 states, which is manageable.
    With budget 20 and n=40, that's huge.
    
    Let's assume the DAGs are small enough for Dijkstra's or BFS on the state space.
    If the state space is too large, we might need a heuristic, but the directive asks for Dijkstra's.
    
    Let's first inspect the DAG structure to understand the size.
    Since we can't inspect at runtime easily without printing, we'll write a robust Dijkstra that can handle moderate state spaces.
    
    Steps:
    1. Parse the DAG to get adjacency list (parents for each vertex).
    2. Identify inputs (no parents) and outputs (no children, or specified as outputs).
    3. Define the state as a frozenset of red vertices.
    4. Use Dijkstra's algorithm where:
       - Cost of an edge is 1 if the operation is load or store, 0 if compute or delete.
       - We want to reach a state where all outputs are blue (not in the red set).
       - Actually, the problem says "all outputs must finish blue". It doesn't say other vertices must be blue.
       - So the target is any state where no output vertex is in the red set.
       - But wait, if we don't store/delete non-output vertices, they remain red. Is that allowed?
       - "every red value consumes one fast-memory slot". So we can have non-output vertices red at the end, as long as we don't exceed the budget.
       - The goal is to have all outputs blue.
       
    Let's refine the target: A state S is valid if for all o in outputs, o not in S.
    
    Operations from a state S:
    1. Load v: if v not in S, and v is an input or has been computed before (i.e., its value is available in slow memory).
       Wait, can we load any vertex? "Inputs start blue". What about computed vertices? 
       If we compute a vertex and then store it, it becomes blue. Then we can load it again.
       So a vertex is loadable if it is an input OR it has been computed and stored at least once.
       This makes the state space larger because we need to track which vertices have been "materialized" in slow memory.
       
    Alternative interpretation: 
    - Inputs are always available in slow memory.
    - Computed vertices are only available in slow memory if they have been stored.
    - So the state needs to track:
      - R: set of red vertices (in fast memory)
      - B: set of blue vertices that are available (inputs + stored computed vertices)
      
    This doubles the state complexity.
    
    Let's look at the "naive schedule" for reference. 
    A naive schedule would be:
    - Load all inputs.
    - Compute each vertex in topological order, storing results as needed.
    - This might use more transfers than optimal.
    
    Given the complexity, let's try a simpler model first:
    Assume that any vertex whose parents have been computed is "available" in slow memory? No, that's not standard pebbling.
    
    Standard pebbling rules:
    - A vertex is pebbled (red) if it's in fast memory.
    - You can place a pebble on a vertex if all its parents are pebbled (compute).
    - You can remove a pebble (delete/store).
    - You can load an input (if it's an input).
    
    The key is: can you reload a previously computed vertex? 
    In standard pebbling, once you remove a pebble, you can't get it back unless you recompute it.
    So the state is just the set of pebbled vertices.
    Inputs are special: they can always be pebbled (loaded) because they are given.
    
    So:
    - State: frozenset of red vertices.
    - Load v: allowed if v is an input. Cost 1. New state: S ∪ {v}.
    - Compute v: allowed if all parents of v are in S. Cost 0. New state: S ∪ {v}.
    - Store v: allowed if v is in S. Cost 1. New state: S \ {v}.
    - Delete v: allowed if v is in S. Cost 0. New state: S \ {v}.
      (Store and delete are the same in terms of state transition, but store makes it available in slow memory? 
       In this model, "store" and "delete" are indistinguishable unless we track slow memory availability.
       But if we can't reload non-inputs, then store and delete are the same.
       If we CAN reload non-inputs, we need to track slow memory.
       
    Let's assume the simpler model: only inputs can be loaded. Non-inputs must be recomputed if deleted/stored.
    This is consistent with "Recomputations allowed".
    
    So:
    - Load v: if v is input and v not in S. Cost 1.
    - Compute v: if all parents in S and v not in S. Cost 0.
    - Store/Delete v: if v in S. Cost 1 (for store) or 0 (for delete). 
      But if we delete, we can't reload. If we store, can we reload? 
      The task says "Inputs start blue". It doesn't say computed values start blue.
      So likely, only inputs can be loaded.
      Therefore, Store and Delete are equivalent in terms of future possibilities.
      We can just use one operation: "remove v" with cost 1 (since it's a transfer out of fast memory).
      Wait, "mean transfers" usually counts both load and store.
      So removing a vertex costs 1 transfer.
      
    Let's assume:
    - Load: cost 1
    - Store: cost 1
    - Compute: cost 0
    - Delete: cost 0 (but why would we delete instead of store? Maybe delete is for freeing space without writing to slow memory? 
      But the task says "every red value consumes one fast-memory slot". 
      If we delete, the value is gone. If we store, the value is in slow memory.
      But if we can't reload non-inputs, then store and delete are the same.
      Let's assume the cost of removing a vertex is 1 (store), and we don't have a separate delete operation.
      Or maybe delete is cost 0 and store is cost 1, but we can't reload after delete.
      Since we can't reload non-inputs anyway, let's just have one "remove" operation with cost 1.
      
    Target: A state S where all outputs are not in S.
    
    Let's implement Dijkstra's on this state space.
    """
    # Parse DAG
    # Assume dag is a dict or object with 'vertices', 'parents', 'inputs', 'outputs'
    # Let's try to infer the structure from common patterns.
    
    # For now, let's assume the following structure:
    # dag = {
    #   'vertices': [...],
    #   'parents': {v: [p1, p2, ...]},
    #   'inputs': [...],
    #   'outputs': [...]
    # }
    
    # If the structure is different, we need to handle it.
    # Let's try to be robust.
    
    try:
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    except:
        # Fallback: try to infer
        vertices = list(dag.keys()) if isinstance(dag, dict) else list(dag)
        parents = {}
        inputs = set()
        outputs = set()
        # This is risky. Let's assume the standard structure.
    
    # Build children map for topological sort if needed
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Initial state: no red vertices
    initial_state = frozenset()
    
    # Target: all outputs are not in the state
    def is_target(state):
        for o in outputs:
            if o in state:
                return False
        return True
    
    # Dijkstra's
    # State: frozenset of red vertices
    # Cost: number of transfers (loads + stores)
    
    # Priority queue: (cost, state)
    heap = [(0, initial_state)]
    visited = {initial_state: 0}
    parent_map = {}  # state -> (prev_state, operation, vertex)
    
    while heap:
        cost, state = heapq.heappop(heap)
        
        if is_target(state):
            # Reconstruct path
            path = []
            current = state
            while current in parent_map:
                prev_state, op, v = parent_map[current]
                path.append([op, v])
                current = prev_state
            path.reverse()
            return path
        
        if cost > visited.get(state, float('inf')):
            continue
            
        # Generate neighbors
        # 1. Load inputs
        for v in inputs:
            if v not in state:
                new_state = frozenset(state | {v})
                new_cost = cost + 1
                if new_cost < visited.get(new_state, float('inf')):
                    visited[new_state] = new_cost
                    parent_map[new_state] = (state, 'load', v)
                    heapq.heappush(heap, (new_cost, new_state))
        
        # 2. Compute vertices
        for v in vertices:
            if v not in state:
                # Check if all parents are in state
                can_compute = True
                for p in parents.get(v, []):
                    if p not in state:
                        can_compute = False
                        break
                if can_compute:
                    new_state = frozenset(state | {v})
                    new_cost = cost  # Compute is free
                    if new_cost < visited.get(new_state, float('inf')):
                        visited[new_state] = new_cost
                        parent_map[new_state] = (state, 'compute', v)
                        heapq.heappush(heap, (new_cost, new_state))
        
        # 3. Store/Delete vertices
        for v in state:
            new_state = frozenset(state - {v})
            new_cost = cost + 1  # Store is a transfer
            if new_cost < visited.get(new_state, float('inf')):
                visited[new_state] = new_cost
                parent_map[new_state] = (state, 'store', v)
                heapq.heappush(heap, (new_cost, new_state))
    
    # If no solution found, return empty list (should not happen)
    return []

# EVOLVE-BLOCK-END
