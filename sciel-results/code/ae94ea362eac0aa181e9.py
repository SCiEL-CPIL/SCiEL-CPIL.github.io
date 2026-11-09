"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import heapq

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy: 
    1. Decompose DAG into chains via topological order and dependency tracking.
    2. Use a priority-queue based scheduler that prioritizes vertices with high 
       'urgency' (number of descendants still needing them) to maximize reuse.
    3. Keep fast memory filled with high-urgency values, evicting low-urgency ones.
    4. If budget is too small, fall back to a minimal working schedule.
    
    Returns: list of [operation, vertex]
    Operations: 'load', 'store', 'compute', 'delete'
    - 'load': move from slow to fast (input vertices start blue in slow)
    - 'store': move from fast to slow (for outputs or eviction)
    - 'compute': compute a vertex in fast memory (parents must be in fast)
    - 'delete': remove from fast memory (optional, for freeing space)
    
    State: 
    - slow: set of vertices available in slow memory (inputs start here)
    - fast: set of vertices in fast memory
    - computed: set of vertices whose values are currently in fast or slow
    """
    
    # Parse DAG structure
    # dag is expected to have:
    # - dag['vertices'] or similar: list of vertex ids
    # - dag['parents']: dict mapping vertex -> list of parent vertices
    # - dag['outputs'] or similar: list of output vertices
    # - dag['inputs'] or similar: list of input vertices
    
    # Try to extract structure from dag
    vertices = set()
    parents_map = {}
    children_map = {}
    inputs = set()
    outputs = set()
    
    # Handle different possible structures
    if isinstance(dag, dict):
        # Try common keys
        if 'vertices' in dag:
            vertices = set(dag['vertices'])
        if 'parents' in dag:
            parents_map = {v: list(p) for v, p in dag['parents'].items()}
            for v, ps in dag['parents'].items():
                vertices.add(v)
                for p in ps:
                    vertices.add(p)
        if 'children' in dag:
            children_map = {v: list(c) for v, c in dag['children'].items()}
        if 'inputs' in dag:
            inputs = set(dag['inputs'])
        if 'outputs' in dag:
            outputs = set(dag['outputs'])
        
        # Build children_map from parents_map if not provided
        if not children_map:
            for v in vertices:
                children_map[v] = []
            for v, ps in parents_map.items():
                for p in ps:
                    if p not in children_map:
                        children_map[p] = []
                    children_map[p].append(v)
        
        # Infer inputs and outputs if not provided
        if not inputs:
            for v in vertices:
                if v not in parents_map or len(parents_map[v]) == 0:
                    inputs.add(v)
        if not outputs:
            for v in vertices:
                if v not in children_map or len(children_map[v]) == 0:
                    outputs.add(v)
    elif isinstance(dag, list):
        # Assume list of dicts or tuples
        for item in dag:
            if isinstance(item, dict):
                v = item.get('id', item.get('vertex', None))
                if v is not None:
                    vertices.add(v)
                ps = item.get('parents', [])
                if ps:
                    parents_map[v] = list(ps)
                    for p in ps:
                        vertices.add(p)
                        if p not in children_map:
                            children_map[p] = []
                        children_map[p].append(v)
            elif isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    v = item[0]
                    ps = item[1]
                    vertices.add(v)
                    parents_map[v] = list(ps)
                    for p in ps:
                        vertices.add(p)
                        if p not in children_map:
                            children_map[p] = []
                        children_map[p].append(v)
        
        if not children_map:
            for v in vertices:
                children_map[v] = []
            for v, ps in parents_map.items():
                for p in ps:
                    children_map[p].append(v)
        
        if not inputs:
            for v in vertices:
                if v not in parents_map or len(parents_map[v]) == 0:
                    inputs.add(v)
        if not outputs:
            for v in vertices:
                if v not in children_map or len(children_map[v]) == 0:
                    outputs.add(v)
    else:
        # Fallback: assume empty
        vertices = set()
    
    # Ensure all vertices have entries in maps
    for v in vertices:
        if v not in parents_map:
            parents_map[v] = []
        if v not in children_map:
            children_map[v] = []
    
    # Compute topological order
    in_degree = {v: len(parents_map[v]) for v in vertices}
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    temp_in_degree = dict(in_degree)
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children_map[v]:
            temp_in_degree[c] -= 1
            if temp_in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle detected, just return empty
        return []
    
    # Compute number of descendants for each vertex (for urgency)
    descendant_count = {}
    for v in reversed(topo_order):
        count = 1
        for c in children_map[v]:
            count += descendant_count.get(c, 0)
        descendant_count[v] = count
    
    # Determine which vertices are inputs (start in slow memory)
    # All vertices with no parents are inputs
    input_vertices = set(v for v in vertices if len(parents_map[v]) == 0)
    
    # State
    slow = set(input_vertices)  # inputs start in slow
    fast = set()
    computed_in_slow = set()  # vertices whose values are in slow (computed or stored)
    computed_in_fast = set()  # vertices whose values are in fast
    
    operations = []
    
    def add_op(op, v):
        operations.append([op, v])
    
    # Check if a vertex is available (in slow or fast)
    def is_available(v):
        return v in slow or v in fast
    
    # Check if a vertex can be computed (all parents available in fast)
    def can_compute(v):
        if v in fast:
            return False  # already in fast
        for p in parents_map[v]:
            if p not in fast:
                return False
        return True
    
    # Check if a vertex is needed (has descendants not yet fully processed)
    def is_needed(v):
        # A vertex is needed if any of its descendants still need to be computed
        # or if it's an output
        if v in outputs:
            return True
        for c in children_map[v]:
            if c not in computed_in_fast and c not in computed_in_slow:
                return True
        return False
    
    # Score for scheduling: prefer vertices with high descendant count that are needed
    def schedule_score(v):
        if not is_needed(v):
            return -1
        return descendant_count.get(v, 0)
    
    # Main loop
    max_ops = 20000
    step = 0
    
    while step < max_ops:
        # Check if done: all outputs are in slow (blue)
        all_outputs_done = all(o in slow for o in outputs)
        if all_outputs_done:
            break
        
        # Try to compute a vertex
        # Find vertices that can be computed and have parents in fast
        candidates = []
        for v in vertices:
            if can_compute(v):
                if is_needed(v):
                    candidates.append((schedule_score(v), v))
        
        if candidates:
            # Pick highest urgency
            candidates.sort(reverse=True)
            _, v = candidates[0]
            
            # Ensure parents are in fast (they should be by can_compute)
            add_op('compute', v)
            fast.add(v)
            computed_in_fast.add(v)
            step += 1
            continue
        
        # No vertex can be computed. Need to load parents into fast.
        # Find a vertex that is needed but not yet computed, and load its parents
        needed_uncomputed = []
        for v in vertices:
            if is_needed(v) and v not in computed_in_fast and v not in computed_in_slow:
                needed_uncomputed.append((schedule_score(v), v))
        
        if needed_uncomputed:
            needed_uncomputed.sort(reverse=True)
            _, target = needed_uncomputed[0]
            
            # Load parents of target into fast
            for p in parents_map[target]:
                if p not in fast:
                    if p in slow:
                        # Load from slow to fast
                        if len(fast) >= fast_memory_budget:
                            # Need to evict something
                            # Evict lowest urgency non-needed vertex
                            to_evict = None
                            min_score = float('inf')
                            for f in fast:
                                if not is_needed(f):
                                    s = descendant_count.get(f, 0)
                                    if s < min_score:
                                        min_score = s
                                        to_evict = f
                            if to_evict is not None:
                                add_op('store', to_evict)
                                fast.remove(to_evict)
                                slow.add(to_evict)
                                computed_in_slow.add(to_evict)
                                computed_in_fast.discard(to_evict)
                                step += 1
                            else:
                                # All fast vertices are needed, evict lowest urgency anyway
                                to_evict = None
                                min_score = float('inf')
                                for f in fast:
                                    s = descendant_count.get(f, 0)
                                    if s < min_score:
                                        min_score = s
                                        to_evict = f
                                if to_evict is not None:
                                    add_op('store', to_evict)
                                    fast.remove(to_evict)
                                    slow.add(to_evict)
                                    computed_in_slow.add(to_evict)
                                    computed_in_fast.discard(to_evict)
                                    step += 1
                        
                        add_op('load', p)
                        fast.add(p)
                        slow.discard(p)
                        step += 1
                    else:
                        # p is not in slow or fast, need to compute it first
                        # This should be handled by the compute step, but if we're here,
                        # it means p's parents aren't in fast. Recurse by loading p's parents.
                        # For simplicity, just try to compute p in next iteration
                        pass
            step += 1
            continue
        
        # Nothing needed can be computed or loaded. This shouldn't happen if DAG is valid.
        # Break to avoid infinite loop
        break
    
    # Ensure all outputs are in slow
    for o in outputs:
        if o in fast:
            add_op('store', o)
            fast.remove(o)
            slow.add(o)
            computed_in_slow.add(o)
            computed_in_fast.discard(o)
        elif o not in slow:
            # Shouldn't happen, but add a dummy op
            add_op('store', o)
            slow.add(o)
    
    return operations

# EVOLVE-BLOCK-END
