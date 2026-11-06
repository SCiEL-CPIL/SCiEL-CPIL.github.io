"""DAG Pebbling - Communication-minimizing schedule with bounded fast memory."""

import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Generate a pebbling schedule for a DAG that minimizes communication.
    
    Args:
        dag: A DAG structure with topologically ordered parents and output vertices.
             Expected to have attributes like 'vertices', 'parents', 'outputs', etc.
        fast_memory_budget: Number of fast memory slots available.
    
    Returns:
        List of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Parse DAG structure
    # Handle different possible DAG formats
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
        parents_map = dict(dag.parents) if hasattr(dag, 'parents') else {}
        outputs = list(dag.outputs) if hasattr(dag, 'outputs') else []
    elif isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        parents_map = dict(dag.get('parents', {}))
        outputs = list(dag.get('outputs', []))
    elif isinstance(dag, (list, tuple)):
        # Assume format: (vertices, parents, outputs) or similar
        if len(dag) >= 3:
            vertices = list(dag[0])
            parents_map = dict(dag[1]) if isinstance(dag[1], (dict, list)) else {}
            outputs = list(dag[2])
        else:
            vertices = list(dag)
            parents_map = {}
            outputs = []
    else:
        vertices = list(dag) if hasattr(dag, '__iter__') else []
        parents_map = {}
        outputs = []
    
    if not vertices:
        return []
    
    # Build adjacency: children of each vertex
    children_map = defaultdict(list)
    for v in vertices:
        par = parents_map.get(v, [])
        if isinstance(par, (list, tuple, set)):
            for p in par:
                children_map[p].append(v)
        elif par is not None:
            children_map[par].append(v)
    
    # Compute future usage count: how many times each vertex's value is needed
    # A vertex is "used" by each of its children, and transitively
    # For our heuristic: count direct + indirect children (topological depth of use)
    
    # Topological order (vertices should already be topologically ordered)
    # But let's verify and get a proper topo order
    in_degree = {v: 0 for v in vertices}
    for v in vertices:
        par = parents_map.get(v, [])
        if isinstance(par, (list, tuple, set)):
            in_degree[v] = len(par)
        elif par is not None:
            in_degree[v] = 1
    
    # Get topological order
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    temp_in_degree = dict(in_degree)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map[v]:
            temp_in_degree[c] -= 1
            if temp_in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle detected or issue - just use original order
        topo_order = list(vertices)
    
    # Compute "importance" of each vertex:
    # importance[v] = number of times v's value will be needed in future computations
    # This includes direct children and the number of paths through them
    # Simple heuristic: count of descendants (transitive children)
    
    # For a more refined heuristic, we can compute the number of distinct 
    # computation paths that require v's value
    
    # Compute descendant count (how many vertices depend on v transitively)
    descendant_count = {}
    for v in reversed(topo_order):
        desc = set()
        for c in children_map[v]:
            desc.add(c)
            desc.update(descendant_count.get(c, set()))
        descendant_count[v] = desc
    
    # Importance: weighted by how "deep" the dependencies are
    # Vertices with many descendants that are far away are more valuable to keep
    def compute_importance(v):
        if v not in descendant_count:
            return 0
        desc = descendant_count[v]
        if not desc:
            return 1
        # Weight by depth: vertices closer to outputs matter more
        # Also weight by number of children using this value
        return len(desc) + len(children_map[v]) * 2
    
    importance = {v: compute_importance(v) for v in vertices}
    
    # Determine which vertices are inputs (no parents) and outputs
    input_vertices = [v for v in vertices if not parents_map.get(v)]
    if not outputs:
        # Outputs are vertices with no children (sinks)
        outputs = [v for v in vertices if not children_map.get(v)]
    
    # Build schedule using greedy heuristic
    # State:
    #   fast_mem: set of vertices currently in fast memory
    #   computed: set of vertices whose values have been computed (and possibly stored)
    #   operations: list of operations
    
    fast_mem = set()
    computed = set()
    operations = []
    op_count = 0
    MAX_OPS = 20000
    
    # We need all outputs to be blue (in fast memory) at the end
    # Strategy: process vertices in topological order, computing each when needed
    # Keep high-importance vertices in fast memory, evict low-importance ones when budget exceeded
    
    # First, load all input vertices (they start blue, but we may need to load them)
    # Actually, "inputs start blue" means they're already available
    # But we need to track what's in fast memory
    
    # Let's think about this more carefully:
    # - Inputs start blue (available in slow memory or already loaded)
    # - We need to compute intermediate vertices
    # - All outputs must finish blue (in fast memory)
    # - Each red value (computed but not in fast mem) consumes one fast memory slot
    #   Wait, re-reading: "every red value consumes one fast-memory slot, including newly computed results"
    #   So red values ARE in fast memory? Let me re-read.
    #   
    #   "Inputs start blue, all outputs must finish blue; every red value consumes one fast-memory slot"
    #   I think: blue = in fast memory, red = computed but evicted to slow memory
    #   Actually, typically in pebbling: 
    #   - Black pebble = value is in fast memory
    #   - White pebble = value is computed and stored in slow memory
    #   - No pebble = value not computed
    #   
    #   But the task says "red value consumes one fast-memory slot" which is confusing.
    #   Let me assume: blue = in fast memory (costs a slot), red = in slow memory (free)
    #   Operations: load (slow->fast), store (fast->slow), compute (create value), delete (remove)
    
    # Let me assume:
    # - Initially: inputs are blue (in fast memory) if budget allows, else we need to load them
    # - Actually "inputs start blue" suggests they're already in fast memory
    # - We compute vertices, keeping them blue if possible
    # - When budget is exceeded, we store (evict) low-priority vertices
    # - At the end, all outputs must be blue
    
    # Revised strategy:
    # 1. Start with inputs in fast memory (if they fit)
    # 2. Process in topological order
    # 3. When computing a vertex, if it's not in fast memory, we need its parents to be blue
    # 4. If budget exceeded, evict lowest-importance non-output vertices
    
    # Let's track:
    #   in_fast: set of vertices currently in fast memory (blue)
    #   in_slow: set of vertices that have been computed and stored in slow memory
    
    in_fast = set()
    in_slow = set()
    
    # Initialize: inputs start blue
    for v in input_vertices:
        if len(in_fast) < fast_memory_budget:
            in_fast.add(v)
        else:
            # Need to make room - but inputs should fit
            pass
    
    # If budget is too small for all inputs, we'll need to load/unload as needed
    
    # Process vertices in topological order
    for v in topo_order:
        if time.time() > DEADLINE:
            break
        if op_count >= MAX_OPS:
            break
        
        # If v is an input, it should already be handled
        if v in input_vertices:
            # Ensure it's in fast memory if needed later
            continue
        
        # To compute v, all its parents must be in fast memory (blue)
        parents = parents_map.get(v, [])
        if isinstance(parents, (list, tuple, set)):
            parents = list(parents)
        elif parents is not None:
            parents = [parents]
        else:
            parents = []
        
        # Check if all parents are in fast memory
        missing_parents = [p for p in parents if p not in in_fast]
        
        if missing_parents:
            # Need to load missing parents from slow memory
            for p in missing_parents:
                if p in in_slow:
                    # Load from slow to fast
                    # First, make room if needed
                    if len(in_fast) >= fast_memory_budget:
                        # Evict a vertex: choose lowest importance, not an output, not a parent we need
                        evict_candidates = [x for x in in_fast if x not in outputs and x not in missing_parents]
                        if evict_candidates:
                            evict_v = min(evict_candidates, key=lambda x: importance.get(x, 0))
                            in_fast.remove(evict_v)
                            in_slow.add(evict_v)
                            operations.append(['store', evict_v])
                            op_count += 1
                        else:
                            # Can't evict, this is a problem
                            pass
                    in_fast.add(p)
                    operations.append(['load', p])
                    op_count += 1
                else:
                    # Parent not computed yet - shouldn't happen in topo order
                    pass
        
        # Now compute v
        # v's value will be in fast memory
        in_fast.add(v)
        operations.append(['compute', v])
        op_count += 1
        
        # If budget exceeded, evict low-importance vertices
        while len(in_fast) > fast_memory_budget:
            # Evict lowest importance vertex that's not an output
            evict_candidates = [x for x in in_fast if x not in outputs]
            if not evict_candidates:
                # All in fast mem are outputs - can't evict
                break
            evict_v = min(evict_candidates, key=lambda x: importance.get(x, 0))
            in_fast.remove(evict_v)
            in_slow.add(evict_v)
            operations.append(['store', evict_v])
            op_count += 1
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v not in in_fast:
            if v in in_slow:
                # Make room
                while len(in_fast) >= fast_memory_budget:
                    evict_candidates = [x for x in in_fast if x not in outputs and x != v]
                    if not evict_candidates:
                        break
                    evict_v = min(evict_candidates, key=lambda x: importance.get(x, 0))
                    in_fast.remove(evict_v)
                    in_slow.add(evict_v)
                    operations.append(['store', evict_v])
                    op_count += 1
                in_fast.add(v)
                operations.append(['load', v])
                op_count += 1
            elif op_count < MAX_OPS:
                # Compute it if not already computed
                # This shouldn't happen if we processed in topo order
                in_fast.add(v)
                operations.append(['compute', v])
                op_count += 1
    
    return operations


# Fallback baseline if the above doesn't work well
def pebble_baseline(dag, fast_memory_budget):
    """Simple baseline: compute in topological order, minimal eviction."""
    # This is a simplified version - the main pebble function above should be used
    return pebble(dag, fast_memory_budget)
