"""DAG pebbling with communication minimization using dynamic scheduling."""
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimize the number of memory transfers (load/store) when pebbling a DAG.
    
    The DAG is given with topologically ordered parents and output vertices.
    Inputs start blue (in memory). All outputs must finish blue.
    Every red value consumes one fast-memory slot, including newly computed results.
    
    Strategy: We use a greedy approach that prioritizes computing vertices whose
    results will be used multiple times (to avoid recomputation) and carefully
    manages the fast memory budget by evicting values that are least likely to be
    reused soon.
    
    Operations: load, store, compute, delete
    """
    # Parse the DAG structure
    # dag format: assume it's a dict or has attributes
    # Let's infer from typical usage: dag has 'vertices', 'parents', 'outputs'
    # or similar structure
    
    # Try to extract structure from dag
    try:
        if hasattr(dag, 'vertices'):
            vertices = dag.vertices
            parents = dag.parents  # dict: vertex -> list of parent vertices
            outputs = dag.outputs  # list of output vertices
        elif isinstance(dag, dict):
            vertices = dag.get('vertices', [])
            parents = dag.get('parents', {})
            outputs = dag.get('outputs', [])
        else:
            # Assume dag is a tuple/list
            vertices, parents, outputs = dag[0], dag[1], dag[2]
    except:
        # Fallback: try to access attributes directly
        vertices = getattr(dag, 'vertices', None)
        parents = getattr(dag, 'parents', None)
        outputs = getattr(dag, 'outputs', None)
    
    if vertices is None or parents is None or outputs is None:
        # Very defensive fallback - assume simple structure
        # If dag is just a list of (vertex, parents) tuples
        if isinstance(dag, list):
            vertices = []
            parents = {}
            outputs = []
            for i, item in enumerate(dag):
                if isinstance(item, tuple) and len(item) == 2:
                    v, p = item
                    vertices.append(v)
                    parents[v] = p
            outputs = [vertices[-1]] if vertices else []
        else:
            raise ValueError("Cannot parse dag structure")
    
    # Build adjacency structure
    # children: vertex -> list of vertices that depend on it
    children = defaultdict(list)
    for v in vertices:
        if v not in parents:
            parents[v] = []
        for p in parents[v]:
            children[p].append(v)
    
    # Compute in-degree (number of parents not yet computed) for each vertex
    # and out-degree (number of children)
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    out_degree = {v: len(children.get(v, [])) for v in vertices}
    
    # Topological order
    topo_order = list(vertices)  # Already topologically ordered per problem statement
    
    # Identify input vertices (no parents) and output vertices
    input_vertices = [v for v in vertices if len(parents.get(v, [])) == 0]
    
    # State tracking
    # fast_memory: set of vertices currently in fast memory
    fast_memory = set()
    
    # Operations log
    operations = []
    
    # Count remaining uses for each vertex (how many times it's needed)
    # remaining_uses[v] = number of children of v that haven't been processed yet
    remaining_uses = dict(out_degree)
    
    # Track which vertices have been computed
    computed = set()
    # Track which vertices are currently in fast memory
    in_fast = set()
    
    # Track which children have been satisfied for each vertex
    # satisfied_children[v] = set of children of v that have already been computed
    satisfied_children = defaultdict(set)
    
    # Initialize: all input vertices are blue (available)
    # We assume inputs are available without needing to load them
    # but let's be safe and track them
    
    # Process vertices in topological order
    # Use a priority queue to decide which vertex to compute next
    # Priority: (negative remaining_uses, vertex_index) - prioritize high reuse
    
    # Actually, let me think about this differently.
    # We need to ensure that when we compute a vertex, all its parents are available.
    # A parent is available if it's in fast memory or it's an input.
    
    # Let's use a worklist algorithm:
    # 1. Start with all input vertices as available
    # 2. When a vertex's all parents are available, it can be computed
    # 3. After computing, decrement remaining_uses of its parents
    # 4. When a parent's remaining_uses reaches 0, it can be deleted from fast memory
    
    # Available vertices: those that can be computed now (all parents in fast memory or inputs)
    available = []
    
    # First, mark input vertices as available in fast memory
    # Actually, inputs start blue, so they're already in memory? Or do we need to load them?
    # "Inputs start blue" - this suggests they're already available, no load needed
    
    # Let's assume inputs are free to use without transfer cost
    
    for v in vertices:
        p = parents.get(v, [])
        if len(p) == 0:
            # Input vertex, mark as available
            heapq.heappush(available, (0, v))  # priority 0 for inputs
    
    # Track which vertices are "free" (inputs) vs those that need to be in fast memory
    is_input = set(v for v in vertices if len(parents.get(v, [])) == 0)
    
    # Now process
    # We'll maintain a set of vertices that are currently "ready" to be computed
    # (all their parents are either inputs or in fast memory)
    
    ready = set()
    for v in input_vertices:
        ready.add(v)
    
    # Priority queue for ready vertices: prioritize those with high remaining_uses
    # to maximize reuse
    pq = []
    for v in ready:
        # Higher remaining_uses means more valuable to keep in memory
        heapq.heappush(pq, (-remaining_uses[v], v))
    
    # Main loop
    while pq:
        # Pick the highest priority ready vertex
        neg_reuse, v = heapq.heappop(pq)
        
        # Check if v is still ready (its parents might have been deleted)
        # Actually, once a vertex is ready, it stays ready until computed
        # But we need to ensure its parents are still available
        # Since we only compute vertices when all parents are available,
        # and we don't delete parents until their remaining_uses is 0,
        # this should be fine
        
        # Compute vertex v
        operations.append(('compute', v))
        computed.add(v)
        in_fast.add(v)
        fast_memory.add(v)
        
        # Check fast memory budget
        if len(fast_memory) > fast_memory_budget:
            # Need to evict a vertex
            # Find a vertex to evict: one with lowest remaining_uses
            # and not an input (inputs don't take up budget? or do they?)
            # Let's assume all vertices in fast_memory take up budget
            # But inputs are special - they can be reloaded for free?
            # Actually, "Inputs start blue" might mean they're in a separate memory
            
            # Let's assume inputs don't count toward fast_memory_budget
            # So only computed non-input vertices count
            non_input_in_fast = [x for x in fast_memory if x not in is_input]
            if non_input_in_fast:
                # Evict the one with lowest remaining_uses
                # Ties broken by... let's say lower index
                evict_candidate = min(non_input_in_fast, key=lambda x: (remaining_uses[x], x))
                # But we shouldn't evict if remaining_uses > 0 unless we have to
                # If remaining_uses == 0, it's safe to evict
                # If remaining_uses > 0, we'll need to recompute later
                if remaining_uses[evict_candidate] == 0:
                    # Safe to evict
                    fast_memory.discard(evict_candidate)
                    in_fast.discard(evict_candidate)
                    operations.append(('delete', evict_candidate))
                else:
                    # Must evict anyway
                    fast_memory.discard(evict_candidate)
                    in_fast.discard(evict_candidate)
                    operations.append(('store', evict_candidate))  # store to slow memory
            # If only inputs in fast_memory, they don't count, so no issue
        
        # Now, for each parent of v, decrement remaining_uses
        for p in parents.get(v, []):
            remaining_uses[p] -= 1
            satisfied_children[p].add(v)
            
            # If p's remaining_uses is now 0, it can be deleted
            # But we'll handle eviction based on budget, not immediately
        
        # Now, check if any of v's children become ready
        for c in children.get(v, []):
            # Check if all parents of c are now available
            all_parents_available = True
            for pc in parents.get(c, []):
                if pc in is_input:
                    continue  # Inputs are always available
                if pc not in fast_memory:
                    all_parents_available = False
                    break
            if all_parents_available:
                # c is now ready
                heapq.heappush(pq, (-remaining_uses[c], c))
        
        # Also, after computing v, check if any previously not-ready vertices
        # become ready because v is now in fast memory
        # This is already handled by checking children above
    
    # After all computations, ensure all outputs are in fast memory (blue)
    # They should already be computed, but verify they're in fast_memory
    for o in outputs:
        if o not in fast_memory:
            # Need to load it
            operations.append(('load', o))
            fast_memory.add(o)
    
    # Now we need to ensure that at the end, all outputs are blue
    # They are in fast_memory, so they're blue
    
    # Return the operations
    # Each operation is [operation_type, vertex]
    result = []
    for op, v in operations:
        result.append([op, v])
    
    return result
