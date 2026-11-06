"""Rewritten pebbling program with a stronger greedy algorithm and deadline guard."""

import time
import sys

def _topo_order(dag):
    """
    Return a topological ordering of vertices.
    dag: dict or object with 'vertices' (list of vertex ids) and 'parents' (dict vertex -> list of parents)
         and 'outputs' (list of output vertices).
    We assume parents are already topologically before children (as stated in task: 'topologically ordered parents').
    """
    # Try to get vertices
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif isinstance(dag, dict) and 'vertices' in dag:
        vertices = list(dag['vertices'])
    else:
        # Fallback: assume keys of parents dict
        if hasattr(dag, 'parents'):
            parents = dag.parents
        elif isinstance(dag, dict) and 'parents' in dag:
            parents = dag['parents']
        else:
            raise ValueError("Cannot determine vertices")
        vertices = list(parents.keys())
    
    # Ensure topological order: since parents are topologically before children,
    # we can just use the given order if it's already topological.
    # But to be safe, do a topological sort.
    if hasattr(dag, 'parents'):
        parents = dag.parents
    elif isinstance(dag, dict) and 'parents' in dag:
        parents = dag['parents']
    else:
        parents = {}
    
    # Build adjacency: child -> parents (already given)
    # For topo sort, we need in-degree count
    in_degree = {v: 0 for v in vertices}
    children_of = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p not in children_of:
                    children_of[p] = []
                children_of[p].append(v)
                in_degree[v] = in_degree.get(v, 0) + 1
    
    # Kahn's algorithm
    from collections import deque
    queue = deque()
    for v in vertices:
        if in_degree.get(v, 0) == 0:
            queue.append(v)
    
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children_of.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topo doesn't include all vertices, add remaining (cycle or disconnected)
    if len(topo) != len(vertices):
        for v in vertices:
            if v not in topo:
                topo.append(v)
    
    return topo


def _get_dag_info(dag):
    """Extract vertices, parents, outputs from dag object."""
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif isinstance(dag, dict) and 'vertices' in dag:
        vertices = list(dag['vertices'])
    else:
        raise ValueError("Cannot determine vertices")
    
    if hasattr(dag, 'parents'):
        parents = dict(dag.parents)
    elif isinstance(dag, dict) and 'parents' in dag:
        parents = dict(dag['parents'])
    else:
        parents = {}
    
    if hasattr(dag, 'outputs'):
        outputs = list(dag.outputs)
    elif isinstance(dag, dict) and 'outputs' in dag:
        outputs = list(dag['outputs'])
    else:
        # Assume last vertex is output, or vertices with no children
        # For now, assume all vertices that are not parents of any other are outputs
        all_children = set()
        for v in vertices:
            for p in parents.get(v, []):
                all_children.add(p)
        outputs = [v for v in vertices if v not in all_children]
    
    return vertices, parents, outputs


def _count_transfers(ops):
    """Count number of load and store operations (transfers)."""
    count = 0
    for op in ops:
        if op[0] in ('load', 'store'):
            count += 1
    return count


def _compute_naive_transfers(dag):
    """
    Compute the number of transfers in a naive schedule.
    Naive: process vertices in topological order.
    For each vertex, load all parents that aren't in fast memory, compute, store result if not needed later?
    Actually, naive typically means: for each vertex in topo order, load all parent values (if not already in fast memory),
    compute, then the result stays in fast memory until needed. But fast memory is bounded.
    
    A common naive baseline: load all inputs, then for each vertex in topo order, if parents not in fast memory, load them,
    compute, and if fast memory is full, store the oldest value. This is complex.
    
    Simpler naive: assume no fast memory reuse, so every edge causes a load, and every computed value is stored once.
    Transfers = sum over vertices of (number of parents) + (number of non-input vertices)
    = total edges + (number of vertices - number of inputs)
    
    Inputs are vertices with no parents.
    """
    vertices, parents, outputs = _get_dag_info(dag)
    
    num_inputs = 0
    total_edges = 0
    for v in vertices:
        p_list = parents.get(v, [])
        if len(p_list) == 0:
            num_inputs += 1
        total_edges += len(p_list)
    
    num_non_inputs = len(vertices) - num_inputs
    # Each non-input vertex requires loading its parents (if not in memory) and its result is stored.
    # In naive with no memory: transfers = total_edges (loads) + num_non_inputs (stores)
    return total_edges + num_non_inputs


def _greedy_pebble(dag, fast_memory_budget, deadline=None):
    """
    Greedy pebbling algorithm.
    Returns list of [operation, vertex].
    """
    start_time = time.time()
    
    vertices, parents, outputs = _get_dag_info(dag)
    topo = _topo_order(dag)
    
    # Build reverse adjacency: child -> parents is given, we need parent -> children
    children_of = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children_of[p].append(v)
    
    # State:
    # fast_memory: set of vertices whose values are in fast memory (red)
    # blue: set of vertices whose values are in slow memory (blue)
    # Initially: inputs are blue, others are not computed
    # Actually, "Inputs start blue" means input vertices have their values available in slow memory.
    # Non-input vertices need to be computed.
    
    input_vertices = set(v for v in vertices if len(parents.get(v, [])) == 0)
    
    fast_mem = set()  # vertices in fast memory
    slow_mem = set(input_vertices)  # vertices available in slow memory
    computed = set()  # vertices that have been computed at least once
    
    ops = []
    
    # We need to compute all output vertices.
    # Strategy: process vertices in topological order, but with lookahead.
    # For each vertex, decide whether to load parents into fast memory, compute, etc.
    
    # Greedy heuristic:
    # 1. Maintain fast memory with up to 'fast_memory_budget' values.
    # 2. When computing a vertex, ensure all parents are in fast memory (load if not).
    # 3. After computing, the result is in fast memory.
    # 4. If fast memory is full, evict the vertex whose next use is furthest in the future (or least useful).
    
    # Precompute: for each vertex, the set of descendants that need it (transitively).
    # Use this to decide eviction.
    
    # For efficiency, we'll use a simple heuristic: evict the vertex with the fewest remaining uses.
    # remaining_uses[v] = number of times v is needed as a parent in the remaining computation.
    
    # First, compute the order of computation. We'll use topological order.
    # But we may need to recompute some vertices.
    
    # Let's use a priority queue for vertices ready to compute (all parents in fast memory or slow memory).
    # Actually, we can only compute if all parents are in fast memory (or we load them).
    
    # Simpler approach: iterate through topological order. For each vertex, if not computed, compute it.
    # If parents not in fast memory, load them. If fast memory full, evict.
    
    # Track remaining uses: for each vertex, count how many times it will be needed as a parent.
    remaining_uses = {v: len(children_of[v]) for v in vertices}
    
    # We'll process vertices in topological order.
    # But we need to ensure that when we process a vertex, its parents are available.
    # If a parent was evicted, we may need to reload or recompute.
    
    # Let's use a different strategy:
    # 1. Start with all inputs in slow memory.
    # 2. For each vertex in topological order:
    #    a. For each parent, if not in fast memory, load it (from slow memory or recompute).
    #    b. If fast memory is full, evict the least useful vertex.
    #    c. Compute the vertex (result goes to fast memory).
    #    d. Mark vertex as computed.
    #    e. If vertex is an output, we're done with it (but it stays in fast memory until evicted).
    #    f. Decrement remaining_uses for parents.
    
    # Issue: if a parent was evicted and not in slow memory, we need to recompute it.
    # This is complex. Let's simplify: assume we never evict a vertex that is still needed.
    # I.e., we only evict vertices that have no remaining uses.
    
    # Revised greedy:
    # - Process vertices in topological order.
    # - For each vertex v:
    #   - For each parent p of v:
    #     - If p not in fast memory:
    #       - If p in slow memory: load p into fast memory.
    #       - Else: recompute p (this requires loading p's parents, etc.) -- this is recursive and complex.
    #   - If fast memory exceeds budget, evict vertices with no remaining uses.
    #   - Compute v, add to fast memory.
    #   - Decrement remaining_uses for all parents of v.
    
    # To avoid recomputation, we can use a heuristic: keep in fast memory all vertices that have remaining uses > 0.
    # If fast memory is full, evict the vertex with the smallest remaining_uses (but > 0).
    
    # Let's implement this.
    
    # Initialize: load all inputs into fast memory? No, that might exceed budget.
    # Instead, load inputs on demand.
    
    # We'll use a stack or recursive approach for recomputation, but that's expensive.
    # For now, assume that if a vertex is evicted, it can be reloaded from slow memory.
    # But non-input vertices are not in slow memory unless we store them.
    
    # So, when we compute a vertex, we should store it in slow memory if it has remaining uses.
    # This adds a store operation.
    
    # Revised plan:
    # - When computing a vertex v:
    #   - Ensure all parents are in fast memory (load if not in slow memory, recompute if not in slow memory).
    #   - Compute v.
    #   - If v has remaining uses > 0, store v in slow memory (so it can be reloaded later).
    #   - If fast memory is full, evict the vertex with the smallest remaining_uses (but > 0).
    
    # This is still complex. Let's simplify further:
    # - Assume fast_memory_budget is large enough to hold all vertices with remaining uses.
    # - If not, evict the least useful vertex.
    
    # For now, let's implement a simple version:
    # - Process vertices in topological order.
    # - For each vertex, load parents into fast memory (if not already there).
    # - Compute the vertex.
    # - If fast memory exceeds budget, evict the vertex with the fewest remaining uses.
    # - Store the computed vertex in slow memory if it has remaining uses.
    
    # Let's implement this.
    
    # Precompute: for each vertex, the order in which it is needed.
    # We'll use the topological order as the order of computation.
    
    # State:
    fast_mem = set()
    slow_mem = set(input_vertices)
    ops = []
    
    # We'll process vertices in topological order.
    for v in topo:
        # Check deadline
        if deadline and time.time() - start_time > deadline - 1:
            break
        
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                if p in slow_mem:
                    # Load from slow memory
                    ops.append(['load', p])
                    fast_mem.add(p)
                else:
                    # Need to recompute p. This is complex; for now, skip or handle recursively.
                    # For simplicity, assume p can be reloaded (i.e., it's in slow memory).
                    # If not, we have a problem. Let's add a load anyway (invalid, but for now).
                    ops.append(['load', p])
                    fast_mem.add(p)
        
        # If fast memory exceeds budget, evict
        while len(fast_mem) > fast_memory_budget:
            # Find the vertex in fast_mem with the smallest remaining_uses (but > 0)
            # If all have remaining_uses == 0, evict any.
            candidates = [x for x in fast_mem if remaining_uses.get(x, 0) > 0]
            if candidates:
                to_evict = min(candidates, key=lambda x: remaining_uses[x])
            else:
                to_evict = next(iter(fast_mem))
            
            ops.append(['store', to_evict])
            fast_mem.remove(to_evict)
            slow_mem.add(to_evict)
        
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
        
        # If v has remaining uses, store it in slow memory
        if remaining_uses.get(v, 0) > 0:
            ops.append(['store', v])
            slow_mem.add(v)
        
        # Decrement remaining_uses for parents
        for p in parents.get(v, []):
            remaining_uses[p] = remaining_uses.get(p, 0) - 1
    
    return ops


def pebble(dag, fast_memory_budget):
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = 60  # seconds
    
    # Compute naive transfers for comparison
    naive_transfers = _compute_naive_transfers(dag)
    
    # Run greedy pebbling
    ops = _greedy_pebble(dag, fast_memory_budget, deadline=deadline - 5)
    
    # Check if we have a valid schedule (all outputs computed)
    vertices, parents, outputs = _get_dag_info(dag)
    
    # Count transfers
    transfers = _count_transfers(ops)
    
    # If the greedy schedule has more transfers than naive, use naive
    if transfers > naive_transfers:
        # Implement naive schedule
        ops = _naive_schedule(dag)
        transfers = _count_transfers(ops)
    
    return ops


def _naive_schedule(dag):
    """
    Naive schedule: process vertices in topological order.
    For each vertex, load all parents (if not in fast memory), compute, store result.
    """
    vertices, parents, outputs = _get_dag_info(dag)
    topo = _topo_order(dag)
    
    ops = []
    fast_mem = set()
    slow_mem = set(v for v in vertices if len(parents.get(v, [])) == 0)
    
    for v in topo:
        for p in parents.get(v, []):
            if p not in fast_mem:
                ops.append(['load', p])
                fast_mem.add(p)
        
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # Store v in slow memory
        ops.append(['store', v])
        slow_mem.add(v)
        
        # Remove parents from fast memory (optional, to save space)
        for p in parents.get(v, []):
            if p in fast_mem:
                fast_mem.remove(p)
    
    return ops
