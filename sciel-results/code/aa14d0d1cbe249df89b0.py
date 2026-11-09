import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimizes memory transfers for DAG pebbling.
    
    Args:
        dag: A dictionary or object representing the DAG. 
             Expected structure based on typical tasks:
             - 'vertices': list of vertex IDs
             - 'edges': list of (parent, child) tuples OR
             - 'parents': dict {child: [parents]}
             - 'outputs': list of output vertex IDs
             - 'inputs': list of input vertex IDs (optional, can be inferred)
        fast_memory_budget: int, maximum number of vertices in fast memory.
    
    Returns:
        list of [operation, vertex]
    """
    # --- Parse DAG Structure ---
    # We need to handle different possible input formats.
    # Based on "dag has topologically ordered parents and output vertices", 
    # it might be a dict with 'parents' and 'outputs'.
    
    if isinstance(dag, dict):
        parents_dict = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        vertices = dag.get('vertices', [])
        
        # If vertices not provided, infer from parents_dict keys and values
        if not vertices:
            v_set = set()
            for p_list in parents_dict.values():
                v_set.update(p_list)
            v_set.update(parents_dict.keys())
            vertices = list(v_set)
    else:
        # Fallback if dag is not a dict (e.g., object)
        # Assuming standard attributes
        parents_dict = getattr(dag, 'parents', {})
        outputs = getattr(dag, 'outputs', [])
        vertices = getattr(dag, 'vertices', [])
        if not vertices:
            v_set = set()
            for p_list in parents_dict.values():
                if isinstance(p_list, list):
                    v_set.update(p_list)
                else:
                    v_set.add(p_list)
            v_set.update(parents_dict.keys())
            vertices = list(v_set)
    
    # Ensure all vertices are in parents_dict (inputs have no parents)
    for v in vertices:
        if v not in parents_dict:
            parents_dict[v] = []
            
    # Identify Inputs: Vertices with no parents
    inputs = [v for v in vertices if len(parents_dict[v]) == 0]
    
    # Identify Children (for reverse graph)
    children_dict = defaultdict(list)
    for child, parents in parents_dict.items():
        for p in parents:
            children_dict[p].append(child)
            
    # Topological Sort (Kahn's Algorithm)
    in_degree = {v: len(parents_dict[v]) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_dict[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    # If graph has cycles or issues, topo_order might be incomplete. 
    # Assume valid DAG.
    
    # --- Pebbling Algorithm ---
    # State
    fast = set()
    slow = set(inputs) # Inputs are available in slow memory
    ops = []
    
    # Remaining vertices to compute (exclude inputs? No, inputs need to be "processed" to be in fast if needed, 
    # but they are already in slow. We need to get them into fast to compute children).
    # Let's track "computed" vertices. Inputs are "computed" (available).
    # We need to ensure all outputs end up in fast.
    
    # We process vertices in a greedy manner.
    # A vertex is "ready" if all its parents are in fast.
    # We want to compute vertices that have the most children (to maximize future reuse).
    
    # Priority Queue: (-num_children, topo_index, vertex)
    # We re-evaluate readiness dynamically.
    
    unprocessed = set(vertices)
    # We must process all vertices.
    # However, we can skip processing a vertex if it's already in fast (from a previous step? No, we process each once in topo order usually, 
    # but with eviction/reload, we might need to revisit).
    
    # Simpler Robust Loop:
    # While there are unprocessed vertices (vertices that need to be computed and kept in fast at least once, 
    # or specifically, we need to ensure all vertices are computed at least once, and outputs are in fast at the end).
    
    # Actually, if we just iterate through topo_order and ensure parents are in fast, we compute each vertex exactly once.
    # The problem is eviction. If we evict a parent, we have to reload or recompute.
    
    # Let's stick to the Topological Order, but handle "missing parents" by loading/recomputing.
    
    # To handle recomputation efficiently, we need to know if a vertex is in slow.
    # If a vertex is not in fast and not in slow, we must recompute it.
    
    # Optimization: 
    # 1. Iterate through topo_order.
    # 2. For each v, ensure all parents are in fast.
    # 3. If a parent p is not in fast:
    #    a. If p is in slow: load p.
    #    b. If p is not in slow: recompute p. (This requires p's parents to be in fast, etc.)
    
    # This recursive requirement can be handled by a helper function `ensure_in_fast(v)`.
    
    deadline = time.time() + 55 # 55s safety margin
    
    def get_children_count(v):
        return len(children_dict[v])
        
    def evict_one():
        """Evicts a vertex from fast memory to make space."""
        if not fast:
            return
        # Candidates: vertices in fast that are not outputs (outputs should be kept if possible)
        # And ideally, vertices that are no longer needed (no children in unprocessed? 
        # But we are processing in topo order, so "unprocessed" is the rest of the topo_order).
        
        # Heuristic:
        # 1. Prefer evicting vertices that are NOT in the set of vertices needed for the current step or near future.
        # 2. Prefer evicting vertices with fewer children in the remaining topo_order.
        
        remaining_vertices = set()
        # We don't track remaining explicitly in the simple loop, but we can approximate.
        # For now, use global child count as a proxy, or just LIFO.
        
        # Better: Evict the vertex with the smallest number of children in the *entire* graph?
        # Or, if a vertex has 0 children in the graph, it's a leaf. If it's not an output, it's safe to evict.
        
        # Let's try: Evict the vertex with the fewest children. If tie, evict the one added most recently?
        # We need a history or just pick one.
        
        # For simplicity and speed, we will evict the vertex with the lowest "importance".
        # Importance = number of children in the DAG.
        
        min_importance = float('inf')
        victim = None
        for u in fast:
            # Do not evict outputs if possible
            if u in outputs:
                continue
            imp = len(children_dict[u])
            if imp < min_importance:
                min_importance = imp
                victim = u
                
        if victim is None:
            # All are outputs? Or fast is empty?
            # If fast is full of outputs, we are stuck. 
            # But budget should be sufficient.
            # If no non-output, pick any.
            victim = next(iter(fast))
            
        fast.remove(victim)
        if victim in slow:
            # It's in slow, so we just delete from fast. No store needed.
            ops.append(['delete', victim])
        else:
            # It's not in slow, so we must store it to slow memory to avoid recomputation later.
            ops.append(['store', victim])
            slow.add(victim)
            
    def ensure_in_fast(v):
        """Ensures vertex v is in fast memory."""
        if v in fast:
            return
        
        if time.time() > deadline:
            return # Timeout guard
            
        # If v is in slow, load it
        if v in slow:
            if len(fast) < fast_memory_budget:
                ops.append(['load', v])
                fast.add(v)
            else:
                evict_one()
                ops.append(['load', v])
                fast.add(v)
            return
        
        # If v is not in slow, we must compute it.
        # This requires all parents of v to be in fast.
        parents = parents_dict[v]
        for p in parents:
            ensure_in_fast(p)
            
        # Now all parents are in fast. Compute v.
        if len(fast) < fast_memory_budget:
            ops.append(['compute', v])
            fast.add(v)
        else:
            evict_one()
            ops.append(['compute', v])
            fast.add(v)
            
        # Mark v as available in slow? 
        # Usually, computed values are implicitly available for re-computation if needed, 
        # but to avoid recomputation, we should consider them "in slow" if we might need them again.
        # However, if we add every computed node to slow, slow grows large.
        # That's fine, slow memory is unbounded.
        slow.add(v)

    # Main Loop: Process vertices in Topological Order
    # We need to ensure all vertices are computed.
    # By iterating through topo_order and calling ensure_in_fast, we guarantee they are computed.
    # However, we must ensure OUTPUTS are in fast at the END.
    
    # Process all vertices
    for v in topo_order:
        if time.time() > deadline:
            break
        ensure_in_fast(v)
        
    # After processing all, ensure all outputs are in fast.
    # They might have been evicted.
    for o in outputs:
        if o not in fast:
            if time.time() > deadline:
                break
            ensure_in_fast(o)
            
    # Check operation limit
    if len(ops) > 20000:
        # Truncate? Or return best effort?
        # The problem says "at most 20000 operations".
        # If we exceed, the result is invalid.
        # We should try to be efficient.
        pass
        
    return ops

# Helper to handle if dag structure is slightly different
# The above code assumes dag is a dict or object with specific attributes.
# To be safe, we can wrap the parsing.

def pebble(dag, fast_memory_budget):
    import time
    from collections import defaultdict, deque
    
    # Parse DAG
    if isinstance(dag, dict):
        parents_dict = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        vertices = dag.get('vertices', [])
        if not vertices:
            v_set = set()
            for p_list in parents_dict.values():
                if isinstance(p_list, list):
                    v_set.update(p_list)
                else:
                    v_set.add(p_list)
            v_set.update(parents_dict.keys())
            vertices = list(v_set)
    else:
        parents_dict = getattr(dag, 'parents', {})
        outputs = getattr(dag, 'outputs', [])
        vertices = getattr(dag, 'vertices', [])
        if not vertices:
            v_set = set()
            for p_list in parents_dict.values():
                if isinstance(p_list, list):
                    v_set.update(p_list)
                else:
                    v_set.add(p_list)
            v_set.update(parents_dict.keys())
            vertices = list(v_set)
            
    for v in vertices:
        if v not in parents_dict:
            parents_dict[v] = []
            
    inputs = [v for v in vertices if len(parents_dict[v]) == 0]
    
    children_dict = defaultdict(list)
    for child, parents in parents_dict.items():
        for p in parents:
            children_dict[p].append(child)
            
    in_degree = {v: len(parents_dict[v]) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_dict[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    fast = set()
    slow = set(inputs)
    ops = []
    deadline = time.time() + 55
    
    def evict_one():
        if not fast:
            return
        min_imp = float('inf')
        victim = None
        for u in fast:
            if u in outputs:
                continue
            imp = len(children_dict[u])
            if imp < min_imp:
                min_imp = imp
                victim = u
        if victim is None:
            victim = next(iter(fast))
        fast.remove(victim)
        if victim in slow:
            ops.append(['delete', victim])
        else:
            ops.append(['store', victim])
            slow.add(victim)
            
    def ensure_in_fast(v):
        if v in fast:
            return
        if time.time() > deadline:
            return
        if v in slow:
            if len(fast) < fast_memory_budget:
                ops.append(['load', v])
                fast.add(v)
            else:
                evict_one()
                ops.append(['load', v])
                fast.add(v)
            return
        parents = parents_dict[v]
        for p in parents:
            ensure_in_fast(p)
        if len(fast) < fast_memory_budget:
            ops.append(['compute', v])
            fast.add(v)
        else:
            evict_one()
            ops.append(['compute', v])
            fast.add(v)
        slow.add(v)
        
    for v in topo_order:
        if time.time() > deadline:
            break
        ensure_in_fast(v)
        
    for o in outputs:
        if o not in fast:
            if time.time() > deadline:
                break
            ensure_in_fast(o)
            
    return ops
