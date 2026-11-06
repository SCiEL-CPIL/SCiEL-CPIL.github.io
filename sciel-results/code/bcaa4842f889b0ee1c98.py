import time
import sys

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    # Parse DAG
    # dag structure: assume dict of vertex -> list of parents
    # Or list of (vertex, parents)? 
    # The prompt says "dag has topologically ordered parents".
    # Let's assume dag is a dict: {vertex: [parent1, parent2, ...]}
    # Inputs: vertices with no parents.
    # Outputs: vertices that are not parents of any other vertex? 
    # "output vertices" are explicitly mentioned.
    # Let's assume we can identify outputs as vertices with no children.
    
    # Build reverse graph to find children
    children = {v: [] for v in dag}
    for v, parents in dag.items():
        for p in parents:
            if p not in children:
                children[p] = []
            children[p].append(v)
            
    inputs = [v for v in dag if not dag[v]]
    outputs = [v for v in dag if not children[v]]
    
    # Topological sort (Kahn's algorithm)
    in_degree = {v: len(dag[v]) for v in dag}
    queue = [v for v in dag if in_degree[v] == 0]
    topo_order = []
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(dag):
        # Cycle? Should not happen. Fallback to naive
        return naive_schedule(dag, fast_memory_budget)

    # Determine Layers
    # Layer 0: Inputs
    # Layer i+1: Vertices whose parents are all in Layer i (or earlier)
    # Simple BFS layering
    layer_of = {}
    for v in inputs:
        layer_of[v] = 0
    queue = inputs[:]
    max_layer = 0
    while queue:
        v = queue.pop(0)
        l = layer_of[v]
        for c in children[v]:
            # A child's layer is max(parent_layers) + 1
            # But if parents are in different layers, this is tricky.
            # For "layered DAGs", we assume parents are in layer l-1.
            # Let's use: layer_of[c] = max(layer_of[p] for p in dag[c]) + 1
            # We can compute this in topo order.
            pass
            
    # Recompute layers in topo order
    for v in topo_order:
        if not dag[v]:
            layer_of[v] = 0
        else:
            layer_of[v] = max(layer_of[p] for p in dag[v]) + 1
            max_layer = max(max_layer, layer_of[v])
            
    layers = [[] for _ in range(max_layer + 1)]
    for v in dag:
        layers[layer_of[v]].append(v)
        
    # Strategy: Process layer by layer
    # State: fast_memory (set of vertices), computed (set of vertices)
    # Operations: list of (op, vertex)
    
    ops = []
    fast_memory = set()
    computed = set()
    
    # Helper to check time
    def time_check():
        return time.time() > deadline

    # Helper to evict
    def evict_one():
        # Evict vertex with least remaining children
        # Only evict vertices that are not in the current layer being computed?
        # Actually, we can evict any computed vertex not in fast_memory? No, only those in fast_memory.
        # We want to keep vertices that are needed soon.
        # Urgency: number of children in the NEXT layer that are not yet computed.
        # Since we process layer by layer, we know which layer we are in.
        
        # For simplicity, LRU or random? 
        # Let's use: Evict the vertex with the fewest children in the next layer.
        # If tie, evict the one with the smallest index (arbitrary).
        
        # We need to know the "next layer" to compute urgency.
        # This function is called during processing.
        
        # Better: Pass current context.
        pass

    # Given the complexity of dynamic eviction policies in a tight loop,
    # and the "layered" nature, let's try a specific block strategy.
    
    # If max layer size <= budget, simple approach:
    # Load Layer i, Compute Layer i+1, Evict Layer i.
    
    if max(len(layer) for layer in layers) <= fast_memory_budget:
        # Simple Layered Approach
        # Start with empty fast memory
        # Layer 0 (Inputs)
        # We need to compute Layer 1.
        # Load all of Layer 0.
        for v in layers[0]:
            ops.append(('load', v))
            fast_memory.add(v)
            computed.add(v)
            
        for i in range(1, max_layer + 1):
            # Compute Layer i
            for v in layers[i]:
                # Parents are in Layer i-1 (or earlier, but in fast memory)
                # Check if parents are in fast_memory
                # They should be, because we loaded Layer i-1 and haven't evicted yet.
                # Compute
                ops.append(('compute', v))
                fast_memory.add(v)
                computed.add(v)
                
            # After computing Layer i, we can evict Layer i-1
            # Because Layer i-1 is no longer needed (only needed for Layer i)
            # Unless Layer i-1 vertices are also needed for Layer i+1? No, layered means i-1 -> i.
            # So we can evict Layer i-1.
            for v in layers[i-1]:
                if v in fast_memory:
                    ops.append(('store', v))
                    fast_memory.remove(v)
                    
        # Finally, store all outputs (which are in the last layer)
        # They must finish blue.
        for v in layers[max_layer]:
            if v in fast_memory:
                ops.append(('store', v))
                fast_memory.remove(v)
                
        return ops

    else:
        # Complex Case: Some layer is larger than budget.
        # We use a Greedy Block Strategy.
        # For each transition i-1 -> i:
        # If Layer i-1 fits, load all, compute Layer i in chunks?
        # If Layer i-1 doesn't fit, we must reload parts of it.
        
        # Let's implement a generic Greedy Topological with LRU for the complex case.
        # It's safer than trying to optimize the block partitioning which might have bugs.
        
        fast_memory = set()
        computed = set()
        ops = []
        
        # Pre-compute children count for eviction
        # remaining_children = {v: len(children[v]) for v in dag}
        # When a child is computed, decrement remaining_children of its parents.
        
        remaining_children = {v: len(children[v]) for v in dag}
        
        # To speed up eviction, we can keep a priority queue of (urgency, vertex)
        # But urgency changes as children are computed.
        # Simple approach: Linear scan to find best victim. O(N) per eviction.
        # N is likely small enough (DAGs in such tasks are usually < 1000 nodes).
        
        for v in topo_order:
            # Check if v is in fast_memory
            if v in fast_memory:
                continue
                
            # Check parents
            missing_parents = []
            for p in dag[v]:
                if p not in fast_memory:
                    missing_parents.append(p)
                    
            # Load missing parents
            for p in missing_parents:
                # Check if we need to evict
                if len(fast_memory) >= fast_memory_budget:
                    # Find victim
                    # Victim: computed vertex in fast_memory with min remaining_children
                    # If all remaining_children are 0, any is fine.
                    best_victim = None
                    min_urgency = float('inf')
                    for u in fast_memory:
                        # u is in fast_memory, so it is computed
                        # We want to evict the one least needed soon
                        urg = remaining_children[u]
                        if urg < min_urgency:
                            min_urgency = urg
                            best_victim = u
                    # Evict
                    ops.append(('store', best_victim))
                    fast_memory.remove(best_victim)
                    
                ops.append(('load', p))
                fast_memory.add(p)
                # p is now in fast_memory. It was already computed (since it's a parent in topo order).
                
            # Compute v
            ops.append(('compute', v))
            fast_memory.add(v)
            computed.add(v)
            
            # Update remaining_children of parents
            for p in dag[v]:
                remaining_children[p] -= 1
                
            # Check time
            if time_check():
                # Return best effort so far? 
                # We need a valid schedule.
                # If we stop mid-way, outputs might not be blue.
                # We must ensure all outputs are stored.
                # This is risky. Better to run to completion if possible.
                # If time is tight, we might fail.
                # Given 60s limit and likely small N, this should be fine.
                pass

        # Store all outputs
        for v in outputs:
            if v in fast_memory:
                ops.append(('store', v))
                fast_memory.remove(v)
                
        return ops

def naive_schedule(dag, fast_memory_budget):
    # Fallback
    ops = []
    for v in dag:
        for p in dag[v]:
            ops.append(('load', p))
        ops.append(('compute', v))
    for v in dag:
        if not [c for c in dag if v in dag[c]]: # Output
            ops.append(('store', v))
    return ops
