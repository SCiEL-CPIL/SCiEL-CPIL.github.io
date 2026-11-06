def pebble(dag, fast_memory_budget):
    """
    Minimizes transfers by keeping live values in fast memory.
    """
    # Parse the DAG structure
    # dag is likely a dict or list of nodes with parents/children
    # Based on "dag has topologically ordered parents and output vertices"
    # Let's assume dag is a list of dicts or a specific structure.
    # The baseline function `pebble_baseline` exists in `autoresearch_suite.science_tcs_expansion`.
    # I don't have access to that module's source, but I can infer the structure from typical tasks.
    # Usually, dag is a list of nodes, each with 'id', 'parents', 'children'.
    # Or a dict {node_id: {'parents': [...], 'children': [...]}}.
    
    # Since I cannot import the baseline to inspect, I will write a generic parser 
    # that handles common structures, or rely on the fact that the input is standard.
    # The prompt says "dag has topologically ordered parents".
    # This might mean the nodes are already sorted, and parents are listed with indices.
    
    # Let's assume the standard format used in such suites:
    # dag: list of nodes. Each node is a dict with 'id' or index, 'parents' (list of indices/ids), 'children'.
    # Or simply a list of parents for each node in topological order.
    # "topologically ordered parents" -> Node i's parents are among 0..i-1.
    
    # Let's assume:
    # dag is a list of lists of parent indices.
    # dag[i] is the list of parents of node i.
    # Outputs are nodes with no children.
    
    # I need to be careful. If the format is different, this will crash.
    # However, the baseline `pebble_baseline` works, so the format is consistent.
    # Common format in these generated tasks:
    # dag: List[List[int]] where dag[i] is parents of node i.
    # fast_memory_budget: int.
    
    # Let's try to detect the structure.
    # If dag is a list of lists:
    if isinstance(dag, list) and (not dag or isinstance(dag[0], list)):
        parents_list = dag
        n = len(dag)
        children_list = [[] for _ in range(n)]
        for i in range(n):
            for p in parents_list[i]:
                children_list[p].append(i)
    elif isinstance(dag, dict):
        # Handle dict case if necessary
        pass
    else:
        # Fallback
        return []

    # Identify inputs and outputs
    inputs = [i for i in range(n) if not parents_list[i]]
    outputs = [i for i in range(n) if not children_list[i]]
    
    # Topological order is 0..n-1 based on "topologically ordered parents"
    
    # State
    in_memory = set(inputs)
    remaining_uses = {i: len(children_list[i]) for i in range(n)}
    
    # If inputs exceed budget, we have a problem.
    # But let's assume valid budget.
    
    operations = []
    
    # Process nodes in topological order
    for v in range(n):
        if v in inputs:
            # Inputs are already in memory
            continue
            
        # 1. Ensure parents are in memory
        for p in parents_list[v]:
            if p not in in_memory:
                operations.append(['load', p])
                in_memory.add(p)
                
        # 2. Compute v
        operations.append(['compute', v])
        in_memory.add(v)
        
        # 3. Update remaining uses for parents
        for p in parents_list[v]:
            remaining_uses[p] -= 1
            
        # 4. Memory management
        if len(in_memory) > fast_memory_budget:
            # Find candidates to evict
            # Priority 1: Dead nodes (remaining_uses == 0)
            # Priority 2: Live nodes (remaining_uses > 0)
            
            dead_candidates = [u for u in in_memory if remaining_uses[u] == 0]
            if dead_candidates:
                # Evict one dead candidate
                # Pick the one with smallest index (FIFO) to keep newer ones?
                # Or just any.
                u = min(dead_candidates)
            else:
                # No dead candidates. Must evict a live node.
                # This will cause a reload later.
                # Evict the one with smallest index (FIFO) to keep the most recent?
                # Or evict the one with largest index (LRU)?
                # In DAGs, dependencies are forward. Older nodes are less likely to be needed by *immediate* next nodes?
                # Actually, if a node is old, its children are also old (already processed).
                # So remaining_uses for an old node is likely 0.
                # If remaining_uses > 0, it means it has children that are *not yet processed*.
                # Since we process in topological order, if a node has remaining uses, its children are in the future.
                # So we should keep nodes that are needed soon.
                # Evict the node with the *largest* topological index? No, that's the most recent.
                # Evict the node with the *smallest* topological index? (Oldest)
                # Let's stick to FIFO (smallest index) for eviction among live nodes as well.
                live_candidates = [u for u in in_memory if remaining_uses[u] > 0]
                if live_candidates:
                    u = min(live_candidates)
                else:
                    # Should not happen if len(in_memory) > budget and budget >= 1
                    u = min(in_memory)
                    
            operations.append(['store', u])
            in_memory.remove(u)
            
    # 5. Ensure outputs are in memory
    for o in outputs:
        if o not in in_memory:
            operations.append(['load', o])
            in_memory.add(o)
            
    return operations
