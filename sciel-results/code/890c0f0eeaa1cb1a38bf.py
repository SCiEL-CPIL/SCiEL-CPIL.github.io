<<<<<<< SEARCH
    # Parse the DAG
    nodes = list(dag['nodes'])
    parents = dict(dag['parents'])
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Build children map
    children = defaultdict(list)
    for node in nodes:
        for p in parents.get(node, []):
            children[p].append(node)
    
    # Topological order is given by nodes list
    # Determine which nodes are inputs (start blue), which are computed
    # A node is an input if it has no parents or is explicitly in inputs
    # Let's assume inputs are nodes with no parents, or explicitly listed
    
    # Identify input nodes: those with no parents
    input_nodes = set()
    for node in nodes:
        if not parents.get(node, []):
            input_nodes.add(node)
    # Also add explicitly listed inputs
    input_nodes |= inputs
    
    # Output nodes must finish blue
    output_nodes = outputs if outputs else input_nodes  # fallback
=======
    # Parse the DAG robustly
    parents = dict(dag.get('parents', {}))
    
    # Infer nodes if not explicitly provided
    if 'nodes' in dag:
        nodes = list(dag['nodes'])
    else:
        # Nodes are all keys in parents and all values in parents (to catch leaves)
        node_set = set(parents.keys())
        for plist in parents.values():
            node_set.update(plist)
        # Sort to have a deterministic order if not topologically sorted in keys
        # However, the problem says "topologically ordered parents", implying the input structure respects this.
        # If we infer, we might need to topologically sort. 
        # Let's assume if 'nodes' is missing, the keys of 'parents' are the nodes, but we must include children.
        # Actually, standard DAG representation usually lists all nodes in 'nodes'. 
        # If 'nodes' is missing, we assume the graph is defined by 'parents'.
        # We need a topological order. Let's perform a simple topo sort on the inferred graph.
        import collections
        in_degree = {n: 0 for n in node_set}
        children_map = defaultdict(list)
        for node, plist in parents.items():
            for p in plist:
                children_map[p].append(node)
                in_degree[node] += 1
        
        queue = collections.deque([n for n in node_set if in_degree[n] == 0])
        nodes = []
        while queue:
            u = queue.popleft()
            nodes.append(u)
            for v in children_map[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # If topo sort didn't cover all nodes (cycle?), append remaining
        if len(nodes) < len(node_set):
            remaining = [n for n in node_set if n not in set(nodes)]
            nodes.extend(remaining)

    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Build children map (re-use if already built during topo sort, else build)
    if 'nodes' in dag:
        children = defaultdict(list)
        for node in nodes:
            for p in parents.get(node, []):
                children[p].append(node)
    else:
        children = children_map # Reuse from topo sort above

    # Topological order is given by nodes list (now guaranteed)
    
    # Identify input nodes: those with no parents AND/OR explicitly listed
    input_nodes = set()
    for node in nodes:
        if not parents.get(node, []):
            input_nodes.add(node)
    # Explicitly listed inputs are also inputs (even if they have parents? Unlikely, but safe to union)
    input_nodes |= inputs
    
    # Output nodes must finish blue
    # If outputs are not specified, assume all leaves (nodes with no children) are outputs?
    # Or just the explicitly specified ones. The contract says "output vertices" are part of dag.
    # If 'outputs' is empty, we should probably ensure all leaves are blue or just return empty ops if no outputs?
    # Let's default to all leaves if outputs is not specified, to be safe.
    if not outputs:
        output_nodes = set(node for node in nodes if not children.get(node, []))
    else:
        output_nodes = outputs
>>>>>>> REPLACE

