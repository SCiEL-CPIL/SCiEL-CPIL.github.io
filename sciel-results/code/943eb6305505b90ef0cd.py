import random

def _generate_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def _solve_csp(tree, k, n, used_edges):
    """
    Solve the embedding of a single tree using CSP with MRV and Forward Checking.
    Returns a list of host vertices for the tree, or None.
    """
    if k == 1:
        return [0]
    if k == 2:
        for a in range(n):
            for b in range(a + 1, n):
                if (a, b) not in used_edges:
                    return [a, b]
        return None

    # Build tree adjacency
    adj = [[] for _ in range(k)]
    for u, v in tree:
        adj[u].append(v)
        adj[v].append(u)
    
    # Domains: list of sets of possible host vertices
    domains = [set(range(n)) for _ in range(k)]
    
    def propagate(var, val, domains_copy):
        """
        Assign domains_copy[var] = {val} and propagate constraints.
        Returns False if a domain becomes empty.
        """
        # 1. All-Different constraint: remove val from all other domains
        for i in range(k):
            if i == var:
                continue
            if val in domains_copy[i]:
                domains_copy[i].remove(val)
                if not domains_copy[i]:
                    return False
        
        # 2. Edge constraints: for each neighbor, remove values that conflict
        for neighbor in adj[var]:
            for h in list(domains_copy[neighbor]):
                edge = (min(val, h), max(val, h))
                if edge in used_edges:
                    domains_copy[neighbor].remove(h)
                    if not domains_copy[neighbor]:
                        return False
        
        return True

    def backtrack(domains):
        # Check if all assigned
        unassigned = [i for i in range(k) if len(domains[i]) > 1]
        if not unassigned:
            # All domains have size 1
            return [list(domains[i])[0] for i in range(k)]
        
        # MRV: pick unassigned variable with smallest domain
        var = min(unassigned, key=lambda i: len(domains[i]))
        
        # Try each value in the domain
        for val in domains[var]:
            # Copy domains for propagation
            new_domains = [set(d) for d in domains]
            new_domains[var] = {val}
            
            if propagate(var, val, new_domains):
                result = backtrack(new_domains)
                if result is not None:
                    return result
        return None

    return backtrack(domains)

def construct():
    n = 20
    trees = _generate_trees(n)
    
    result = [None] * n
    used_edges = set()
    
    # We will try multiple random orders of embedding the trees.
    # T1 is always trivial. We embed T2..T20.
    tree_indices = list(range(1, n))  # 1 to 19
    
    # Try up to 100 random permutations
    rng = random.Random(42)
    best_result = [None] * n
    best_score = 0
    
    for attempt in range(100):
        # Shuffle order
        current_order = tree_indices[:]
        rng.shuffle(current_order)
        
        # Try to embed in this order
        temp_result = [None] * n
        temp_used_edges = set()
        success = True
        
        # T1 is always None (or [0])
        temp_result[0] = None
        
        for idx in current_order:
            k = idx + 1
            tree = trees[idx]
            
            embedding = _solve_csp(tree, k, n, temp_used_edges)
            
            if embedding is None:
                success = False
                break
            
            temp_result[idx] = embedding
            for u, v in tree:
                hu = embedding[u]
                hv = embedding[v]
                edge = (min(hu, hv), max(hu, hv))
                temp_used_edges.add(edge)
        
        if success:
            # We found a complete packing!
            return temp_result
        
        # Track best partial solution
        current_score = sum(len(trees[i]) for i in range(n) if temp_result[i] is not None)
        if current_score > best_score:
            best_score = current_score
            best_result = temp_result[:]
    
    return best_result
