def construct():
    import time
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    n = 256
    # 1. Precompute conflicts
    conflicts = [set() for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            # Check if a, b form part of a 3-AP
            # Cases:
            # 1. a, b, c -> c = 2b - a. Valid if c <= n and c != a, c != b.
            # 2. c, a, b -> c = 2a - b. Valid if c >= 1 and c != a, c != b.
            # 3. a, c, b -> c = (a+b)/2. Valid if (a+b) % 2 == 0 and 1 <= c <= n and c != a, c != b.
            
            # Actually, it's easier to just check if any 3-AP contains both a and b.
            # A 3-AP is defined by start s and diff d: s, s+d, s+2d.
            # a and b are in it if:
            # {a,b} is a subset of {s, s+d, s+2d}.
            
            # Let's just iterate through all possible 3-APs and add edges.
            pass
            
    # More efficient conflict generation:
    # Iterate through all 3-APs (a, b, c) with a < b < c.
    # a + (c-a) = 2b -> c = 2b - a.
    # For each pair (a, c) with same parity, b = (a+c)/2.
    # If 1 <= a < b < c <= n, then a, b, c form a 3-AP.
    # Edges: (a,b), (b,c), (a,c).
    
    conflicts = [set() for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            # Check if a, b are in a 3-AP
            # 1. a, b, c: c = 2b - a. If c <= n and c > b:
            c = 2*b - a
            if c <= n and c > b:
                conflicts[a].add(b)
                conflicts[a].add(c)
                conflicts[b].add(a)
                conflicts[b].add(c)
                conflicts[c].add(a)
                conflicts[c].add(b)
            # 2. c, a, b: c = 2a - b. If c >= 1 and c < a:
            c = 2*a - b
            if c >= 1 and c < a:
                conflicts[a].add(b)
                conflicts[a].add(c)
                conflicts[b].add(a)
                conflicts[b].add(c)
                conflicts[c].add(a)
                conflicts[c].add(b)
            # 3. a, c, b: c = (a+b)/2. If (a+b) % 2 == 0 and a < c < b:
            if (a+b) % 2 == 0:
                c = (a+b)//2
                if a < c < b:
                    conflicts[a].add(b)
                    conflicts[a].add(c)
                    conflicts[b].add(a)
                    conflicts[b].add(c)
                    conflicts[c].add(a)
                    conflicts[c].add(b)
    
    # Weights
    weights = [0.0] + [1.0/i for i in range(1, n+1)]
    
    # Sort nodes by weight descending
    nodes_sorted = sorted(range(1, n+1), key=lambda x: -weights[x])
    
    best_weight = 0.0
    best_set = set()
    
    # B&B
    # candidates: set of available nodes
    # current_set: set of selected nodes
    # current_weight: sum of weights
    
    def search(candidates, current_set, current_weight):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
            
        # Upper bound
        # Sum of weights of all candidates
        # To speed up, we can precompute prefix sums of sorted weights?
        # But candidates is a subset.
        # Simple bound: current_weight + sum(w[x] for x in candidates)
        # This is O(N) per call. Might be slow.
        # Better bound: current_weight + sum of top k weights where k is max possible independent set size? No.
        # Let's stick to the simple bound first. If too slow, optimize.
        
        ub = current_weight + sum(weights[x] for x in candidates)
        if ub <= best_weight:
            return
            
        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
            
        # Pick a node to branch on.
        # Heuristic: Pick the node with highest weight in candidates.
        # Since nodes_sorted is global, find the first one in candidates.
        x = None
        for node in nodes_sorted:
            if node in candidates:
                x = node
                break
                
        if x is None:
            return
            
        # Branch 1: Include x
        # New candidates = candidates - {x} - neighbors(x)
        new_candidates = candidates - {x}
        for neighbor in conflicts[x]:
            new_candidates.discard(neighbor)
            
        new_set = current_set | {x}
        search(new_candidates, new_set, current_weight + weights[x])
        
        # Branch 2: Exclude x
        new_candidates = candidates - {x}
        search(new_candidates, current_set, current_weight)

    initial_candidates = set(range(1, n+1))
    search(initial_candidates, set(), 0.0)
    
    return list(best_set)
