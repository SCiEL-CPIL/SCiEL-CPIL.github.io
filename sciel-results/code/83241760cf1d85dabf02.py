import time

def construct():
    N = 180
    # Precompute all forbidden triples
    triples = []
    for k in range(1, N + 1):
        for m in range(1, N + 1):
            for n in range(m + 1, N + 1):
                if (m + n) * k > N: 
                    break
                a = k * m * n
                b = k * m * (m + n)
                c = k * n * (m + n)
                if a <= N and b <= N and c <= N:
                    # Check distinct
                    if a != b and a != c and b != c:
                        triples.append((a, b, c))
    
    # Initial set
    S = set(range(1, N + 1))
    
    # Map node to list of indices of triples containing it
    node_to_triples = {i: [] for i in range(1, N + 1)}
    for idx, t in enumerate(triples):
        for x in t:
            node_to_triples[x].append(idx)
    
    # Active triples
    active_triples = set(range(len(triples)))
    
    # Degrees
    degrees = {i: len(node_to_triples[i]) for i in range(1, N + 1)}
    
    # Greedy removal
    # We want to remove nodes that are in many active triples.
    # To be efficient, we can just scan for a triple and remove the max degree node.
    
    # Optimization: 
    # If we remove a node, we must remove all triples containing it from active_triples.
    
    while True:
        if not active_triples:
            break
        
        # Pick a random active triple to break ties or just the first
        # To be deterministic and potentially better, pick the triple with the highest sum of degrees?
        # Or just the first one found.
        # Let's pick the triple containing the node with the highest degree.
        
        max_deg_node = -1
        max_deg = -1
        for idx in active_triples:
            a, b, c = triples[idx]
            # Check if still active (nodes in S)
            # We maintain S, so we can check membership.
            # But active_triples should only contain triples fully in S.
            # So we just need to check degrees.
            for x in (a, b, c):
                if degrees[x] > max_deg:
                    max_deg = degrees[x]
                    max_deg_node = x
        
        if max_deg_node == -1:
            break
            
        # Remove max_deg_node
        S.remove(max_deg_node)
        degrees[max_deg_node] = 0 # Mark as removed
        
        # Update active_triples
        # Remove all triples containing max_deg_node
        to_remove = set()
        for idx in node_to_triples[max_deg_node]:
            if idx in active_triples:
                to_remove.add(idx)
        
        for idx in to_remove:
            active_triples.remove(idx)
            # Update degrees of other nodes in this triple?
            # No, degree is static count of total triples.
            # But for "highest degree in active triples", we should use active degree.
            
    # The above uses static degree. It might be better to use active degree.
    # Let's refine: use active degree.
    
    return list(S)
