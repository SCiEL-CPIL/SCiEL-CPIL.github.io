import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d  # 128
    n_edges = n_vertices * d // 2  # 448
    
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute all edges
    edges = []
    edge_map = {}  # (u, i) -> idx where u < u^(1<<i)
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, i))
                edge_map[(u, i)] = idx
                idx += 1
    
    # Precompute conflicts: for each edge e=(u,i), which other edges conflict with it?
    # Edge (u,i) conflicts with (w,j) if they are in the same 2-face and form a C4
    # Specifically, for edge (u, u^(1<<i)), it's in a face spanned by dims i and j (j!=i)
    # The face vertices are: u, u^(1<<i), u^(1<<j), u^(1<<i)^(1<<j)
    # The four edges of this face are:
    #   (u, i), (u^(1<<j), i)  -- dim i edges
    #   (u, j), (u^(1<<i), j)  -- dim j edges
    # So edge (u,i) conflicts with:
    #   (u^(1<<j), i) and (u, j) and (u^(1<<i), j) for each j != i
    # But we need to be careful: a C4 is formed when ALL FOUR edges are present.
    # So edge (u,i) "conflicts" with edge (w,j) if adding both could complete a C4.
    # More precisely: if we have edge (u,i) and want to add edge (w,j), we need to check
    # if there exists a face containing both edges where the other two edges are already present.
    
    # Alternative approach: for each pair of edges, determine if they are in a common face.
    # Two edges (u,i) and (w,j) are in a common face if:
    #   - i != j (different dimensions)
    #   - The four vertices u, u^(1<<i), w, w^(1<<j) form a 2-face
    #   - This happens if w = u^(1<<j) or w = u^(1<<i)^(1<<j) or w = u (but w!=u since different edges)
    
    # Let me think differently. For edge (u,i), the faces it belongs to are:
    # For each j != i, the face with corners u, u^(1<<i), u^(1<<j), u^(1<<i)^(1<<j)
    # In this face, the other three edges are:
    #   (u^(1<<j), i), (u, j), (u^(1<<i), j)
    # So if any of these three edges is also present, and then we add the fourth, we get a C4.
    # But actually, we get a C4 only when ALL FOUR are present.
    # So the "conflict" is: edge (u,i) cannot be added if for some j != i,
    #   all of (u^(1<<j), i), (u, j), (u^(1<<i), j) are already present.
    
    # This is what the current code does with can_add(). Let me keep that logic but optimize.
    
    # State: edge_present as a 1D boolean array of length n_edges
    # But for fast lookup, use a 2D array: ep[u][i] = True if edge (u, u^(1<<i)) is present
    # Note: ep[u][i] == ep[u^(1<<i)][i]
    
    ep = np.zeros((n_vertices, d), dtype=bool)
    
    def can_add(u, i):
        v = u ^ (1 << i)
        for j in range(d):
            if j == i:
                continue
            if ep[u][j] and ep[v][j] and ep[u ^ (1 << j)][i]:
                return False
        return True
    
    def add_edge(u, i):
        v = u ^ (1 << i)
        ep[u][i] = True
        ep[v][i] = True
    
    def remove_edge(u, i):
        v = u ^ (1 << i)
        ep[u][i] = False
        ep[v][i] = False
    
    def count_edges():
        return int(np.sum(ep)) // 2
    
    # Phase 1: Multiple greedy passes with different random orders
    best_count = 0
    best_state = None
    
    def greedy_random_order():
        nonlocal ep
        ep[:] = False
        order = list(range(n_edges))
        random.shuffle(order)
        for idx in order:
            u, i = edges[idx]
            if can_add(u, i):
                add_edge(u, i)
        return count_edges()
    
    num_greedy = 10
    for _ in range(num_greedy):
        if time.time() - start_time > time_limit - 10:
            break
        c = greedy_random_order()
        if c > best_count:
            best_count = c
            best_state = ep.copy()
    
    if best_state is not None:
        ep[:] = best_state
    
    # Phase 2: Local search with proper state management
    # We'll use a different approach: repeatedly try to swap edges
    # Remove a random edge, then greedily refill, see if we improve
    
    current_count = best_count
    best_overall = current_count
    best_overall_state = ep.copy()
    
    # Precompute a list of all edge indices for random selection
    all_edge_indices = list(range(n_edges))
    
    iteration = 0
    temperature = 1.0
    cooling = 0.999
    t_min = 0.01
    
    while time.time() - start_time < time_limit - 3:
        iteration += 1
        
        # Save current state
        saved_state = ep.copy()
        saved_count = current_count
        
        # Remove 1-2 random edges
        num_remove = random.choice([1, 1, 1, 2])
        removed = []
        attempts = 0
        while len(removed) < num_remove and attempts < 30:
            attempts += 1
            idx = random.randint(0, n_edges - 1)
            u, i = edges[idx]
            if ep[u][i]:
                remove_edge(u, i)
                removed.append(idx)
        
        if not removed:
            continue
        
        # Greedily refill: try to add back removed edges and other random edges
        for idx in removed:
            u, i = edges[idx]
            if can_add(u, i):
                add_edge(u, i)
        
        # Try to add more random edges
        num_fill = random.randint(5, 20)
        for _ in range(num_fill):
            if time.time() - start_time > time_limit - 2:
                break
            idx = random.randint(0, n_edges - 1)
            u, i = edges[idx]
            if not ep[u][i]:
                if can_add(u, i):
                    add_edge(u, i)
        
        new_count = count_edges()
        
        # Accept or reject
        delta = new_count - saved_count
        if delta >= 0 or random.random() < np.exp(delta / temperature):
            current_count = new_count
            if current_count > best_overall:
                best_overall = current_count
                best_overall_state = ep.copy()
        else:
            # Revert
            ep[:] = saved_state
        
        temperature = max(t_min, temperature * cooling)
        
        if best_overall >= 308:
            break
    
    # Restore best state
    ep[:] = best_overall_state
    
    # Convert to output format
    out = []
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v and ep[u][i]:
                out.append([u, v])
    
    return out
