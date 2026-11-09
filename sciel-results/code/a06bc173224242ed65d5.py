import random
import time

def construct():
    d = 7
    n = 2 ** d
    total_edges = d * (n // 2)
    
    # Precompute all edges as (u, v, dim) where v = u ^ (1 << dim)
    # u < v to ensure unique representation
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    
    # Helper to check if adding edge (u, v, i) creates a 4-cycle
    # A 4-cycle is formed if for some j != i, the edges (u, u^2^j), (v, v^2^j), (u^2^j, v^2^j) are all selected
    def is_safe(u, v, i, selected):
        for j in range(d):
            if j == i:
                continue
            mask_j = 1 << j
            u_j = u ^ mask_j
            v_j = v ^ mask_j
            
            # The three other edges in the square are:
            # 1. (u, u_j)
            # 2. (v, v_j)
            # 3. (u_j, v_j)
            
            # Normalize edges to (min, max) for set lookup
            e1 = (min(u, u_j), max(u, u_j))
            e2 = (min(v, v_j), max(v, v_j))
            e3 = (min(u_j, v_j), max(u_j, v_j))
            
            if e1 in selected and e2 in selected and e3 in selected:
                return False
        return True

    best_count = 0
    best_edges = []
    
    start_time = time.time()
    max_time = 110.0  # Leave 10s buffer
    
    # Run multiple random trials
    # Each trial is O(E * d) which is fast.
    # 448 edges * 6 checks * 3 set lookups ~ 8000 ops per trial.
    # We can run thousands of trials.
    
    trials = 0
    while time.time() - start_time < max_time:
        random.shuffle(all_edges)
        selected = set()
        current_edges = []
        
        for u, v, i in all_edges:
            if is_safe(u, v, i, selected):
                selected.add((u, v))
                current_edges.append([u, v])
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
        
        trials += 1
        
        # If we hit a very high score, we might want to break early or continue to try for perfection
        # 308 is the upper bound. If we hit 308, we can stop.
        if best_count >= 308:
            break

    return best_edges
