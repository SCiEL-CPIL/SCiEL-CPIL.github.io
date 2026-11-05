import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges_total = len(all_edges)
    
    best_edges = []
    best_score = 0
    best_num_edges = 0
    best_violations = float('inf')

    # Helper to get edge list from adj
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= lsb
        return edges

    # Run Simulated Annealing
    def run_sa(seed, T_start, T_end, cooling, max_steps):
        nonlocal best_edges, best_score, best_num_edges, best_violations
        
        random.seed(seed)
        
        # Initialize
        adj = [0] * n
        C = [[0] * n for _ in range(n)]
        violations = 0
        num_edges = 0
        
        # Random initial state
        # p=0.3 to start dense-ish
        for u, v in all_edges:
            if random.random() < 0.3:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                num_edges += 1
                # Update C
                # For w in N(v): C[u][w]++, C[w][u]++
                mask_v = adj[v] & ~(1 << u) # exclude u
                while mask_v:
                    lsb = mask_v & -mask_v
                    w = lsb.bit_length() - 1
                    old_cu_w = C[u][w]
                    C[u][w] += 1
                    C[w][u] += 1
                    new_cu_w = C[u][w]
                    if new_cu_w >= 3 and old_cu_w < 3:
                        violations += 1
                    mask_v -= lsb
                    
                mask_u = adj[u] & ~(1 << v)
                while mask_u:
                    lsb = mask_u & -mask_u
                    w = lsb.bit_length() - 1
                    old_cv_w = C[v][w]
                    C[v][w] += 1
                    C[w][v] += 1
                    new_cv_w = C[v][w]
                    if new_cv_w >= 3 and old_cv_w < 3:
                        violations += 1
                    mask_u -= lsb

        current_violations = violations
        current_num_edges = num_edges
        
        if current_violations == 0 and current_num_edges > best_num_edges:
            best_num_edges = current_num_edges
            best_edges = get_edges_from_adj(adj)
        
        T = T_start
        steps = 0
        
        while T > T_end and steps < max_steps:
            if time.time() > deadline - 5.0:
                break
                
            # Pick random edge
            u, v = random.choice(all_edges)
            
            # Determine if adding or removing
            is_present = bool(adj[u] & (1 << v))
            
            # Calculate delta energy
            # Energy = violations * 100 - num_edges
            # We want to minimize energy.
            
            if is_present:
                # Remove edge
                # Check if removal improves violations
                # C[u][w]-- for w in N(v) \ {u}
                # C[v][w]-- for w in N(u) \ {v}
                
                # We can just compute new violations
                # But incremental is faster.
                
                delta_violations = 0
                # We need to know which pairs cross the threshold.
                
                # List of (x,y) to decrement
                to_dec = []
                mask_v = adj[v] & ~(1 << u)
                while mask_v:
                    lsb = mask_v & -mask_v
                    w = lsb.bit_length() - 1
                    to_dec.append((u, w))
                    to_dec.append((w, u))
                    mask_v -= lsb
                    
                mask_u = adj[u] & ~(1 << v)
                while mask_u:
                    lsb = mask_u & -mask_u
                    w = lsb.bit_length() - 1
                    to_dec.append((v, w))
                    to_dec.append((w, v))
                    mask_u -= lsb
                
                # Apply decrements to a copy? No, just check.
                # Since we might revert, we need to be careful.
                # Let's just compute the change.
                
                # Actually, let's just do the update and revert if needed.
                
                # Save old state of C for affected pairs?
                # Or just calculate delta.
                
                # For each (x,y) in to_dec:
                # if C[x][y] == 3, then violation decreases by 1.
                # if C[x][y] == 2, violation stays 0.
                
                dec_count = 0
                for x, y in to_dec:
                    if C[x][y] == 3:
                        dec_count += 1
                    # Note: (x,y) and (y,x) are same pair.
                    # My to_dec has duplicates?
                    # mask_v gives w. (u,w) and (w,u).
                    # mask_u gives w. (v,w) and (w,v).
                    # These are distinct pairs unless u==v (impossible) or w is shared?
                    # If w is in N(u) and N(v), then (u,w) and (v,w) are distinct.
                    # But (u,w) appears once from mask_v loop.
                    # (w,u) appears once from mask_v loop.
                    # So we count each directed pair.
                    # Violation is undirected.
                    # So we should only count if x < y.
                    
                # Let's refine the delta calculation.
                # It's safer to just apply, check, and revert.
                
                # Apply removal
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                num_edges -= 1
                
                mask_v = adj[v]
                while mask_v:
                    lsb = mask_v & -mask_v
                    w = lsb.bit_length() - 1
                    if w == u: 
                        mask_v -= lsb
                        continue
                    old_cu_w = C[u][w]
                    C[u][w] -= 1
                    C[w][u] -= 1
                    new_cu_w = C[u][w]
                    if old_cu_w >= 3 and new_cu_w < 3:
                        violations -= 1
                    mask_v -= lsb
                    
                mask_u = adj[u]
                while mask_u:
                    lsb = mask_u & -mask_u
                    w = lsb.bit_length() - 1
                    if w == v:
                        mask_u -= lsb
                        continue
                    old_cv_w = C[v][w]
                    C[v][w] -= 1
                    C[w][v] -= 1
                    new_cv_w = C[v][w]
                    if old_cv_w >= 3 and new_cv_w < 3:
                        violations -= 1
                    mask_u -= lsb

                new_energy = violations * 100 - num_edges
                old_energy = current_violations * 100 - current_num_edges
                delta_energy = new_energy - old_energy
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    current_violations = violations
                    current_num_edges = num_edges
                    if current_violations == 0 and current_num_edges > best_num_edges:
                        best_num_edges = current_num_edges
                        best_edges = get_edges_from_adj(adj)
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    num_edges += 1
                    
                    mask_v = adj[v] & ~(1 << u)
                    while mask_v:
                        lsb = mask_v & -mask_v
                        w = lsb.bit_length() - 1
                        old_cu_w = C[u][w]
                        C[u][w] += 1
                        C[w][u] += 1
                        new_cu_w = C[u][w]
                        if new_cu_w >= 3 and old_cu_w < 3:
                            violations += 1
                        mask_v -= lsb
                        
                    mask_u = adj[u] & ~(1 << v)
                    while mask_u:
                        lsb = mask_u & -mask_u
                        w = lsb.bit_length() - 1
                        old_cv_w = C[v][w]
                        C[v][w] += 1
                        C[w][v] += 1
                        new_cv_w = C[v][w]
                        if new_cv_w >= 3 and old_cv_w < 3:
                            violations += 1
                        mask_u -= lsb

            else:
                # Add edge
                # Similar logic
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                num_edges += 1
                
                mask_v = adj[v] & ~(1 << u)
                while mask_v:
                    lsb = mask_v & -mask_v
                    w = lsb.bit_length() - 1
                    old_cu_w = C[u][w]
                    C[u][w] += 1
                    C[w][u] += 1
                    new_cu_w = C[u][w]
                    if new_cu_w >= 3 and old_cu_w < 3:
                        violations += 1
                    mask_v -= lsb
                    
                mask_u = adj[u] & ~(1 << v)
                while mask_u:
                    lsb = mask_u & -mask_u
                    w = lsb.bit_length() - 1
                    old_cv_w = C[v][w]
                    C[v][w] += 1
                    C[w][v] += 1
                    new_cv_w = C[v][w]
                    if new_cv_w >= 3 and old_cv_w < 3:
                        violations += 1
                    mask_u -= lsb

                new_energy = violations * 100 - num_edges
                old_energy = current_violations * 100 - current_num_edges
                delta_energy = new_energy - old_energy
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    current_violations = violations
                    current_num_edges = num_edges
                    if current_violations == 0 and current_num_edges > best_num_edges:
                        best_num_edges = current_num_edges
                        best_edges = get_edges_from_adj(adj)
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    num_edges -= 1
                    
                    mask_v = adj[v]
                    while mask_v:
                        lsb = mask_v & -mask_v
                        w = lsb.bit_length() - 1
                        if w == u:
                            mask_v -= lsb
                            continue
                        old_cu_w = C[u][w]
                        C[u][w] -= 1
                        C[w][u] -= 1
                        new_cu_w = C[u][w]
                        if old_cu_w >= 3 and new_cu_w < 3:
                            violations -= 1
                        mask_v -= lsb
                        
                    mask_u = adj[u]
                    while mask_u:
                        lsb = mask_u & -mask_u
                        w = lsb.bit_length() - 1
                        if w == v:
                            mask_u -= lsb
                            continue
                        old_cv_w = C[v][w]
                        C[v][w] -= 1
                        C[w][v] -= 1
                        new_cv_w = C[v][w]
                        if old_cv_w >= 3 and new_cv_w < 3:
                            violations -= 1
                        mask_u -= lsb

            T *= cooling
            steps += 1

    # Run multiple SA instances
    num_runs = 5
    for i in range(num_runs):
        if time.time() > deadline - 10.0:
            break
        seed = 12345 + i
        run_sa(seed, 1.0, 0.01, 0.99, 50000)

    # Final Greedy Add on best_edges
    if best_edges:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if adj[u] & (1 << v):
                    continue
                # Check validity
                # C[u][v] < 3
                # C[u][w] + (1 if w~v) < 3 for all w~u
                # C[v][w] + (1 if w~u) < 3 for all w~v
                
                # Calculate C[u][v]
                cuv = (adj[u] & adj[v]).bit_count()
                if cuv >= 3:
                    continue
                
                valid = True
                # Check neighbors of u
                mask_u = adj[u]
                while mask_u:
                    lsb = mask_u & -mask_u
                    w = lsb.bit_length() - 1
                    if w == v:
                        mask_u -= lsb
                        continue
                    # New C[u][w] = C[u][w] + (1 if w~v)
                    cuw = (adj[u] & adj[w]).bit_count()
                    if adj[w] & (1 << v):
                        cuw += 1
                    if cuw >= 3:
                        valid = False
                        break
                    mask_u -= lsb
                    if not valid: break
                
                if valid:
                    mask_v = adj[v]
                    while mask_v:
                        lsb = mask_v & -mask_v
                        w = lsb.bit_length() - 1
                        if w == u:
                            mask_v -= lsb
                            continue
                        cvw = (adj[v] & adj[w]).bit_count()
                        if adj[w] & (1 << u):
                            cvw += 1
                        if cvw >= 3:
                            valid = False
                            break
                        mask_v -= lsb
                        if not valid: break
                
                if valid:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    best_edges.append([u, v])
                    best_num_edges += 1
                    improved = True

    return best_edges

def _construct_greedy():
    pass

def _greedy_construct(n, all_edges, seed):
    pass

def _can_add_edge_fast(adj, u, v):
    pass

def _local_search(n, edges, all_edges, deadline):
    pass

