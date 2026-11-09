import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # Safety margin for 120s limit
    best_edges = 0
    best_adj = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Precompute 4-cycles
    # A 4-cycle is defined by 4 distinct vertices a, b, c, d.
    # The 3 possible cycles on these 4 vertices are:
    # 1: (a,b), (b,c), (c,d), (d,a)
    # 2: (a,b), (b,d), (d,c), (c,a) -> (a,b), (b,d), (c,d), (a,c)
    # 3: (a,c), (c,b), (b,d), (d,a) -> (a,c), (b,c), (b,d), (a,d)
    
    all_4cycles = []
    edge_to_cycles = {}
    
    def get_edge_key(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
        
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Cycle 1: a-b-c-d-a
                    e1 = get_edge_key(a, b)
                    e2 = get_edge_key(b, c)
                    e3 = get_edge_key(c, d)
                    e4 = get_edge_key(d, a)
                    cyc1 = [e1, e2, e3, e4]
                    idx = len(all_4cycles)
                    all_4cycles.append(cyc1)
                    for e in cyc1:
                        if e not in edge_to_cycles:
                            edge_to_cycles[e] = []
                        edge_to_cycles[e].append(idx)
                    
                    # Cycle 2: a-b-d-c-a
                    e1 = get_edge_key(a, b)
                    e2 = get_edge_key(b, d)
                    e3 = get_edge_key(d, c)
                    e4 = get_edge_key(c, a)
                    cyc2 = [e1, e2, e3, e4]
                    idx = len(all_4cycles)
                    all_4cycles.append(cyc2)
                    for e in cyc2:
                        if e not in edge_to_cycles:
                            edge_to_cycles[e] = []
                        edge_to_cycles[e].append(idx)
                    
                    # Cycle 3: a-c-b-d-a
                    e1 = get_edge_key(a, c)
                    e2 = get_edge_key(c, b)
                    e3 = get_edge_key(b, d)
                    e4 = get_edge_key(d, a)
                    cyc3 = [e1, e2, e3, e4]
                    idx = len(all_4cycles)
                    all_4cycles.append(cyc3)
                    for e in cyc3:
                        if e not in edge_to_cycles:
                            edge_to_cycles[e] = []
                        edge_to_cycles[e].append(idx)
                        
    num_cycles = len(all_4cycles)
    
    # Initialize adjacency matrix with random colors
    # adj[u, v] = color, -1 if no edge (though we start with K_n)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            c = random.randint(0, n-1)
            adj[i, j] = c
            adj[j, i] = c
            
    # Calculate initial violations
    violations = np.zeros(num_cycles, dtype=int)
    def is_violated(cyc, adj):
        colors = []
        for e in cyc:
            u, v = e
            colors.append(adj[u, v])
        return 1 if len(set(colors)) < 4 else 0
        
    for i in range(num_cycles):
        violations[i] = is_violated(all_4cycles[i], adj)
        
    current_violations = int(np.sum(violations))
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    steps = 0
    max_steps = 500000
    
    # Track best solution (lowest violations)
    best_violations = current_violations
    best_adj_copy = adj.copy()
    
    while time.time() < deadline - 10: # Leave 10s for fallback if needed
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        old_c = adj[u, v]
        new_c = random.randint(0, n-1)
        if old_c == new_c:
            continue
            
        # Calculate delta
        delta = 0
        cycles_for_edge = edge_to_cycles.get((u, v), [])
        for ci in cycles_for_edge:
            cyc = all_4cycles[ci]
            # Get colors of the 4 edges
            # One of them is (u, v)
            c1, c2, c3, c4 = -1, -1, -1, -1
            count = 0
            for e in cyc:
                eu, ev = e
                c = adj[eu, ev]
                if e == (u, v):
                    c = new_c # Use new color for calculation
                # We need to check if the 4 colors are distinct
                # Instead of building a list every time, we can do a quick check
                # But building a list of 4 ints is cheap.
                if count == 0: c1 = c
                elif count == 1: c2 = c
                elif count == 2: c3 = c
                else: c4 = c
                count += 1
                
            # Check distinctness
            # We know old violation state is violations[ci]
            # New violation state:
            new_viol = 1
            if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                new_viol = 0
                
            old_viol = violations[ci]
            if new_viol != old_viol:
                delta += new_viol - old_viol
                # We don't update violations[ci] yet, only if we accept
                
        # Acceptance
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            adj[u, v] = new_c
            adj[v, u] = new_c
            
            # Update violations
            for ci in cycles_for_edge:
                cyc = all_4cycles[ci]
                c1, c2, c3, c4 = -1, -1, -1, -1
                count = 0
                for e in cyc:
                    eu, ev = e
                    c = adj[eu, ev]
                    if count == 0: c1 = c
                    elif count == 1: c2 = c
                    elif count == 2: c3 = c
                    else: c4 = c
                    count += 1
                new_viol = 1
                if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                    new_viol = 0
                violations[ci] = new_viol
                
            current_violations += delta
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_adj_copy = adj.copy()
                if best_violations == 0:
                    # Found a perfect solution!
                    break
                    
        T *= cooling
        steps += 1
        
    # If we found a valid K_20, return it
    if best_violations == 0:
        res = []
        for i in range(n):
            for j in range(i+1, n):
                res.append([i, j, int(best_adj_copy[i, j])])
        return res
        
    # Fallback: If SA didn't find a perfect K_20, we need to return a VALID graph.
    # We can use the best_adj_copy, but it has violations.
    # We must remove edges to eliminate all violations.
    # Greedy removal:
    # While there are violations, pick an edge involved in a violation and remove it.
    
    adj_fb = best_adj_copy.copy()
    # Re-evaluate violations for adj_fb
    viol_fb = np.zeros(num_cycles, dtype=int)
    for i in range(num_cycles):
        viol_fb[i] = is_violated(all_4cycles[i], adj_fb)
        
    cur_viol_fb = int(np.sum(viol_fb))
    
    # Map edge to index for removal
    # We need to be careful: removing an edge makes all cycles containing it "invalid" (not 4-cycles in the graph anymore).
    # The problem says "Every 4-cycle (as a subgraph) must receive four distinct colours".
    # If an edge is removed, it's not a 4-cycle anymore.
    # So we just need to ensure that for all *remaining* 4-cycles, the colors are distinct.
    
    # Efficient removal:
    # Keep a list of active edges.
    # Actually, simpler:
    # Iterate. Find a violated cycle. Pick one of its edges. Remove it.
    # Repeat until no violations.
    
    # To speed up, we can track which cycles are still "active" (all 4 edges present).
    # But for n=20, even naive O(N_cycles) check per removal is okay if removals are few.
    # Max removals 190. 190 * 14535 = 2.7M ops. Very fast.
    
    # Let's just use the adjacency matrix and a check function.
    
    def check_validity(adj_mat):
        for i in range(num_cycles):
            if viol_fb[i] == 0: # Optimization: skip if we know it's ok? No, adj changed.
                 pass
            cyc = all_4cycles[i]
            # Check if all edges exist
            e1, e2, e3, e4 = cyc
            if adj_mat[e1[0], e1[1]] == -1 or adj_mat[e2[0], e2[1]] == -1 or \
               adj_mat[e3[0], e3[1]] == -1 or adj_mat[e4[0], e4[1]] == -1:
                continue
            c1 = adj_mat[e1[0], e1[1]]
            c2 = adj_mat[e2[0], e2[1]]
            c3 = adj_mat[e3[0], e3[1]]
            c4 = adj_mat[e4[0], e4[1]]
            if len({c1, c2, c3, c4}) < 4:
                return False, i
        return True, -1
        
    # Initialize adj_fb with -1 for removed, but we start with best_adj_copy which has all edges.
    # We need a separate matrix for "existence" or just use -1.
    # best_adj_copy has colors. Let's make a copy where we can set to -1.
    
    adj_work = best_adj_copy.copy()
    
    # We need to identify edges to remove.
    # Let's just loop.
    max_iter = 190
    for _ in range(max_iter):
        valid, viol_idx = check_validity(adj_work)
        if valid:
            break
        # Remove an edge from the violated cycle
        cyc = all_4cycles[viol_idx]
        # Pick a random edge from the cycle to remove
        e_to_remove = random.choice(cyc)
        u, v = e_to_remove
        adj_work[u, v] = -1
        adj_work[v, u] = -1
        
    # Construct result
    res = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_work[i, j] != -1:
                res.append([i, j, int(adj_work[i, j])])
                
    return res
