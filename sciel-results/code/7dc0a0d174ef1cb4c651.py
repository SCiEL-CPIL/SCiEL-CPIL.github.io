import numpy as np
import time
import random
import math

def count_edges(adj):
    return np.sum(adj != -1) // 2

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115
    
    random.seed(42)
    np.random.seed(42)
    
    # Precompute 4-cycle witnesses for each edge (u,v)
    # witness[(u,v)] = list of (x, y) such that u-v-x-y-u is a 4-cycle
    # Note: The cycle is u-v-x-y-u. The edges are (u,v), (v,x), (x,y), (y,u).
    witness = {}
    for u in range(n):
        for v in range(u+1, n):
            wl = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    wl.append((x, y))
            witness[(u, v)] = wl
            witness[(v, u)] = wl
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(all_edges)
    
    # Map edge to index for faster access?
    # adj is n x n matrix.
    
    def get_violation_count(u, v, adj):
        """
        Returns the number of 4-cycles containing edge (u,v) that are currently violated.
        A cycle is violated if all 4 edges are present and their colors are not all distinct.
        """
        if adj[u, v] == -1:
            return 0
        c_uv = adj[u, v]
        violations = 0
        for (x, y) in witness[(u, v)]:
            # Edges: (u,v), (v,x), (x,y), (y,u)
            if adj[v, x] == -1 or adj[x, y] == -1 or adj[y, u] == -1:
                continue
            
            c_vx = adj[v, x]
            c_xy = adj[x, y]
            c_yu = adj[y, u]
            
            if len({c_uv, c_vx, c_xy, c_yu}) < 4:
                violations += 1
        return violations

    def delta_violations_add(u, v, c, adj):
        """
        Calculate change in violation count if we ADD edge (u,v) with color c.
        Returns (delta_violations).
        """
        delta = 0
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1 or adj[x, y] == -1 or adj[y, u] == -1:
                continue
            
            c_vx = adj[v, x]
            c_xy = adj[x, y]
            c_yu = adj[y, u]
            
            # Before: (u,v) missing, so no violation from this cycle.
            # After: (u,v) present with c.
            if len({c, c_vx, c_xy, c_yu}) < 4:
                delta += 1
        return delta

    def delta_violations_remove(u, v, adj):
        """
        Calculate change in violation count if we REMOVE edge (u,v).
        """
        if adj[u, v] == -1:
            return 0
        c_uv = adj[u, v]
        delta = 0
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1 or adj[x, y] == -1 or adj[y, u] == -1:
                continue
            
            c_vx = adj[v, x]
            c_xy = adj[x, y]
            c_yu = adj[y, u]
            
            # Before: (u,v) present.
            if len({c_uv, c_vx, c_xy, c_yu}) < 4:
                delta -= 1
            # After: (u,v) missing, so no violation.
        return delta

    def delta_violations_recolor(u, v, new_c, adj):
        """
        Calculate change in violation count if we RECOLOR edge (u,v) to new_c.
        """
        if adj[u, v] == -1:
            return 0
        old_c = adj[u, v]
        if old_c == new_c:
            return 0
        
        delta = 0
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1 or adj[x, y] == -1 or adj[y, u] == -1:
                continue
            
            c_vx = adj[v, x]
            c_xy = adj[x, y]
            c_yu = adj[y, u]
            
            # Before
            before_violation = 0
            if len({old_c, c_vx, c_xy, c_yu}) < 4:
                before_violation = 1
            
            # After
            after_violation = 0
            if len({new_c, c_vx, c_xy, c_yu}) < 4:
                after_violation = 1
            
            delta += after_violation - before_violation
            
        return delta

    def total_violations(adj):
        """
        Calculate total violations in the graph.
        """
        total = 0
        # Sum violations for each edge, but each violation is counted 4 times (once per edge in the cycle)
        # So we sum and divide by 4.
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] != -1:
                    total += get_violation_count(u, v, adj)
        return total // 4

    def score(adj):
        e_count = count_edges(adj)
        v_count = total_violations(adj)
        # We want to maximize edges and minimize violations.
        # Penalty should be high enough to prioritize fixing violations.
        return e_count - 10 * v_count

    # Simulated Annealing
    best_adj = None
    best_score = -np.inf
    
    # Initialize with a random graph
    adj = -np.ones((n, n), dtype=int)
    np.fill_diagonal(adj, -1)
    
    # Start with a random sparse graph to have some structure
    # Or start empty? Starting empty is safe but might be slow to build.
    # Let's start with a random 50% density graph and then SA.
    for u, v in all_edges:
        if random.random() < 0.5:
            c = random.randint(0, n-1)
            adj[u, v] = c
            adj[v, u] = c
            
    current_score = score(adj)
    if current_score > best_score:
        best_score = current_score
        best_adj = adj.copy()
    
    T = 5.0
    cooling = 0.999
    steps = 0
    max_steps = 500000
    
    while time.time() < deadline - 5:
        # Random move
        u, v = random.choice(all_edges)
        move_type = random.randint(0, 2) # 0: add, 1: remove, 2: recolor
        
        old_score = current_score
        delta_score = 0
        
        if move_type == 0: # Add
            if adj[u, v] == -1:
                c = random.randint(0, n-1)
                dv = delta_violations_add(u, v, c, adj)
                # Score change: +1 edge, -10 * dv violations
                delta_score = 1 - 10 * dv
                
                if delta_score > 0 or random.random() < math.exp(delta_score / T):
                    adj[u, v] = c
                    adj[v, u] = c
                    current_score = old_score + delta_score
                    if current_score > best_score:
                        best_score = current_score
                        best_adj = adj.copy()
        
        elif move_type == 1: # Remove
            if adj[u, v] != -1:
                dv = delta_violations_remove(u, v, adj)
                # Score change: -1 edge, -10 * dv violations (dv is negative if we remove a violation)
                delta_score = -1 - 10 * dv
                
                if delta_score > 0 or random.random() < math.exp(delta_score / T):
                    adj[u, v] = -1
                    adj[v, u] = -1
                    current_score = old_score + delta_score
                    if current_score > best_score:
                        best_score = current_score
                        best_adj = adj.copy()
                        
        else: # Recolor
            if adj[u, v] != -1:
                new_c = random.randint(0, n-1)
                if new_c != adj[u, v]:
                    dv = delta_violations_recolor(u, v, new_c, adj)
                    # Score change: 0 edges, -10 * dv violations
                    delta_score = -10 * dv
                    
                    if delta_score > 0 or random.random() < math.exp(delta_score / T):
                        adj[u, v] = new_c
                        adj[v, u] = new_c
                        current_score = old_score + delta_score
                        if current_score > best_score:
                            best_score = current_score
                            best_adj = adj.copy()
                            
        T *= cooling
        steps += 1
        
        # If T is too low, restart with higher T?
        if T < 0.1:
            T = 5.0
            
    # Final Polish: Remove any remaining violations by deleting edges
    if best_adj is not None:
        adj = best_adj.copy()
        # Iteratively remove edges involved in violations
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v] != -1:
                        # Check if this edge is part of any violation
                        for (x, y) in witness[(u, v)]:
                            if adj[v, x] != -1 and adj[x, y] != -1 and adj[y, u] != -1:
                                c1 = adj[u, v]
                                c2 = adj[v, x]
                                c3 = adj[x, y]
                                c4 = adj[y, u]
                                if len({c1, c2, c3, c4}) < 4:
                                    # Remove one of the edges, e.g., (u,v)
                                    adj[u, v] = -1
                                    adj[v, u] = -1
                                    changed = True
                                    break
                            if changed:
                                break
                if changed:
                    break
                    
        cnt = count_edges(adj)
        if cnt > count_edges(best_adj):
            best_adj = adj

    # Convert to result
    res = []
    if best_adj is None:
        # Fallback: Empty graph or minimal
        return []
        
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
