import time
import sys
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    def time_left():
        return deadline - time.time()

    # 1. Build the base K32 Greenwood-Gleason coloring
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    n_base = 32
    colors = [0] * (n_base * (n_base - 1) // 2)
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i // 16 == j // 16:
                colors[idx] = log[(i % 16) ^ (j % 16)] % 3
            else:
                colors[idx] = 3
            idx += 1

    def build_adj(n, col_list):
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = col_list[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        return adj

    adj = build_adj(n_base, colors)
    current_n = n_base

    # Precompute conflict masks for the current graph
    # conflict_mask[c][u] is a bitmask of vertices w such that adj[u][w] == c
    def compute_conflict_masks(n, adj):
        masks = [[0] * n for _ in range(4)]
        for c in range(4):
            for u in range(n):
                mask = 0
                for w in range(n):
                    if w != u and adj[u][w] == c:
                        mask |= (1 << w)
                masks[c][u] = mask
        return masks

    conflict_masks = compute_conflict_masks(current_n, adj)

    # Function to add a single vertex using backtracking
    def add_vertex_backtrack(n, adj, conflict_masks):
        assignment = [None] * n
        order = list(range(n))
        
        # Sort by degree in the union of all conflict graphs (most constrained first)
        def total_degree(u):
            deg = 0
            for c in range(4):
                deg += bin(conflict_masks[c][u]).count('1')
            return deg
        
        order.sort(key=total_degree, reverse=True)
        
        def backtrack(pos, color_sets):
            if pos == n:
                return assignment
            
            u = order[pos]
            
            # Try colors 0,1,2,3
            for c in range(4):
                # Check constraint: u cannot be connected to any vertex already in color_sets[c] via color c
                if conflict_masks[c][u] & color_sets[c]:
                    continue
                
                # Assign
                assignment[u] = c
                new_color_sets = color_sets[:]
                new_color_sets[c] |= (1 << u)
                
                # Forward checking
                valid = True
                for pos2 in range(pos + 1, n):
                    v = order[pos2]
                    has_valid = False
                    for c2 in range(4):
                        if not (conflict_masks[c2][v] & new_color_sets[c2]):
                            has_valid = True
                            break
                    if not has_valid:
                        valid = False
                        break
                    if time.time() > deadline:
                        valid = False
                        break
                
                if valid:
                    result = backtrack(pos + 1, new_color_sets)
                    if result is not None:
                        return result
                
                # Undo
                assignment[u] = None
            
            return None

        color_sets = [0, 0, 0, 0]
        return backtrack(0, color_sets)

    # Add 18 vertices to reach 50
    for step in range(18):
        if time.time() > deadline:
            break
        
        assignment = add_vertex_backtrack(current_n, adj, conflict_masks)
        
        if assignment is None:
            # If backtracking fails, try a simpler greedy approach or break
            # For robustness, we break and return current_n
            break
        
        # Update adj and colors
        new_idx = current_n
        adj.append([0] * (current_n + 1))
        for i in range(current_n):
            adj[i].append(assignment[i])
            adj[new_idx][i] = assignment[i]
        adj[new_idx][new_idx] = -1
        
        for i in range(current_n):
            colors.append(assignment[i])
        
        current_n += 1
        
        # Recompute conflict masks for the next vertex
        conflict_masks = compute_conflict_masks(current_n, adj)

    # Now try to extend to 51 using Simulated Annealing on the new vertex's edges
    if current_n == 50 and time.time() < deadline - 20:
        # We have a valid K50. Try to add vertex 50 (index 50).
        # The edges are (50, u) for u in 0..49.
        # We need to assign colors to these 50 edges.
        
        n_curr = current_n
        # conflict_masks for K50 is already computed
        
        def sa_add_vertex(n, conflict_masks, time_limit):
            # Variables: x[0..n-1] where x[u] is color of edge (n, u)
            # Objective: Minimize mono-triangles involving vertex n.
            # A triangle (u, v, n) is mono if x[u]=x[v]=c and adj[u][v]=c.
            
            # Initialize randomly
            x = [random.randint(0, 3) for _ in range(n)]
            
            def count_violations(x):
                count = 0
                for u in range(n):
                    c_u = x[u]
                    # Check all v > u
                    for v in range(u + 1, n):
                        c_v = x[v]
                        if c_u == c_v:
                            # Check if adj[u][v] == c_u
                            if adj[u][v] == c_u:
                                count += 1
                return count
            
            curr_violations = count_violations(x)
            best_x = x[:]
            best_violations = curr_violations
            
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            start = time.time()
            
            while time.time() - start < time_limit:
                if best_violations == 0:
                    return best_x
                
                # Pick random u
                u = random.randint(0, n - 1)
                old_c = x[u]
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue
                
                # Calculate delta
                # Violations involving u:
                # For each v, if x[v] == c, and adj[u][v] == c, it's a violation.
                # Changing x[u] from old_c to new_c:
                # Removes violations where x[v] == old_c and adj[u][v] == old_c
                # Adds violations where x[v] == new_c and adj[u][v] == new_c
                
                delta = 0
                for v in range(n):
                    if v == u:
                        continue
                    # Remove
                    if x[v] == old_c and adj[u][v] == old_c:
                        delta -= 1
                    # Add
                    if x[v] == new_c and adj[u][v] == new_c:
                        delta += 1
                
                if delta <= 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                    x[u] = new_c
                    curr_violations += delta
                    if curr_violations < best_violations:
                        best_violations = curr_violations
                        best_x = x[:]
                
                T *= cooling
            
            return best_x if best_violations == 0 else None

        # Run SA with a time budget
        sa_time_budget = min(50.0, deadline - time.time() - 5.0)
        if sa_time_budget > 5.0:
            new_assignment = sa_add_vertex(current_n, conflict_masks, sa_time_budget)
            if new_assignment is not None:
                # Success! Add the vertex
                new_idx = current_n
                adj.append([0] * (current_n + 1))
                for i in range(current_n):
                    adj[i].append(new_assignment[i])
                    adj[new_idx][i] = new_assignment[i]
                adj[new_idx][new_idx] = -1
                
                for i in range(current_n):
                    colors.append(new_assignment[i])
                
                current_n += 1

    # Build final colors list
    final_colors = []
    idx = 0
    for i in range(current_n):
        for j in range(i + 1, current_n):
            final_colors.append(adj[i][j])
    
    return {'n': current_n, 'colors': final_colors}
