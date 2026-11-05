import time
import random
import numpy as np

def construct():
    deadline = time.time() + 118  # Safety margin
    
    def build_adj(colors, n):
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        return adj

    def has_k222(adj, n):
        """Return True if adj contains a K2,2,2 subgraph. Early exit."""
        for u1 in range(n):
            a1 = adj[u1]
            for u2 in range(u1 + 1, n):
                if not (a1 >> u2) & 1:
                    continue
                s_mask = a1 & adj[u2]
                s_mask &= ~(1 << u1)
                s_mask &= ~(1 << u2)
                if s_mask.bit_count() < 4:
                    continue
                # Check if G[S] contains K2,2
                # For each pair (v1,v2) in S, check if they have >=2 common neighbors in S
                # Use bit-level: collect bits of S
                verts = []
                m = s_mask
                while m:
                    lsb = m & -m
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                l = len(verts)
                for i in range(l):
                    v1 = verts[i]
                    a_v1 = adj[v1]
                    for j in range(i + 1, l):
                        v2 = verts[j]
                        common = a_v1 & adj[v2] & s_mask
                        # Check if at least 2 bits set
                        if common & (common - 1):
                            return True
        return False

    def build_adj_for_color(colors, n, color):
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == color:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        return adj

    def is_valid_coloring(colors, n):
        """Return True if neither color class contains K2,2,2."""
        adj0 = build_adj_for_color(colors, n, 0)
        if has_k222(adj0, n):
            return False
        adj1 = build_adj_for_color(colors, n, 1)
        if has_k222(adj1, n):
            return False
        return True

    def count_violations_fast(adj, n):
        """Count K2,2,2 violations using early termination per pair."""
        count = 0
        for u1 in range(n):
            a1 = adj[u1]
            for u2 in range(u1 + 1, n):
                if not (a1 >> u2) & 1:
                    continue
                s_mask = a1 & adj[u2]
                s_mask &= ~(1 << u1)
                s_mask &= ~(1 << u2)
                if s_mask.bit_count() < 4:
                    continue
                verts = []
                m = s_mask
                while m:
                    lsb = m & -m
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                l = len(verts)
                for i in range(l):
                    v1 = verts[i]
                    a_v1 = adj[v1]
                    for j in range(i + 1, l):
                        v2 = verts[j]
                        common = a_v1 & adj[v2] & s_mask
                        if common & (common - 1):
                            count += 1
                            # Early exit: we just need to know violations exist
                            # But for gradient, let's count a few
                            if count >= 100:
                                return count
        return count

    def total_violations_fast(colors, n):
        adj0 = build_adj_for_color(colors, n, 0)
        adj1 = build_adj_for_color(colors, n, 1)
        return count_violations_fast(adj0, n) + count_violations_fast(adj1, n)

    # 1. Paley 29
    q = 29
    squares = {x * x % q for x in range(1, q)}
    colors_29 = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                colors_29.append(0)
            else:
                colors_29.append(1)
    
    # Verify
    if total_violations_fast(colors_29, 29) == 0:
        best_n = 29
        best_colors = colors_29
    else:
        best_n = 1
        best_colors = [0]

    # 2. Try n=30
    n = 30
    n_edges = n * (n - 1) // 2
    
    best_n_30 = False
    best_colors_30 = None
    
    # Multiple restarts
    max_restarts = 5
    restart_time_budget = (deadline - 5 - time.time()) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize with Paley 29 + random 29 edges
        init_colors = colors_29.copy()
        for _ in range(29):
            init_colors.append(random.randint(0, 1))
        
        current_colors = init_colors
        current_viol = total_violations_fast(current_colors, n)
        
        if current_viol == 0:
            best_n_30 = True
            best_colors_30 = current_colors
            break
        
        # Simulated Annealing / Hill Climbing
        T = 2.0
        T_min = 0.05
        alpha = 0.9995
        no_improve_count = 0
        best_viol_this_restart = current_viol
        best_colors_this_restart = current_colors.copy()
        
        while time.time() < deadline - 5:
            edge_idx = random.randint(0, n_edges - 1)
            old_val = current_colors[edge_idx]
            new_val = 1 - old_val
            current_colors[edge_idx] = new_val
            
            new_viol = total_violations_fast(current_colors, n)
            
            delta = new_viol - current_viol
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_viol = new_viol
                if current_viol < best_viol_this_restart:
                    best_viol_this_restart = current_viol
                    best_colors_this_restart = current_colors.copy()
                    no_improve_count = 0
                if current_viol == 0:
                    best_n_30 = True
                    best_colors_30 = current_colors
                    break
            else:
                current_colors[edge_idx] = old_val
                no_improve_count += 1
            
            T *= alpha
            if T < T_min:
                T = T_min
            
            # If stuck for too long, restart from best
            if no_improve_count > 500:
                current_colors = best_colors_this_restart.copy()
                current_viol = best_viol_this_restart
                no_improve_count = 0
                T = max(T, 1.0)
        
        if best_n_30:
            break
    
    if best_n_30:
        return {'n': n, 'colors': best_colors_30}
    
    return {'n': best_n, 'colors': best_colors}
