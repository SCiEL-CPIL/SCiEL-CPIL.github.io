"""Construct a 5-colouring of K_n with no monochromatic C4."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Seed: n=11 via circular construction
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_n, best_colors = n_seed, colors_seed
    
    def build_adj(colors, n):
        """Build adjacency bitmasks: adj[c][u] = bitmask of neighbors of u in color c."""
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj
    
    def count_c4(adj, n):
        """Count monochromatic C4s: pairs (u,v) with |N_c(u) ∩ N_c(v)| >= 2."""
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[c][u] & adj[c][v]
                    # Popcount
                    cnt = 0
                    x = common
                    while x:
                        x &= x - 1
                        cnt += 1
                    if cnt >= 2:
                        total += 1
        return total
    
    def try_local_search(n, max_time):
        m = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(m)]
        adj = build_adj(colors, n)
        violations = count_c4(adj, n)
        t0 = time.time()
        
        while time.time() - t0 < max_time:
            if violations == 0:
                return colors
            
            # Find a violating edge: pick random edge, check if it's in a C4
            # Simpler: random edge, try recoloring
            edge_idx = random.randint(0, m - 1)
            old_c = colors[edge_idx]
            
            # Recover i, j from edge_idx
            # Lexicographic order: (0,1),(0,2),...,(0,n-1),(1,2),...
            # Find i, j such that sum_{k=0}^{i-1}(n-1-k) + (j-i-1) = edge_idx
            # Or just iterate
            i, j = 0, 1
            found = False
            for ii in range(n):
                remaining = edge_idx
                for jj in range(ii + 1, n):
                    if remaining == 0:
                        i, j = ii, jj
                        found = True
                        break
                    remaining -= 1
                if found:
                    break
            
            best_new_c = old_c
            best_violations = violations
            
            for new_c in range(5):
                if new_c == old_c:
                    continue
                # Apply change
                colors[edge_idx] = new_c
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                v = count_c4(adj, n)
                if v < best_violations:
                    best_violations = v
                    best_new_c = new_c
                # Revert
                colors[edge_idx] = old_c
                adj[new_c][i] &= ~(1 << j)
                adj[new_c][j] &= ~(1 << i)
                adj[old_c][i] |= (1 << j)
                adj[old_c][j] |= (1 << i)
            
            if best_new_c != old_c:
                colors[edge_idx] = best_new_c
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                adj[best_new_c][i] |= (1 << j)
                adj[best_new_c][j] |= (1 << i)
                violations = best_violations
            else:
                # Perturb
                for _ in range(3):
                    ei = random.randint(0, m - 1)
                    # Find i,j
                    rem = ei
                    ii, jj = 0, 1
                    for iii in range(n):
                        for jjj in range(iii + 1, n):
                            if rem == 0:
                                ii, jj = iii, jjj
                                break
                            rem -= 1
                        if rem == 0:
                            break
                    nc = random.randint(0, 4)
                    oc = colors[ei]
                    adj[oc][ii] &= ~(1 << jj)
                    adj[oc][jj] &= ~(1 << ii)
                    adj[nc][ii] |= (1 << jj)
                    adj[nc][jj] |= (1 << ii)
                    colors[ei] = nc
                violations = count_c4(adj, n)
        
        return None
    
    # Try n from 26 down to 12
    for target_n in range(26, 11, -1):
        if time.time() >= deadline - 5.0:
            break
        remaining = deadline - time.time() - 2.0
        if remaining < 0.5:
            break
        max_t = min(remaining, 15.0)
        result = try_local_search(target_n, max_t)
        if result is not None:
            adj = build_adj(result, target_n)
            if count_c4(adj, target_n) == 0:
                best_n = target_n
                best_colors = result
                break
    
    return {'n': best_n, 'colors': best_colors}
