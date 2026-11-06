import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Fallback: small valid coloring (n=11)
    n_fb = 11
    colors_fb = [min(j - i, n_fb - j + i) - 1 for i in range(n_fb) for j in range(i + 1, n_fb)]
    best_n = n_fb
    best_colors = colors_fb

    def valid_n(n, colors):
        """Check no monochromatic C4."""
        num_edges = n * (n - 1) // 2
        edge_index = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[i, j] = idx
                edge_index[j, i] = idx
                idx += 1
        for c in range(5):
            adj = np.zeros((n, n), dtype=np.int8)
            for e in range(num_edges):
                if colors[e] == c:
                    i = e
                    # decode
                    # faster: rebuild
            # rebuild adjacency per color
        # Simpler: build adj matrices
        adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i, j] = 1
                adj[c][j, i] = 1
                idx += 1
        for c in range(5):
            # C4 exists iff (A^2)_{ij} >= 2 for some i != j
            A2 = adj[c] @ adj[c]
            np.fill_diagonal(A2, 0)
            if np.any(A2 >= 2):
                return False
        return True

    def algebraic_seed(n):
        """
        Try to build a C4-free 5-coloring inspired by finite geometry / algebraic methods.
        Use a simple structured approach: assign colors based on difference classes mod 5
        with perturbation, then verify. If n is small enough, this may already be valid.
        For larger n, use as a starting point for SA.
        """
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % 5
                # Add some structure: use (i+j) mod 5 as secondary
                c = (d + (i + j) % 5) % 5
                colors.append(c)
        return colors

    def random_seed(n):
        num_edges = n * (n - 1) // 2
        return list(np.random.randint(0, 5, size=num_edges))

    def count_c4s(n, colors):
        """Count total monochromatic C4s (each C4 counted once per color)."""
        num_edges = n * (n - 1) // 2
        adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i, j] = 1
                adj[c][j, i] = 1
                idx += 1
        total = 0
        for c in range(5):
            A2 = adj[c] @ adj[c]
            np.fill_diagonal(A2, 0)
            # Each C4 contributes 2 to A2[i][j] for opposite vertices... 
            # Actually number of C4s = sum over i<j of C(A2[i][j], 2) / 2? 
            # No: A2[i][j] = number of common neighbors. A C4 is two vertices with >=2 common neighbors.
            # Number of C4s in color c = sum_{i<j} C(A2[i][j], 2)
            # But each C4 is counted once by this (it's defined by its two diagonals).
            # Wait: a C4 has two pairs of opposite vertices. Each pair has exactly 2 common neighbors.
            # So each C4 contributes C(2,2)=1 for each of its two diagonal pairs. So it's counted twice.
            # Thus #C4s = sum_{i<j} C(A2[i][j], 2) / 2
            vals = A2[np.triu_indices(n, 1)]
            total += int(np.sum(vals * (vals - 1) // 2)) // 2
        return total

    def sa_search(n, initial_colors, max_time):
        """Simulated annealing local search."""
        num_edges = n * (n - 1) // 2
        colors = np.array(initial_colors, dtype=np.int8)
        
        # Precompute edge endpoints
        endpoints = []
        for i in range(n):
            for j in range(i + 1, n):
                endpoints.append((i, j))
        
        # Build adjacency per color
        adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = int(colors[idx])
                adj[c][i, j] = 1
                adj[c][j, i] = 1
                idx += 1
        
        current_cost = count_c4s(n, colors.tolist())
        if current_cost == 0:
            return colors.tolist()
        
        # SA parameters
        T_start = 2.0
        T_end = 0.01
        n_iter = 30000
        cool_rate = (T_end / T_start) ** (1.0 / max(n_iter - 1, 1))
        T = T_start
        
        best_cost = current_cost
        best_colors = colors.copy()
        
        # Neighbor list for faster updates
        # For each edge, we need to compute delta cost when changing its color.
        # Delta cost for changing edge (u,v) from c_old to c_new:
        # For color c_old: removing edge (u,v) reduces A2[c_old][x][y] for x in N(u,c_old)\{v}, y in N(v,c_old)\{u}
        #   specifically, for each x != u,v: if adj[c_old][u,x] and adj[c_old][v,x], then A2[c_old][u,v]... 
        #   Actually we need to track how C4 count changes.
        
        # Simpler: just recompute cost after each move (slower but correct).
        # For n up to 26, num_edges ~ 325, n=26, A2 computation is 26x26 matmul x5 = fast.
        
        for it in range(n_iter):
            if time.time() > deadline - 2:
                break
            
            # Pick random edge
            e_idx = np.random.randint(num_edges)
            u, v = endpoints[e_idx]
            old_c = int(colors[e_idx])
            new_c = np.random.randint(0, 5)
            if new_c == old_c:
                continue
            
            # Compute cost delta
            # Remove edge from old_c, add to new_c
            adj[old_c][u, v] = 0
            adj[old_c][v, u] = 0
            adj[new_c][u, v] = 1
            adj[new_c][v, u] = 1
            
            new_cost = count_c4s(n, colors.tolist())
            # Wait, colors array not updated yet. Let me update temporarily.
            # Actually count_c4s uses colors list. Let me just update adj and compute from adj.
            
            # Recompute cost from adj directly
            def cost_from_adj(adj):
                total = 0
                for c in range(5):
                    A2 = adj[c] @ adj[c]
                    np.fill_diagonal(A2, 0)
                    vals = A2[np.triu_indices(n, 1)]
                    total += int(np.sum(vals * (vals - 1) // 2)) // 2
                return total
            
            new_cost = cost_from_adj(adj)
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Accept
                colors[e_idx] = new_c
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors.tolist()
            else:
                # Reject: revert
                adj[old_c][u, v] = 1
                adj[old_c][v, u] = 1
                adj[new_c][u, v] = 0
                adj[new_c][v, u] = 0
            
            T *= cool_rate
        
        if best_cost == 0:
            return best_colors.tolist()
        return None

    # Strategy: 
    # 1. Try algebraic seeds + SA for n=12..26
    # 2. Also try random seeds + SA as backup
    # Focus most time on higher n
    
    # Allocate time: spend more on larger n
    n_values = list(range(12, 27))
    # Weight by n: larger n gets more time
    weights = [n**2 for n in n_values]
    total_w = sum(weights)
    available_time = deadline - time.time() - 3
    
    # Try from high n down to low n, giving more budget to higher n
    for n in sorted(n_values, key=lambda x: -x):
        if time.time() > deadline - 3:
            break
        
        # Time budget for this n
        frac = weights[n_values.index(n)] / total_w
        n_time = max(5, frac * available_time)
        
        # Try algebraic seed first
        seed_colors = algebraic_seed(n)
        result = sa_search(n, seed_colors, n_time / 2)
        if result is not None:
            best_n = n
            best_colors = result
            break
        
        # Try random seeds
        n_restarts = max(2, int(n_time / 15))
        for r in range(n_restarts):
            if time.time() > deadline - 3:
                break
            seed_colors = random_seed(n)
            result = sa_search(n, seed_colors, n_time / (n_restarts + 1))
            if result is not None:
                best_n = n
                best_colors = result
                break
        
        if best_n >= n:
            break
    
    return {'n': best_n, 'colors': best_colors}
