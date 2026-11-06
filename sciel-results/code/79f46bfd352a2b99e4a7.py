import time
import random
import numpy as np

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Maximizes n.
    """
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
    # Known seed solution for n=11 (cyclic group Z11, colors based on difference)
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()
    
    # Helper to get edge index in lexicographic order
    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (i - 1) // 2 + (j - i - 1)

    # Helper for verification
    def is_valid(n, colors):
        neighbors = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[i][c].add(j)
                neighbors[j][c].add(i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                nu = neighbors[u][c]
                if len(nu) < 2:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[v][c]
                    common = nu & nv
                    if len(common) >= 2:
                        return False
        return True

    # Simulated Annealing approach with NumPy acceleration
    # Objective: minimize number of C4 violations
    # State: color assignment for all edges
    # Move: change color of one edge
    
    def solve_n_sa(n, time_limit):
        total_edges = n * (n - 1) // 2
        
        # Precompute edge endpoints
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        # Precompute for each edge, which other edges share a vertex with it
        # This is for efficient violation counting? 
        # Actually, the NumPy trick is to maintain adjacency matrices per color.
        
        # State: array of colors for each edge
        state = np.random.randint(0, 5, size=total_edges)
        
        # Build initial adjacency matrices
        # adj[c][u][v] = 1 if edge (u,v) is color c
        adj = np.zeros((5, n, n), dtype=np.int32)
        
        def build_adj(state):
            adj.fill(0)
            for idx, (u, v) in enumerate(edges):
                c = state[idx]
                adj[c][u][v] = 1
                adj[c][v][u] = 1
        
        build_adj(state)
        
        # Count violations
        # For each color c, a violation is a pair (u,v) with >=2 common neighbors.
        # Common neighbors of u and v in color c is sum(adj[c][u] * adj[c][v])
        # We want to count pairs (u,v) where this sum >= 2.
        
        def count_violations(adj):
            total_viol = 0
            for c in range(5):
                # Matrix M = adj[c]
                # Common neighbors matrix C = M @ M
                # C[u][v] = number of common neighbors
                # We want to count pairs u < v where C[u][v] >= 2
                # Note: C is symmetric.
                C = adj[c] @ adj[c]
                # Mask for u < v
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                # Count entries in C where mask is True and C >= 2
                viol = np.sum((C >= 2) & mask)
                total_viol += viol
            return total_viol
        
        current_viol = count_violations(adj)
        best_viol = current_viol
        best_state = state.copy()
        
        if current_viol == 0:
            return state.tolist()
        
        # Simulated Annealing
        T = 1.0
        cooling_rate = 0.9999
        min_T = 0.01
        
        start_time = time.time()
        
        # Precompute edge indices for quick access
        # We need to update adj efficiently when changing one edge.
        
        while time.time() < time_limit:
            # Pick a random edge
            idx = random.randint(0, total_edges - 1)
            u, v = edges[idx]
            old_c = state[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Update adj
            adj[old_c][u][v] = 0
            adj[old_c][v][u] = 0
            adj[new_c][u][v] = 1
            adj[new_c][v][u] = 1
            
            # Calculate delta in violations
            # This is the tricky part.
            # Changing edge (u,v) from c1 to c2 affects violations involving u and v.
            # Specifically, it changes the common neighbor counts for pairs (u, w) and (v, w).
            # Let's just recompute total violations?
            # O(n^3) per step? 29^3 = 24389. 
            # 100k steps * 24k = 2.4e9 ops. Too slow in Python/NumPy overhead?
            # NumPy matmul is fast. 5 matmuls of 29x29.
            # 5 * (29*29*29) ~ 120k flops.
            # 100k steps * 120k flops = 12e9 flops.
            # Modern CPU does ~10-50 GFLOPS. So 12e9 is ~0.1-1s.
            # So we can do many steps.
            
            new_viol = count_violations(adj)
            delta = new_viol - current_viol
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_viol = new_viol
                state[idx] = new_c
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_state = state.copy()
                    if best_viol == 0:
                        return state.tolist()
            else:
                # Revert
                adj[new_c][u][v] = 0
                adj[new_c][v][u] = 0
                adj[old_c][u][v] = 1
                adj[old_c][v][u] = 1
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        if best_viol == 0:
            return best_state.tolist()
        return None

    # Execute search
    # Try n from 26 down to 12
    # We know 26 is possible.
    # SA might find 26 quickly if initialized well?
    # Random init is likely bad for n=26.
    # We can try to use a heuristic init?
    # Or just run SA with restarts.
    
    for n in range(26, 11, -1):
        if time.time() > deadline - 5:
            break
        
        t_alloc = min(30.0, deadline - time.time() - 2.0)
        if t_alloc <= 0:
            break
        
        # Try a few restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 2:
                break
            t_step = t_alloc / 5.0
            random.seed(restart * 100 + n)
            res = solve_n_sa(n, time.time() + t_step)
            if res:
                if is_valid(n, res):
                    best_n = n
                    best_colors = res
                    found = True
                    break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found n={res['n']}")
