"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_n = 0
    best_colors = None

    def make_edge_index(n):
        """Return a function to compute flat index for edge (i,j), i<j."""
        # Precompute prefix sums
        # index(i,j) = sum_{k=0}^{i-1} (n-1-k) + (j - i - 1)
        # = i*(n-1) - i*(i-1)//2 + j - i - 1
        # Let's verify: for i=0: 0 + j - 0 - 1 = j-1. Edges (0,1)=0, (0,2)=1, ..., (0,n-1)=n-2. Correct.
        # For i=1: 1*(n-1) - 0 + j - 1 - 1 = n-1 + j - 2. Edges (1,2) = n-2, (1,3)=n-1, etc.
        # Actually let me just precompute a 2D array or use a formula.
        
        # Simpler: precompute start index for each row i
        starts = [0] * (n + 1)
        for i in range(n):
            starts[i] = i * (n - 1) - i * (i - 1) // 2
        # Wait, that's not right either. Let me just use:
        # number of edges with first vertex < i is sum_{k=0}^{i-1} (n-1-k)
        # = i*(n-1) - (i-1)*i//2
        
        def idx(i, j):
            # i < j
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        return idx

    def count_triangles_fast(colors, n, idx):
        """Count monochromatic triangles using numpy-like approach but pure Python for clarity."""
        # For efficiency, build adjacency per color
        # adj[c][v] = list of neighbors u such that edge(v,u) has color c
        # Then for each v and color c, count edges among adj[c][v] that also have color c
        # That's O(sum over v,c of deg^2) which is manageable for n<=52
        
        adj = [[[] for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx(i, j)]
                adj[c][i].append(j)
                adj[c][j].append(i)
        
        count = 0
        for c in range(4):
            for v in range(n):
                neighbors = adj[c][v]
                # For each pair of neighbors (a, b) with a < b, check if edge(a,b) has color c
                # and a, b are both in neighbors
                # Use a set for O(1) lookup
                nset = set(neighbors)
                for a in neighbors:
                    for b in neighbors:
                        if a < b and b in nset:  # always true since b is in neighbors
                            # Check if edge(a,b) has color c
                            if a < b:
                                if colors[idx(a, b)] == c:
                                    count += 1
        # Each triangle is counted 3 times (once per vertex)
        return count // 3

    def greedy_construct(n, seed):
        """Build a valid coloring of K_n by adding vertices one at a time."""
        rng = random.Random(seed)
        colors = {}  # (i,j) -> color
        idx = make_edge_index(n)
        
        for v in range(1, n):
            assigned = {}
            valid = True
            for u in range(v):
                valid_colors = []
                for c in range(4):
                    ok = True
                    for w in range(u):
                        if colors[(u, w)] == c and assigned.get((v, w), None) == c:
                            ok = False
                            break
                    if ok:
                        valid_colors.append(c)
                
                if not valid_colors:
                    valid = False
                    break
                
                c = rng.choice(valid_colors)
                assigned[(v, u)] = c
            
            if not valid:
                return None
            
            for (a, b), c in assigned.items():
                colors[(a, b)] = c
        
        color_list = [colors[(i, j)] for i in range(n) for j in range(i + 1, n)]
        return color_list

    def simulate_annealing(initial_colors, n, idx, max_iters, start_temp, end_temp, seed):
        """Simulated annealing to minimize triangle count."""
        rng = random.Random(seed)
        colors = list(initial_colors)
        cur_cost = count_triangles_fast(colors, n, idx)
        best_colors = colors[:]
        best_cost = cur_cost
        
        num_edges = n * (n - 1) // 2
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
            
            e = rng.randint(0, num_edges - 1)
            old_c = colors[e]
            new_c = rng.randint(0, 3)
            if new_c == old_c:
                continue
            colors[e] = new_c
            
            new_cost = count_triangles_fast(colors, n, idx)
            delta = new_cost - cur_cost
            
            if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_colors = colors[:]
            else:
                colors[e] = old_c
            
            if best_cost == 0:
                break
        
        return best_colors, best_cost

    # Try greedy construction for various n
    for target_n in [16, 20, 24, 28, 32, 36, 40, 44, 48, 50, 52]:
        if time.time() > deadline - 10:
            break
        found = False
        for seed in range(100):
            if time.time() > deadline - 10:
                break
            result = greedy_construct(target_n, seed)
            if result is not None:
                idx = make_edge_index(target_n)
                tri_count = count_triangles_fast(result, target_n, idx)
                if tri_count == 0:
                    if target_n > best_n:
                        best_n = target_n
                        best_colors = result
                    found = True
                    break
        if found and target_n >= 48:
            break  # We found a large one, stop
    
    # If we haven't found a large n, try SA from random initializations
    if best_n < 44:
        for target_n in [44, 48, 50, 52]:
            if time.time() > deadline - 15:
                break
            idx = make_edge_index(target_n)
            num_edges = target_n * (target_n - 1) // 2
            
            for seed in range(10):
                if time.time() > deadline - 15:
                    break
                rng = random.Random(seed)
                initial = [rng.randint(0, 3) for _ in range(num_edges)]
                best_col, best_cost = simulate_annealing(
                    initial, target_n, idx,
                    max_iters=30000, start_temp=10.0, end_temp=0.01, seed=seed
                )
                if best_cost == 0 and target_n > best_n:
                    best_n = target_n
                    best_colors = best_col
                    break
    
    # Final verification
    if best_colors is not None and best_n >= 1:
        idx = make_edge_index(best_n)
        tri_count = count_triangles_fast(best_colors, best_n, idx)
        if tri_count == 0:
            return {'n': best_n, 'colors': best_colors}
    
    # Fallback: return the seed program's result (n=32)
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
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
# EVOLVE-BLOCK-END
