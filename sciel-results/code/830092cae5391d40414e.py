import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    k4s = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    k4s.append((a, b, c, d))
    
    # Precompute edge indices for each K4
    # Edge index for (i, j) with i < j is i*n - i*(i+1)//2 + (j-i-1)
    # Or simpler: use a 2D array for colors
    # colors[i][j] for i < j
    
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)
        
    k4_edges = []
    for a, b, c, d in k4s:
        edges = [
            edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
            edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
        ]
        k4_edges.append(edges)

    def count_bad_k4s(colors):
        bad_count = 0
        for edges in k4_edges:
            c0 = colors[edges[0]]
            c1 = colors[edges[1]]
            c2 = colors[edges[2]]
            c3 = colors[edges[3]]
            c4 = colors[edges[4]]
            c5 = colors[edges[5]]
            # Check if at least 5 distinct
            # A K4 is bad if it uses <= 4 colors
            # This happens if there are at least 2 pairs of edges with the same color, 
            # or one color appears 3+ times, etc.
            # Simpler: just count distinct.
            distinct = len({c0, c1, c2, c3, c4, c5})
            if distinct < 5:
                bad_count += 1
        return bad_count

    def try_find_valid(K, time_limit):
        # colors is a list of length num_edges
        best_colors = None
        best_bad = float('inf')
        
        # Random initialization
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        current_bad = count_bad_k4s(colors)
        if current_bad == 0:
            return colors
        
        best_bad = current_bad
        best_colors = colors[:]
        
        t = 1.0
        cooling = 0.999
        min_t = 0.01
        
        while time.time() < time_limit:
            # Pick random edge
            e = random.randint(0, num_edges - 1)
            old_c = colors[e]
            new_c = random.randint(0, K - 1)
            if old_c == new_c:
                continue
            
            colors[e] = new_c
            new_bad = count_bad_k4s(colors)
            
            if new_bad < current_bad:
                current_bad = new_bad
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = colors[:]
                    if best_bad == 0:
                        return best_colors
            else:
                # Accept with probability exp(-delta / t)
                delta = new_bad - current_bad
                if delta > 0:
                    prob = np.exp(-delta / t)
                    if random.random() < prob:
                        current_bad = new_bad
                    else:
                        colors[e] = old_c
                else:
                    # Should not happen if delta > 0 check is correct, but handle anyway
                    current_bad = new_bad
            
            t *= cooling
            if t < min_t:
                t = min_t
                
            # Occasionally reset if stuck
            if random.random() < 0.001:
                colors = [random.randint(0, K - 1) for _ in range(num_edges)]
                current_bad = count_bad_k4s(colors)
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = colors[:]
                    if best_bad == 0:
                        return best_colors
                t = 1.0

        return best_colors if best_bad == 0 else None

    # Try K=14 first
    remaining = deadline - time.time()
    if remaining > 5:
        result = try_find_valid(14, deadline - 5)
        if result is not None:
            # Compact colors to minimize count
            used = set(result)
            mapping = {c: i for i, c in enumerate(sorted(used))}
            return [mapping[c] for c in result]
        
    # Try K=15
    remaining = deadline - time.time()
    if remaining > 5:
        result = try_find_valid(15, deadline - 5)
        if result is not None:
            used = set(result)
            mapping = {c: i for i, c in enumerate(sorted(used))}
            return [mapping[c] for c in result]
            
    # Fallback to baseline
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
