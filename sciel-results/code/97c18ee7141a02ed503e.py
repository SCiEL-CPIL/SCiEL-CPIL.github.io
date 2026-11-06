"""
Five-colourings of complete graphs without monochromatic C4 (R_5(C4)).
Uses a fast C4 counter with NumPy and a simulated annealing / local search
to maximize n. Known algebraic construction gives n=26; this attempts to
find n=27 or n=28.
"""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit

    def idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + j - i - 1

    def fast_count_c4(colors, n):
        """Count C4s using NumPy for speed."""
        num_edges = n * (n - 1) // 2
        colors = np.array(colors)
        # Build adjacency matrices for each color
        adj = [np.zeros((n, n), dtype=np.uint8) for _ in range(5)]
        for e in range(num_edges):
            # Reconstruct (i, j) from e
            # e = i*n - i*(i+1)//2 + j - i - 1
            # Solve for i: i*(2n-i-1)/2 <= e < (i+1)*(2n-i-2)/2
            # Use precomputed mapping for speed
            pass
        # This is too slow. Let's precompute edge-to-vertex mapping.
        # Actually, let's build the edge mapping once.
        edge_map = np.empty((num_edges, 2), dtype=np.int32)
        e = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[e] = [i, j]
                e += 1
        # Now for each color, build adjacency
        adj = [np.zeros((n, n), dtype=np.uint8) for _ in range(5)]
        for e in range(num_edges):
            c = colors[e]
            i, j = edge_map[e]
            adj[c][i, j] = 1
            adj[c][j, i] = 1
        # Count C4s: for each color, count pairs of vertices with >=2 common neighbors
        # A C4 is counted as: for each pair (u,v), if |N(u) ∩ N(v)| >= 2, that's a C4
        # Actually, each C4 {a,b,c,d} is counted 4 times (once for each pair of opposite vertices)
        # But we just need to count if any exist, so we can count the number of pairs with >=2 common neighbors
        total = 0
        for c in range(5):
            # For each pair (u, v), u < v, count common neighbors
            # common = adj[c][u] & adj[c][v], then count bits
            # Use matrix multiplication: (A @ A.T)[u,v] = number of common neighbors
            AA = adj[c] @ adj[c].T
            # For u < v, if AA[u,v] >= 2, there's a C4
            # But we need to exclude u==v and count only u < v
            # Actually, AA is symmetric. For u != v, AA[u,v] is the number of common neighbors.
            # A C4 exists if there are two vertices u, v with AA[u,v] >= 2
            # But wait: if AA[u,v] >= 2, then there are at least two common neighbors, 
            # which means there's at least one C4 involving u, v and those two neighbors.
            # But we might double count. Let's just count the number of pairs (u,v) with u < v and AA[u,v] >= 2.
            # Each C4 is counted once per pair of opposite vertices, so 2 times per C4? 
            # No: a C4 has 4 vertices a,b,c,d in cycle order. Opposite pairs are (a,c) and (b,d).
            # So each C4 contributes 2 to this count. But we just need to know if it's > 0.
            # For the objective, we can use the sum of max(0, AA[u,v] - 1) for u < v, 
            # or just the count of pairs with AA[u,v] >= 2.
            upper = np.triu_indices(n, 1)
            vals = AA[upper]
            # Count how many are >= 2
            total += np.sum(vals >= 2)
        return int(total)

    def is_valid(colors, n):
        return fast_count_c4(colors, n) == 0

    # Precompute edge map for various n
    def build_edge_map(n):
        num_edges = n * (n - 1) // 2
        edge_map = np.empty((num_edges, 2), dtype=np.int32)
        e = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[e] = [i, j]
                e += 1
        return edge_map

    # Fast C4 count using precomputed edge map
    def fast_count_c4_v2(colors, n, edge_map):
        num_edges = n * (n - 1) // 2
        colors = np.array(colors)
        adj = [np.zeros((n, n), dtype=np.uint8) for _ in range(5)]
        for e in range(num_edges):
            c = colors[e]
            i, j = edge_map[e]
            adj[c][i, j] = 1
            adj[c][j, i] = 1
        total = 0
        for c in range(5):
            AA = adj[c] @ adj[c].T
            upper = np.triu_indices(n, 1)
            vals = AA[upper]
            total += np.sum(vals >= 2)
        return int(total)

    # Try to find a valid coloring for decreasing n
    for target_n in [28, 27, 26, 25, 24]:
        if time.time() > deadline:
            break
        if target_n < 2:
            break
        n = target_n
        num_edges = n * (n - 1) // 2
        edge_map = build_edge_map(n)
        
        best_colors = None
        best_score = float('inf')
        
        # Multiple restarts with simulated annealing
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Initialize
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            score = fast_count_c4_v2(colors, n, edge_map)
            best_colors = colors.copy()
            best_score = score
            
            if score == 0:
                break
            
            # Simulated annealing
            T = 100.0
            cool_rate = 0.9999
            min_T = 0.1
            max_steps = 200000
            
            step = 0
            while T > min_T and step < max_steps:
                if time.time() > deadline:
                    break
                step += 1
                
                # Pick a random edge
                e = random.randint(0, num_edges - 1)
                old_c = colors[e]
                new_c = (old_c + 1) % 5
                colors[e] = new_c
                
                new_score = fast_count_c4_v2(colors, n, edge_map)
                
                if new_score < score:
                    score = new_score
                    if score < best_score:
                        best_score = score
                        best_colors = colors.copy()
                    if score == 0:
                        break
                else:
                    # Accept with probability exp(-(new_score - score) / T)
                    delta = new_score - score
                    if delta == 0:
                        pass  # always accept same score
                    else:
                        prob = np.exp(-delta / T)
                        if random.random() > prob:
                            colors[e] = old_c  # reject
                
                T *= cool_rate
            
            if score == 0:
                break
            if best_score == 0:
                break
        
        if best_score == 0:
            return {'n': n, 'colors': best_colors}
    
    # Fallback: n=11 seed construction
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
