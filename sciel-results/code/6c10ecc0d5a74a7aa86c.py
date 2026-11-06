import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    # Fallback: Seed program result
    def seed_construct():
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

    seed_result = seed_construct()
    
    # Try to find a valid coloring for n=50
    n = 50
    num_edges = n * (n - 1) // 2
    
    # Initialize colors
    # Use the seed for the first 32 vertices, random for the rest
    colors = [0] * num_edges
    
    # Helper to get/set edge index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * n - i * (i + 1) // 2 + j - i - 1
        # Formula: sum_{k=0}^{i-1} (n-k-1) + (j-i-1)
        # Actually, lexicographic order (i,j) with i<j.
        # Index of (i,j) is sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
        # Let's use a simpler mapping or precomputed list.
    
    # Precompute edge indices for speed
    edge_indices = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_indices[(i, j)] = idx
            idx += 1
            
    # Initialize with seed for first 32
    seed_colors = seed_result['colors']
    s_idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            colors[edge_indices[(i, j)]] = seed_colors[s_idx]
            s_idx += 1
            
    # Random for others
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_indices: continue
            if i >= 32 or j >= 32:
                colors[edge_indices[(i, j)]] = random.randint(0, 3)
                
    # Data structure for neighbors
    # neighbors[color][node] = set of neighbors
    neighbors = [[set() for _ in range(n)] for _ in range(4)]
    
    def build_neighbors():
        for c in range(4):
            for u in range(n):
                neighbors[c][u].clear()
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_indices[(i, j)]]
                neighbors[c][i].add(j)
                neighbors[c][j].add(i)
                
    build_neighbors()
    
    def count_triangles():
        total = 0
        for c in range(4):
            for u in range(n):
                for v in neighbors[c][u]:
                    if u < v:
                        # Count common neighbors
                        total += len(neighbors[c][u].intersection(neighbors[c][v]))
        return total // 3 # Each triangle counted 3 times? 
        # Wait, intersection counts w. Triangle (u,v,w). 
        # Pairs (u,v), (u,w), (v,w).
        # For (u,v), w is in intersection.
        # For (u,w), v is in intersection.
        # For (v,w), u is in intersection.
        # So each triangle is counted 3 times in the sum over edges.
        # But we only iterate u < v.
        # So for triangle (u,v,w) with u<v<w:
        # Edge (u,v): w in N(u) cap N(v). Counted.
        # Edge (u,w): v in N(u) cap N(w). Counted.
        # Edge (v,w): u in N(v) cap N(w). Counted.
        # So yes, 3 times.
        
    # Actually, calculating intersection every time is O(n^2) per edge flip?
    # Intersection of two sets of size ~n/4 is fast.
    
    current_triangles = count_triangles()
    best_triangles = current_triangles
    best_colors = colors[:]
    
    # Local Search
    t = 1.0
    cooling = 0.9999
    min_t = 0.01
    
    while time.time() < deadline:
        if current_triangles == 0:
            break
        if current_triangles < best_triangles:
            best_triangles = current_triangles
            best_colors = colors[:]
            
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        # Find u, v from e_idx? 
        # This is slow. Better to pick random u, v.
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v: continue
        if u > v: u, v = v, u
        
        idx = edge_indices[(u, v)]
        old_c = colors[idx]
        new_c = random.randint(0, 3)
        if old_c == new_c: continue
        
        # Calculate delta
        # Triangles lost from old_c
        lost = len(neighbors[old_c][u].intersection(neighbors[old_c][v]))
        # Triangles gained in new_c
        gained = len(neighbors[new_c][u].intersection(neighbors[new_c][v]))
        
        delta = gained - lost
        
        # Accept if delta < 0 or with probability exp(-delta/t)
        accept = False
        if delta < 0:
            accept = True
        else:
            if random.random() < pow(2.71828, -delta / max(t, 0.001)):
                accept = True
                
        if accept:
            # Update structures
            # Remove from old
            neighbors[old_c][u].remove(v)
            neighbors[old_c][v].remove(u)
            # Add to new
            neighbors[new_c][u].add(v)
            neighbors[new_c][v].add(u)
            colors[idx] = new_c
            current_triangles += delta
            t *= cooling
        else:
            t *= cooling
            
    if current_triangles == 0:
        # Verify
        # (Optional: full verification)
        return {'n': n, 'colors': colors}
    else:
        # Return best found if valid? No, best found might not be valid.
        # The task requires a valid coloring.
        # If we didn't find 0 triangles, we must return a valid smaller one.
        # The seed is valid.
        return seed_result
