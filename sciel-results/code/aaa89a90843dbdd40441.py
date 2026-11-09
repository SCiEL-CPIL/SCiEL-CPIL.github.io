import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Known good solution for n=32
    def get_n32():
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

    # Try to find a solution for n=50 using Simulated Annealing
    n = 50
    num_edges = n * (n - 1) // 2
    
    # Initialize colors
    # Use a structured initialization: extend n=32 solution
    n32_sol = get_n32()
    colors32 = n32_sol['colors']
    
    # Map edge (i,j) to index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)
        
    # Wait, the standard lexicographic order index for (i,j) with i<j is:
    # sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    # Or simply:
    # idx = 0
    # for k in range(i):
    #   idx += n - 1 - k
    # idx += j - i - 1
    
    # Let's precompute indices for speed
    edge_indices = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_indices[(i, j)] = idx
            idx += 1
            
    colors = [0] * num_edges
    
    # Initialize first 32 vertices with known solution
    for i in range(32):
        for j in range(i + 1, 32):
            colors[edge_indices[(i, j)]] = colors32[edge_indices[(i, j)] % len(colors32)] # Careful with mapping
            # Actually, the n32 colors list is in lex order for n=32.
            # We need to map the edge (i,j) in K50 to the corresponding edge in K32.
            # If i,j < 32, it's the same edge.
            
    # Re-do initialization properly
    colors = [0] * num_edges
    # Fill edges within 0..31
    for i in range(32):
        for j in range(i + 1, 32):
            # Find index in K32
            idx32 = i * 32 - i * (i + 1) // 2 + (j - i - 1)
            # Find index in K50
            idx50 = edge_indices[(i, j)]
            colors[idx50] = colors32[idx32]
            
    # Fill remaining edges randomly
    random.seed(42)
    for i in range(32, n):
        for j in range(i + 1, n):
            colors[edge_indices[(i, j)]] = random.randint(0, 3)
        for j in range(32):
            colors[edge_indices[(j, i)]] = random.randint(0, 3)

    # Precompute neighbors for each edge?
    # For each edge (u,v), the triangles are (u,v,w) for all w != u,v.
    # We need to update the count of monochromatic triangles.
    
    # Initial count
    def count_mono_triangles(c):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = c[edge_indices[(i, j)]]
                for k in range(j + 1, n):
                    if c[edge_indices[(i, k)]] == c_ij and c[edge_indices[(j, k)]] == c_ij:
                        count += 1
        return count

    # This initial count is O(n^3). For n=50, 19600 checks. Fast enough.
    current_count = count_mono_triangles(colors)
    
    best_count = current_count
    best_colors = colors[:]
    
    if current_count == 0:
        return {'n': n, 'colors': colors}

    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick random edge
        e = random.randint(0, num_edges - 1)
        # Find u, v for e
        # We can precompute edge endpoints
        # Let's precompute endpoints
        # ...
        
        # To avoid precomputing endpoints for 1225 edges (trivial), we can just iterate
        # But for speed, let's precompute
        # endpoints = []
        # for i in range(n):
        #   for j in range(i+1, n):
        #     endpoints.append((i,j))
        
        u, v = endpoints[e]
        old_color = colors[e]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
            
        # Calculate delta
        delta = 0
        for w in range(n):
            if w == u or w == v:
                continue
            cuw = colors[edge_indices[(u, w)]]
            cvw = colors[edge_indices[(v, w)]]
            
            # Old contribution
            if cuw == old_color and cvw == old_color:
                delta += 1
            # New contribution
            if cuw == new_color and cvw == new_color:
                delta -= 1
                
        if delta < 0 or random.random() < math.exp(-delta / T):
            colors[e] = new_color
            current_count += delta
            if current_count < best_count:
                best_count = current_count
                best_colors = colors[:]
                if best_count == 0:
                    return {'n': n, 'colors': best_colors}
        T *= cooling
        if T < 0.01:
            T = 0.01

    # If we didn't find 0, return the best valid one we know (n=32)
    # Or if we found a valid one with n<50? We only searched for n=50.
    # We can return the n=32 solution.
    return get_n32()

# Precompute endpoints for SA
endpoints = []
for i in range(50):
    for j in range(i + 1, 50):
        endpoints.append((i, j))
