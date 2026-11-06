import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Start with the known n=32 coloring from the seed program
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
    
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    best_n = 32
    best_colors = colors_32[:]
    
    # Try to extend to n=33, 34, ... using local search
    # We'll try to add vertices one at a time and fix violations
    
    def count_mono_triangles(colors, n):
        """Count monochromatic triangles in a 4-coloring of K_n."""
        # Build adjacency per color
        # For each color c, count triangles
        # Use bitsets for efficiency
        if n <= 32:
            adj = [np.zeros(n, dtype=np.uint8) for _ in range(4)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] = 1
                    adj[c][j] = 1
                    idx += 1
            count = 0
            for c in range(4):
                for i in range(n):
                    neighbors = adj[c][i]
                    # Count edges among neighbors
                    count += sum(neighbors[i+1:]) * 0  # This is O(n^3)
            # Better: use matrix multiplication
            count = 0
            for c in range(4):
                A = adj[c].astype(np.int64)
                np.fill_diagonal(A, 0)
                # Number of triangles = trace(A^3) / 6
                count += np.trace(A @ A @ A) // 6
            return count
        else:
            # For larger n, use a different approach
            adj = [np.zeros((n, n), dtype=np.uint8) for _ in range(4)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i, j] = 1
                    adj[c][j, i] = 1
                    idx += 1
            count = 0
            for c in range(4):
                A = adj[c].astype(np.int64)
                np.fill_diagonal(A, 0)
                count += np.trace(A @ A @ A) // 6
            return count
    
    def get_edges_for_vertex(colors, n, v):
        """Get the color indices for all edges incident to vertex v."""
        edges = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if i == v:
                    edges[j] = idx
                elif j == v:
                    edges[i] = idx
                idx += 1
        return edges
    
    def set_edge_color(colors, n, i, j, c):
        """Set the color of edge (i,j) in the colors list."""
        if i > j:
            i, j = j, i
        # Calculate index: sum of (n - 1 - k) for k < i, plus (j - i - 1)
        # Actually, lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        idx = i * (n - 1) - i * (i - 1) // 2 - i + (j - i - 1)
        # Let me recalculate: for vertex i, the edges are (i, i+1), ..., (i, n-1)
        # The number of edges before vertex i's edges is sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # Within vertex i's edges, edge (i,j) is at position (j - i - 1)
        idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        colors[idx] = c
        return idx
    
    def get_edge_color(colors, n, i, j):
        if i > j:
            i, j = j, i
        idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        return colors[idx]
    
    def mono_triangles_involving(colors, n, v):
        """Count monochromatic triangles involving vertex v."""
        count = 0
        for c in range(4):
            # Find all neighbors of v with color c
            neighbors = []
            for u in range(n):
                if u != v and get_edge_color(colors, n, v, u) == c:
                    neighbors.append(u)
            # Count edges of color c among neighbors
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if get_edge_color(colors, n, neighbors[i], neighbors[j]) == c:
                        count += 1
        return count
    
    def local_search(colors, n, max_iters=5000):
        """Try to eliminate all monochromatic triangles via local search."""
        colors = colors[:]
        # First, count initial triangles
        tri_count = count_mono_triangles(colors, n)
        
        if tri_count == 0:
            return colors, True
        
        # Simulated annealing
        T = 10.0
        cooling = 0.9995
        rng = random.Random(42)
        
        for it in range(max_iters):
            if tri_count == 0:
                return colors, True
            
            # Pick a random edge
            i = rng.randint(0, n - 1)
            j = rng.randint(i + 1, n - 1)
            old_c = get_edge_color(colors, n, i, j)
            
            # Try changing to a different color
            new_c = rng.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Calculate delta in triangle count
            # This is expensive; do a quick check
            set_edge_color(colors, n, i, j, new_c)
            new_tri = count_mono_triangles(colors, n)
            delta = new_tri - tri_count
            
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                tri_count = new_tri
            else:
                set_edge_color(colors, n, i, j, old_c)
            
            T *= cooling
        
        return colors, tri_count == 0
    
    # Strategy: try to build up from n=32
    # For each target n from 33 to 50:
    #   1. Start with the best coloring for n-1
    #   2. Add a new vertex with random colors
    #   3. Run local search to fix violations
    #   4. If successful, continue; otherwise try different random seeds
    
    current_n = 32
    current_colors = best_colors[:]
    
    for target_n in range(33, 51):
        if time.time() > deadline - 10:
            break
        
        success = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Start from current coloring
            trial_colors = current_colors[:]
            
            # Add edges for new vertex (target_n - 1)
            new_v = current_n
            for u in range(current_n):
                c = random.randint(0, 3)
                trial_colors.append(c)
            
            # Now run local search
            trial_colors, ok = local_search(trial_colors, target_n, max_iters=2000)
            
            if ok:
                success = True
                current_n = target_n
                current_colors = trial_colors
                best_n = target_n
                best_colors = trial_colors[:]
                break
        
        if not success:
            # Try a few more attempts with different random seeds
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                
                trial_colors = current_colors[:]
                new_v = current_n
                for u in range(current_n):
                    c = random.randint(0, 3)
                    trial_colors.append(c)
                
                trial_colors, ok = local_search(trial_colors, target_n, max_iters=3000)
                
                if ok:
                    success = True
                    current_n = target_n
                    current_colors = trial_colors
                    best_n = target_n
                    best_colors = trial_colors[:]
                    break
        
        if not success:
            # Can't extend further, break
            break
    
    # If we're still at n=32, try a completely different approach:
    # Start from scratch with random initialization and local search for larger n
    if best_n == 32:
        for target_n in [40, 45, 50]:
            if time.time() > deadline - 10:
                break
            
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                
                # Random initialization
                colors = [random.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
                
                # Run local search
                colors, ok = local_search(colors, target_n, max_iters=5000)
                
                if ok:
                    best_n = target_n
                    best_colors = colors
                    break
    
    return {'n': best_n, 'colors': best_colors}
