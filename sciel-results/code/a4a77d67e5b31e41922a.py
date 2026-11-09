import time
import random
import numpy as np

def construct():
    """
    Uses bitmask-based simulated annealing to find a 4-coloring of K_n
    with no monochromatic triangles. Uses efficient bit operations for
    triangle counting and a temperature schedule for escape from local minima.
    """
    start_time = time.time()
    time_limit = 100.0
    
    def edge_index(n, i, j):
        """Get flat index for edge (i,j) where i < j."""
        return i * n - i * (i + 1) // 2 + (j - i - 1)
    
    def count_mono_triangles_bitmask(colors, n):
        """
        Count monochromatic triangles using bitmasks for efficiency.
        For each color c, build adjacency bitmask for each vertex.
        A triangle (i,j,k) in color c exists if j is in adj_c[i] and k is in adj_c[i] and k is in adj_c[j].
        """
        # Build adjacency bitmasks for each color
        adj = [0] * (4 * n)
        
        for i in range(n):
            for j in range(i + 1, n):
                idx = edge_index(n, i, j)
                c = colors[idx]
                adj[c * n + i] |= (1 << j)
                adj[c * n + j] |= (1 << i)
        
        count = 0
        for c in range(4):
            base = c * n
            for i in range(n):
                # For vertex i, get all neighbors in color c
                neighbors_i = adj[base + i]
                # For each pair of neighbors (j, k) of i in color c,
                # check if edge (j,k) is also color c
                # Equivalent: count triangles where i is the smallest vertex
                # to avoid double counting
                for j in range(i + 1, n):
                    if not (neighbors_i >> j) & 1:
                        continue
                    # Check if k > j and (j,k) is in color c
                    # neighbors_j = adj[base + j]
                    # Common neighbors of i and j in color c, with index > j
                    common = neighbors_i & adj[base + j]
                    # Mask to only consider vertices > j
                    mask = ((1 << n) - 1) ^ ((1 << (j + 1)) - 1)
                    common = common & mask
                    count += bin(common).count('1')
        return count
    
    def get_color(colors, n, i, j):
        if i > j:
            i, j = j, i
        return colors[edge_index(n, i, j)]
    
    def set_color(colors, n, i, j, c):
        if i > j:
            i, j = j, i
        colors[edge_index(n, i, j)] = c
    
    def find_one_mono_triangle(colors, n):
        """Find one monochromatic triangle, return (i,j,k,c) or None."""
        # Build adjacency bitmasks
        adj = [0] * (4 * n)
        for i in range(n):
            for j in range(i + 1, n):
                idx = edge_index(n, i, j)
                c = colors[idx]
                adj[c * n + i] |= (1 << j)
                adj[c * n + j] |= (1 << i)
        
        for c in range(4):
            base = c * n
            for i in range(n):
                neighbors_i = adj[base + i]
                for j in range(i + 1, n):
                    if not (neighbors_i >> j) & 1:
                        continue
                    common = neighbors_i & adj[base + j]
                    mask = ((1 << n) - 1) ^ ((1 << (j + 1)) - 1)
                    common = common & mask
                    if common:
                        # Get lowest set bit
                        k = (common & -common).bit_length() - 1
                        return (i, j, k, c)
        return None
    
    # Try to find valid colorings for decreasing n
    for target_n in [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() - start_time > time_limit:
            break
        
        found = False
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            n = target_n
            m = n * (n - 1) // 2
            
            # Structured initialization: use a block design
            # Divide vertices into 4 blocks of ~n/4, color within-block edges with distinct colors
            colors = [0] * m
            block_size = n // 4
            for i in range(n):
                for j in range(i + 1, n):
                    idx = edge_index(n, i, j)
                    bi = i // block_size
                    bj = j // block_size
                    if bi == bj:
                        # Same block: use color based on position
                        colors[idx] = (i + j) % 4
                    else:
                        # Different blocks: use color based on block difference
                        colors[idx] = (bi + bj) % 4
            
            # Simulated annealing
            current_triangles = count_mono_triangles_bitmask(colors, n)
            best_colors = colors[:]
            best_triangles = current_triangles
            
            T = 1.0
            cooling_rate = 0.995
            min_T = 0.01
            max_iters = 50000
            
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                
                if current_triangles == 0:
                    break
                
                # Pick a random edge to flip
                e_idx = random.randint(0, m - 1)
                # Find (i,j) from e_idx
                # Solve: e_idx = i*n - i*(i+1)//2 + (j-i-1)
                # Approximate i
                i_approx = int((n - (1 + 8 * e_idx) ** 0.5) / 2)
                for i in range(max(0, i_approx - 2), min(n - 1, i_approx + 3)):
                    base = i * n - i * (i + 1) // 2
                    if base <= e_idx < base + (n - i - 1):
                        j = e_idx - base + i + 1
                        break
                else:
                    continue
                
                old_c = colors[e_idx]
                new_c = random.choice([c for c in range(4) if c != old_c])
                colors[e_idx] = new_c
                
                # Count triangles involving edge (i,j)
                # Only triangles containing edge (i,j) can change
                delta = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # Triangle (i,j,k)
                    c_ik = get_color(colors, n, i, k)
                    c_jk = get_color(colors, n, j, k)
                    
                    # Old triangle status
                    if old_c == c_ik == c_jk:
                        delta -= 1  # was mono, now not (unless new_c also makes it mono)
                    if new_c == c_ik == c_jk:
                        delta += 1  # now mono
                
                current_triangles += delta
                
                # Accept or reject
                if current_triangles < best_triangles:
                    best_triangles = current_triangles
                    best_colors = colors[:]
                
                if delta <= 0:
                    # Accept (improvement or neutral)
                    pass
                else:
                    # Reject with probability exp(-delta/T)
                    if random.random() > np.exp(-delta / T):
                        colors[e_idx] = old_c
                        current_triangles -= delta
                
                T *= cooling_rate
                if T < min_T:
                    T = min_T
            
            if best_triangles == 0:
                return {'n': n, 'colors': best_colors}
        
    # Fallback: return a valid small coloring
    # Use the known n=32 construction
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
