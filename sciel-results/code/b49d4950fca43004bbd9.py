"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def greenwood_gleason_32():
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
        return colors
    
    def count_mono_triangles(n, colors):
        edge_to_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_to_idx[(i,j)] = idx
                idx += 1
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c_ij = colors[edge_to_idx[(i,j)]]
                for k in range(j+1, n):
                    if colors[edge_to_idx[(i,k)]] == c_ij and colors[edge_to_idx[(j,k)]] == c_ij:
                        count += 1
        return count
    
    def is_valid(n, colors):
        return count_mono_triangles(n, colors) == 0
    
    def sa_search(n, max_iter=500000, max_time=105.0):
        if time.time() > deadline:
            return None
        
        m = n * (n-1) // 2
        colors = np.random.randint(0, 4, size=m)
        
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_idx[(i,j)] = idx
                idx += 1
        
        triangles = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    triangles.append((edge_idx[(i,j)], edge_idx[(i,k)], edge_idx[(j,k)]))
        
        n_triangles = len(triangles)
        tri_edges = np.array(triangles)
        
        # For each edge, store which triangles it's in
        edge_to_tris = [[] for _ in range(m)]
        for t in range(n_triangles):
            for e in tri_edges[t]:
                edge_to_tris[e].append(t)
        
        # Precompute triangle edge arrays for fast access
        tri_e1 = tri_edges[:, 0]
        tri_e2 = tri_edges[:, 1]
        tri_e3 = tri_edges[:, 2]
        
        def count_mono():
            c1 = colors[tri_e1]
            c2 = colors[tri_e2]
            c3 = colors[tri_e3]
            return int(np.sum((c1 == c2) & (c2 == c3)))
        
        energy = count_mono()
        best_energy = energy
        best_colors = colors.copy()
        
        T = 5.0
        cooling = 0.9997
        
        for it in range(max_iter):
            if time.time() > max_time:
                break
            
            edge = np.random.randint(0, m)
            old_color = colors[edge]
            new_color = (old_color + 1 + np.random.randint(0, 3)) % 4
            
            # Compute delta E
            affected = edge_to_tris[edge]
            if not affected:
                delta = 0
            else:
                aff_arr = np.array(affected)
                a1 = colors[tri_e1[aff_arr]]
                a2 = colors[tri_e2[aff_arr]]
                a3 = colors[tri_e3[aff_arr]]
                
                # Old mono status
                old_mono = (a1 == a2) & (a2 == a3)
                
                # New colors
                b1 = a1.copy()
                b2 = a2.copy()
                b3 = a3.copy()
                # Update the flipped edge
                b1[tri_e1[aff_arr] == edge] = new_color
                b2[tri_e2[aff_arr] == edge] = new_color
                b3[tri_e3[aff_arr] == edge] = new_color
                
                new_mono = (b1 == b2) & (b2 == b3)
                delta = int(np.sum(new_mono) - np.sum(old_mono))
            
            new_energy = energy + delta
            if new_energy < best_energy:
                best_energy = new_energy
                best_colors = colors.copy()
            
            if new_energy < energy or (delta > 0 and np.random.random() < np.exp(-delta / T)):
                colors[edge] = new_color
                energy = new_energy
            
            T *= cooling
        
        if best_energy == 0:
            return best_colors.tolist()
        return None
    
    # Try SA for n=50
    result = sa_search(50, max_iter=500000, max_time=deadline - 5.0)
    if result is not None:
        return {'n': 50, 'colors': result}
    
    # Try n=40
    if time.time() < deadline - 10.0:
        result = sa_search(40, max_iter=300000, max_time=deadline - 5.0)
        if result is not None:
            return {'n': 40, 'colors': result}
    
    # Try n=32
    if time.time() < deadline - 5.0:
        colors_32 = greenwood_gleason_32()
        if is_valid(32, colors_32):
            return {'n': 32, 'colors': colors_32}
    
    return {'n': 1, 'colors': []}
# EVOLVE-BLOCK-END
