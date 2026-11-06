"""Geometric embedding approach: vertices on sphere, colors from dot-product quadrants."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = None
_START = None

def _now():
    return time.perf_counter()

def _remaining():
    global _DEADLINE, _START
    if _DEADLINE is None:
        _DEADLINE = _START + 110.0
    return _DEADLINE - _now()

def construct():
    global _DEADLINE, _START
    _START = _now()
    _DEADLINE = _START + 110.0
    
    def build_adj(n, colors):
        adj = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                adj[i][j] = colors[idx]
                adj[j][i] = colors[idx]
                idx += 1
        return adj
    
    def count_mono_triangles(adj, n):
        """Count monochromatic triangles in the coloring."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                for k in range(j + 1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        count += 1
        return count
    
    def find_mono_triangle(adj, n):
        """Find one monochromatic triangle, return (i,j,k) or None."""
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                for k in range(j + 1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        return (i, j, k)
        return None
    
    def assign_colors_from_vectors(V):
        """Assign colors based on dot products between vertex vectors.
        For each pair (i,j), compute dot = V[i]·V[j].
        Use the sign pattern of the 4 components of the difference vector
        to assign a color in {0,1,2,3}.
        """
        n = V.shape[0]
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = V[i] - V[j]
                # Use sign pattern of 4-dimensional difference
                signs = np.sign(diff)
                # Map sign pattern to color 0-3
                # Use first two significant sign bits
                if signs[0] >= 0:
                    if signs[1] >= 0:
                        c = 0
                    else:
                        c = 1
                else:
                    if signs[2] >= 0:
                        c = 2
                    else:
                        c = 3
                colors.append(c)
        return colors
    
    def geometric_local_search(V, adj, n, max_iter=2000):
        """Optimize vertex positions to minimize monochromatic triangles."""
        best_count = count_mono_triangles(adj, n)
        best_V = V.copy()
        best_adj = [row[:] for row in adj]
        
        T = 1.0
        cooling = 0.995
        
        for it in range(max_iter):
            if _remaining() < 1.0:
                break
            
            # Pick a random vertex and perturb it
            vi = random.randint(0, n - 1)
            delta = np.random.normal(0, 0.1, size=4)
            V_new = V.copy()
            V_new[vi] = V[vi] + delta
            # Renormalize
            norm = np.linalg.norm(V_new[vi])
            if norm > 1e-10:
                V_new[vi] /= norm
            
            # Recompute colors for edges incident to vi
            new_adj = [row[:] for row in adj]
            new_colors_list = []
            for j in range(n):
                if j == vi:
                    continue
                diff = V_new[vi] - V_new[j]
                signs = np.sign(diff)
                if signs[0] >= 0:
                    if signs[1] >= 0:
                        c = 0
                    else:
                        c = 1
                else:
                    if signs[2] >= 0:
                        c = 2
                    else:
                        c = 3
                new_adj[vi][j] = c
                new_adj[j][vi] = c
            
            new_count = count_mono_triangles(new_adj, n)
            
            delta_count = new_count - best_count
            if delta_count <= 0 or random.random() < np.exp(-delta_count / T):
                V = V_new
                adj = new_adj
                if new_count < best_count:
                    best_count = new_count
                    best_V = V_new.copy()
                    best_adj = [row[:] for row in new_adj]
            
            T *= cooling
        
        return best_count, best_adj, best_V
    
    def geometric_greedy_extend(V, adj, n, max_tries=100):
        """Try to extend by adding a new vertex at a random position on the sphere."""
        if _remaining() < 0.5:
            return None
        
        for attempt in range(max_tries):
            if _remaining() < 0.5:
                break
            
            # Place new vertex at random position on 4-sphere
            new_pos = np.random.normal(0, 1, size=4)
            new_pos /= np.linalg.norm(new_pos)
            
            # Compute colors for edges from new vertex to all existing vertices
            new_edge_colors = []
            valid = True
            for i in range(n):
                diff = new_pos - V[i]
                signs = np.sign(diff)
                if signs[0] >= 0:
                    if signs[1] >= 0:
                        c = 0
                    else:
                        c = 1
                else:
                    if signs[2] >= 0:
                        c = 2
                    else:
                        c = 3
                new_edge_colors.append(c)
            
            # Check for monochromatic triangles involving the new vertex
            for i in range(n):
                for j in range(i + 1, n):
                    c_ij = adj[i][j]
                    if new_edge_colors[i] == c_ij and new_edge_colors[j] == c_ij:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return new_pos, new_edge_colors
        
        return None
    
    # Initialize with random points on 4-sphere
    best_n = 0
    best_adj = None
    
    for target_n in [32, 40, 48, 50]:
        if _remaining() < 3.0:
            break
        
        # Multiple restarts
        best_for_n = None
        best_count_for_n = float('inf')
        
        for restart in range(3):
            if _remaining() < 2.0:
                break
            
            # Random initialization on 4-sphere
            V = np.random.normal(0, 1, size=(target_n, 4))
            for i in range(target_n):
                V[i] /= np.linalg.norm(V[i])
            
            colors = assign_colors_from_vectors(V)
            adj = build_adj(target_n, colors)
            count = count_mono_triangles(adj, target_n)
            
            if count < best_count_for_n:
                best_count_for_n = count
                best_for_n = (count, adj, V.copy())
            
            # Geometric local search
            count, adj, V = geometric_local_search(V, adj, target_n, max_iter=1000)
            
            if count < best_count_for_n:
                best_count_for_n = count
                best_for_n = (count, adj, V.copy())
            
            if best_count_for_n == 0:
                break
        
        if best_for_n is not None and best_for_n[0] == 0:
            # Valid coloring found
            if target_n > best_n:
                best_n = target_n
                best_adj = best_for_n[1]
        
        # If we got a valid smaller one, try to extend
        if best_n > 0 and best_adj is not None and target_n <= 40:
            if _remaining() < 2.0:
                break
            
            # Try to extend from best_n
            cur_n = best_n
            cur_adj = [row[:] for row in best_adj]
            # Reconstruct V from adj is not straightforward, so use adj-based extension
            # Use a simpler extension: try random color assignments for new vertex
            
            while cur_n < 62:
                if _remaining() < 1.0:
                    break
                
                extended = False
                for _ in range(200):
                    if _remaining() < 0.5:
                        break
                    
                    new_colors = [random.randint(0, 3) for _ in range(cur_n)]
                    valid = True
                    for i in range(cur_n):
                        for j in range(i + 1, cur_n):
                            c_ij = cur_adj[i][j]
                            if new_colors[i] == c_ij and new_colors[j] == c_ij:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        new_n = cur_n + 1
                        new_adj = [[0] * new_n for _ in range(new_n)]
                        for i in range(cur_n):
                            for j in range(cur_n):
                                new_adj[i][j] = cur_adj[i][j]
                            new_adj[i][cur_n] = new_colors[i]
                            new_adj[cur_n][i] = new_colors[i]
                        new_adj[cur_n][cur_n] = 0
                        
                        cur_n = new_n
                        cur_adj = new_adj
                        extended = True
                        break
                
                if not extended:
                    break
            
            if cur_n > best_n:
                best_n = cur_n
                best_adj = cur_adj
    
    # Fallback: use the known 32-vertex coloring if nothing better found
    if best_n == 0 or best_adj is None:
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
        
        base_colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                base_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        
        best_n = 32
        best_adj = build_adj(32, base_colors)
    
    # Convert to list
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(best_adj[i][j])
    
    return {'n': best_n, 'colors': colors}
# EVOLVE-BLOCK-END
