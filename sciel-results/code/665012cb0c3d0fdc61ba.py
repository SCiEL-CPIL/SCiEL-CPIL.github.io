# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def _build_adj_lists(colors, n, color):
    """Build adjacency lists for a specific color."""
    adj = [[] for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == color:
                adj[i].append(j)
                adj[j].append(i)
            idx += 1
    return adj

def _count_c6_fast(adj):
    """Count C6s in graph defined by adjacency lists."""
    n = len(adj)
    count = 0
    for v0 in range(n):
        for v1 in adj[v0]:
            for v2 in adj[v1]:
                if v2 == v0: continue
                for v3 in adj[v2]:
                    if v3 in (v0, v1): continue
                    for v4 in adj[v3]:
                        if v4 in (v0, v1, v2): continue
                        excluded = {v0, v1, v2, v3, v4}
                        # Optimize intersection check
                        list_a = adj[v4]
                        list_b = adj[v0]
                        # Check if any v5 in list_a is in list_b and not in excluded
                        # Convert smaller to set for lookup? 
                        # For small lists, linear scan might be faster due to overhead.
                        # But sets are O(1) lookup.
                        set_b = set(list_b)
                        for v5 in list_a:
                            if v5 in excluded: continue
                            if v5 in set_b:
                                count += 1
    return count

def construct():
    """Find a 4-coloring of K_n with no monochromatic C6, maximizing n."""
    random.seed(42)
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # Try from high n down
    for n in range(17, 1, -1):
        m = n * (n - 1) // 2
        
        # Initialize
        colors = [random.randint(0, 3) for _ in range(m)]
        
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * (2 * n - i - 1) // 2 + (j - i - 1)
            
        def get_energy(cols):
            total = 0
            for c in range(4):
                adj = _build_adj_lists(cols, n, c)
                total += _count_c6_fast(adj)
            return total

        current_temp = 100.0
        cooling = 0.99
        min_temp = 0.1
        best_energy = float('inf')
        best_colors = colors[:]
        
        current_energy = get_energy(colors)
        best_energy = current_energy
        best_colors = colors[:]
        
        if best_energy == 0:
            return {'n': n, 'colors': best_colors}
            
        while current_temp > min_temp:
            if time.time() - start_time > time_limit:
                break
                
            # Pick random edge
            idx = random.randint(0, m - 1)
            old_color = colors[idx]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
                
            colors[idx] = new_color
            new_energy = get_energy(colors)
            
            delta = new_energy - current_energy
            if delta < 0 or random.random() < np.exp(-delta / current_temp):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
                    if best_energy == 0:
                        return {'n': n, 'colors': best_colors}
            else:
                colors[idx] = old_color
                
            current_temp *= cooling
            
        if best_energy == 0:
            return {'n': n, 'colors': best_colors}
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
# EVOLVE-BLOCK-END
