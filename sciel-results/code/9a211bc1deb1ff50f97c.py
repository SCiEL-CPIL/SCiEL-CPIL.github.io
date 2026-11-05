import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140
    
    def count_violations(n, colors):
        if n < 4:
            return 0
        m = n * (n - 1) // 2
        if len(colors) != m:
            return float('inf')
        
        colors_arr = np.array(colors, dtype=np.int8)
        total_viol = 0
        
        for c in range(5):
            # Build adjacency matrix for color c
            adj = np.zeros((n, n), dtype=np.int8)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors_arr[idx] == c:
                        adj[i, j] = 1
                        adj[j, i] = 1
                    idx += 1
            
            # Count common neighbors
            # common[i, j] is number of common neighbors
            common = adj @ adj
            np.fill_diagonal(common, 0)
            
            # Count pairs (i, j) with i < j and common[i, j] >= 2
            iu = np.triu_indices(n, 1)
            total_viol += int(np.sum(common[iu] >= 2))
            
        return total_viol

    def local_search(n, colors, max_iter=20000, seed=0, dl=None):
        rng = random.Random(seed)
        colors = list(colors)
        m = n * (n - 1) // 2
        if m == 0:
            return colors, 0
            
        current_viol = count_violations(n, colors)
        best_colors = colors[:]
        best_viol = current_viol
        
        for it in range(max_iter):
            if dl and time.time() > dl - 2:
                break
            if best_viol == 0:
                break
                
            # Pick a random edge to change
            idx = rng.randint(0, m - 1)
            old_c = colors[idx]
            new_c = rng.randint(0, 4)
            
            if new_c == old_c:
                continue
                
            colors[idx] = new_c
            new_viol = count_violations(n, colors)
            
            if new_viol < current_viol:
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_colors = colors[:]
            else:
                colors[idx] = old_c
                
        return best_colors, best_viol

    # Try to find a valid coloring for n=26
    # Use multiple random restarts
    best_n = 1
    best_colors = []
    
    for target_n in [26, 25, 24, 23, 22]:
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            m = target_n * (target_n - 1) // 2
            init_colors = [random.randint(0, 4) for _ in range(m)]
            
            # Run local search
            res_colors, viol = local_search(target_n, init_colors, max_iter=10000, seed=restart, dl=deadline)
            
            if viol == 0:
                best_n = target_n
                best_colors = res_colors
                found = True
                break
                
        if found:
            break
            
    # If n=26 not found, try structured init for smaller n or fallback
    if best_n < 26:
        # Try a simple structured approach for n=25 just in case
        if time.time() < deadline - 10:
             for target_n in [25, 24]:
                 if time.time() > deadline - 5:
                     break
                 m = target_n * (target_n - 1) // 2
                 # Structured: color by (j-i) % 5
                 init_colors = []
                 for i in range(target_n):
                     for j in range(i + 1, target_n):
                         init_colors.append((j - i) % 5)
                 
                 res_colors, viol = local_search(target_n, init_colors, max_iter=5000, seed=123, dl=deadline)
                 if viol == 0:
                     best_n = target_n
                     best_colors = res_colors
                     break

    if best_n == 1:
        # Fallback to n=1 which is always valid
        return {'n': 1, 'colors': []}
        
    return {'n': best_n, 'colors': best_colors}
