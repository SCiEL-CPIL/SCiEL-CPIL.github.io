import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 138  # safety margin
    
    def edge_to_ij(idx, n):
        """Convert edge index to (i, j) pair in lexicographic order."""
        # Vertex 0: edges to 1..n-1 (n-1 edges)
        # Vertex 1: edges to 2..n-1 (n-2 edges)
        # ...
        # Vertex i: edges to i+1..n-1 (n-1-i edges)
        # Cumulative before vertex i: sum_{k=0}^{i-1}(n-1-k) = i*(n-1) - i*(i-1)//2
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cum = mid * (n - 1) - mid * (mid - 1) // 2
            if cum <= idx:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        cum = i * (n - 1) - i * (i - 1) // 2
        j = i + 1 + (idx - cum)
        return i, j
    
    def count_4cycles_adj(adj_bits, n):
        total = 0
        for u in range(n):
            au = adj_bits[u]
            for v in range(u + 1, n):
                cn = (au & adj_bits[v]).bit_count()
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total
    
    def total_violations(adj_bits_list, n):
        total = 0
        for c in range(5):
            total += count_4cycles_adj(adj_bits_list[c], n)
        return total
    
    best_n = 1
    best_colors = None
    
    for target_n in range(2, 29):
        if time.time() > deadline:
            break
        
        found = False
        # More restarts for smaller n (easier), fewer for larger
        num_restarts = max(3, 20 - target_n // 2)
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            seed = restart * 7919 + target_n * 104729 + 1
            rng = random.Random(seed)
            n = target_n
            num_edges = n * (n - 1) // 2
            colors = [rng.randint(0, 4) for _ in range(num_edges)]
            
            # Build adjacency bitmasks
            adj_bits = [[0] * n for _ in range(5)]
            for idx in range(num_edges):
                i, j = edge_to_ij(idx, n)
                c = colors[idx]
                adj_bits[c][i] |= (1 << j)
                adj_bits[c][j] |= (1 << i)
            
            current_violations = total_violations(adj_bits, n)
            
            # Adaptive step budget
            if n <= 8:
                max_steps = 200000
            elif n <= 12:
                max_steps = 100000
            elif n <= 16:
                max_steps = 50000
            else:
                max_steps = 30000
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                if current_violations == 0:
                    found = True
                    break
                
                idx = rng.randint(0, num_edges - 1)
                i, j = edge_to_ij(idx, n)
                old_c = colors[idx]
                
                best_c = -1
                best_v = current_violations
                
                # Try all 4 other colors
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    
                    adj_bits[old_c][i] &= ~(1 << j)
                    adj_bits[old_c][j] &= ~(1 << i)
                    adj_bits[new_c][i] |= (1 << j)
                    adj_bits[new_c][j] |= (1 << i)
                    
                    v = total_violations(adj_bits, n)
                    
                    adj_bits[old_c][i] |= (1 << j)
                    adj_bits[old_c][j] |= (1 << i)
                    adj_bits[new_c][i] &= ~(1 << j)
                    adj_bits[new_c][j] &= ~(1 << i)
                    
                    if v < best_v:
                        best_v = v
                        best_c = new_c
                
                if best_c >= 0:
                    colors[idx] = best_c
                    adj_bits[old_c][i] &= ~(1 << j)
                    adj_bits[old_c][j] &= ~(1 << i)
                    adj_bits[best_c][i] |= (1 << j)
                    adj_bits[best_c][j] |= (1 << i)
                    current_violations = best_v
                else:
                    # Random perturbation to escape local minima
                    new_c = rng.randint(0, 4)
                    colors[idx] = new_c
                    adj_bits[old_c][i] &= ~(1 << j)
                    adj_bits[old_c][j] &= ~(1 << i)
                    adj_bits[new_c][i] |= (1 << j)
                    adj_bits[new_c][j] |= (1 << i)
                    current_violations = total_violations(adj_bits, n)
            
            if found:
                best_n = n
                best_colors = colors
                break
        
        if found:
            continue
        else:
            # If we can't find a solution for this n, we might still try with remaining time
            if time.time() > deadline - 25:
                break
    
    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    else:
        n = 11
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
