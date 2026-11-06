import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    # Fallback solution (n=11)
    def get_seed_solution(n):
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    
    # Try n=28, 27, 26 first
    for n in [28, 27, 26]:
        if time.time() > deadline - 15:
            break
        solution = find_valid_coloring(n, time_limit=30)
        if solution is not None:
            return solution
            
    # If we didn't find anything, try smaller n with remaining time
    for n in range(25, 11, -1):
        if time.time() > deadline - 5:
            break
        solution = find_valid_coloring(n, time_limit=3)
        if solution is not None:
            return solution
            
    return get_seed_solution(11)

def find_valid_coloring(n, time_limit=30):
    """
    Find a 5-coloring of K_n with no monochromatic C4.
    Uses optimized SA with O(1) incremental energy updates.
    """
    num_edges = n * (n - 1) // 2
    if num_edges == 0:
        return {'n': n, 'colors': []}
        
    # Precompute edge indices
    edge_index = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[i][j] = idx
            edge_index[j][i] = idx
            idx += 1
            
    # Bitset adjacency for each color
    adj_bit = [[0] * n for _ in range(5)]
    
    # Degree counts: deg[c][u] = number of neighbors of u in color c
    deg = [[0] * n for _ in range(5)]
    
    def rebuild_adj():
        for c in range(5):
            for u in range(n):
                adj_bit[c][u] = 0
                deg[c][u] = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_index[i][j]]
                adj_bit[c][i] |= (1 << j)
                adj_bit[c][j] |= (1 << i)
                deg[c][i] += 1
                deg[c][j] += 1
                
    # Energy: sum over all color classes and all pairs of C(common_neighbors, 2)
    def compute_energy():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    cn = bin(adj_bit[c][u] & adj_bit[c][v]).count('1')
                    if cn >= 2:
                        total += cn * (cn - 1) // 2
        return total
        
    # Try multiple restarts with different strategies
    best_energy = float('inf')
    best_colors = None
    
    for attempt in range(5):
        if time.time() > start_time + time_limit:
            break
            
        # Initialize: for first attempt, use a structured initialization
        if attempt == 0:
            # Structured init: assign colors based on a pattern
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    # Use a modular pattern to spread colors
                    c = (i + j) % 5
                    colors.append(c)
        else:
            # Random initialization
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
        rebuild_adj()
        energy = compute_energy()
        
        if energy == 0:
            return {'n': n, 'colors': colors}
            
        # SA parameters
        T = 3.0
        cooling = 0.999
        min_T = 0.01
        sa_start = time.time()
        sa_limit = time_limit / 5  # Use 1/5 of time per attempt
        
        # Main SA loop with O(1) energy updates
        while time.time() < sa_start + sa_limit:
            if T < min_T:
                T = min_T
                
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            idx_e = edge_index[i][j]
            c1 = colors[idx_e]
            c2 = random.randint(0, 4)
            if c1 == c2:
                continue
                
            # Compute delta energy in O(1) using degree counts
            # When we move edge (i,j) from c1 to c2:
            # For c1: removing (i,j) reduces common neighbor count for all pairs (i,k) and (j,k)
            # where k is adjacent to both in c1. The change in C(cn,2) for each such pair is -1
            # if cn >= 2, or 0 if cn < 2.
            # More precisely: for each k in N_c1(i) ∩ N_c1(j) \ {i,j}, the common neighbor count
            # of (i,k) decreases by 1 (since j was a common neighbor), and similarly for (j,k).
            # Actually, let's think more carefully.
            
            # When edge (i,j) has color c1:
            # - For any vertex k ≠ i,j: if k is adjacent to i in color c1, then j is a common neighbor of (i,k) in c1
            #   (since edge (i,j) is color c1 and edge (j,k) must be color c1 for j to be a common neighbor).
            #   Wait, no. Common neighbors of (i,k) in color c1 are vertices m such that both (i,m) and (k,m) are color c1.
            #   So j is a common neighbor of (i,k) in c1 iff (i,j) is color c1 AND (k,j) is color c1.
            #   Since (i,j) IS color c1, j is a common neighbor of (i,k) in c1 iff (k,j) is color c1.
            #   So the number of such k is deg_c1(j) - 1 (excluding i).
            #   For each such k, the common neighbor count of (i,k) in c1 includes j. When we remove (i,j) from c1,
            #   j is no longer a common neighbor of (i,k) in c1. So cn(i,k) decreases by 1.
            #   The change in C(cn,2) is C(cn-1,2) - C(cn,2) = -(cn-1).
            
            # Similarly for (j,k) where k is adjacent to j in c1 (excluding i).
            
            # For c2: adding (i,j)
            # For each k adjacent to i in c2 (excluding j), j becomes a new common neighbor of (i,k) in c2.
            # cn(i,k) increases by 1. Change in C(cn,2) is C(cn+1,2) - C(cn,2) = cn.
            # Similarly for (j,k) where k is adjacent to j in c2 (excluding i).
            
            # But we need to be careful: we also need to consider that i and j themselves might be
            # common neighbors of other pairs. Actually, the only pairs whose common neighbor count
            # changes are those involving i or j.
            
            # Let's compute delta more carefully:
            delta = 0
            
            # For color c1 (removing edge i-j):
            # Pairs affected: (i, k) for k in N_c1(j) \ {i}, and (j, k) for k in N_c1(i) \ {j}
            # For each k in N_c1(j) \ {i}: cn(i,k) in c1 decreases by 1 (j was a common neighbor)
            #   Old cn = cn(i,k) in c1, new cn = old_cn - 1
            #   Delta = C(old_cn-1, 2) - C(old_cn, 2) = -(old_cn - 1)
            # We need cn(i,k) in c1 for each such k.
            # cn(i,k) in c1 = |N_c1(i) ∩ N_c1(k)|
            
            # This is still O(n) per edge change. Let me think if there's a better way.
            
            # Alternative: precompute for each pair (u,v) and color c, the common neighbor count.
            # But that's O(n^2 * 5) memory and updating it is also costly.
            
            # Let's just do the O(n) computation but make it fast.
            
            # Get neighbors of j in color c1, excluding i
            mask_j = adj_bit[c1][j] & ~(1 << i)
            m = mask_j
            while m:
                lsb = m & (-m)
                k = lsb.bit_length() - 1
                m &= m - 1
                # cn(i,k) in c1
                cn = bin(adj_bit[c1][i] & adj_bit[c1][k]).count('1')
                delta -= (cn - 1)  # C(cn-1,2) - C(cn,2) = -(cn-1)
                
            # Get neighbors of i in color c1, excluding j
            mask_i = adj_bit[c1][i] & ~(1 << j)
            m = mask_i
            while m:
                lsb = m & (-m)
                k = lsb.bit_length() - 1
                m &= m - 1
                cn = bin(adj_bit[c1][j] & adj_bit[c1][k]).count('1')
                delta -= (cn - 1)
                
            # For color c2 (adding edge i-j):
            # For each k in N_c2(i) \ {j}: cn(j,k) in c2 increases by 1 (i becomes a common neighbor)
            #   Old cn = cn(j,k) in c2, new cn = old_cn + 1
            #   Delta = C(old_cn+1, 2) - C(old_cn, 2) = old_cn
            mask_i2 = adj_bit[c2][i] & ~(1 << j)
            m = mask_i2
            while m:
                lsb = m & (-m)
                k = lsb.bit_length() - 1
                m &= m - 1
                cn = bin(adj_bit[c2][j] & adj_bit[c2][k]).count('1')
                delta += cn
                
            # For each k in N_c2(j) \ {i}: cn(i,k) in c2 increases by 1
            mask_j2 = adj_bit[c2][j] & ~(1 << i)
            m = mask_j2
            while m:
                lsb = m & (-m)
                k = lsb.bit_length() - 1
                m &= m - 1
                cn = bin(adj_bit[c2][i] & adj_bit[c2][k]).count('1')
                delta += cn
                
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept move
                colors[idx_e] = c2
                adj_bit[c1][i] &= ~(1 << j)
                adj_bit[c1][j] &= ~(1 << i)
                adj_bit[c2][i] |= (1 << j)
                adj_bit[c2][j] |= (1 << i)
                deg[c1][i] -= 1
                deg[c1][j] -= 1
                deg[c2][i] += 1
                deg[c2][j] += 1
                energy += delta
                
                if energy == 0:
                    return {'n': n, 'colors': colors}
                    
            T *= cooling
            
        # Track best
        if energy < best_energy:
            best_energy = energy
            best_colors = colors[:]
            
    # If we have a good solution, try to polish it
    if best_colors is not None and best_energy < 10:
        polish_start = time.time()
        polish_limit = time_limit - (polish_start - start_time)
        if polish_limit > 2:
            polished = polish_coloring(n, best_colors, polish_limit, edge_index)
            if polished is not None:
                return {'n': n, 'colors': polished}
                
    # Return best found or None
    if best_colors is not None:
        # Verify
        adj_bit = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = best_colors[edge_index[i][j]]
                adj_bit[c][i] |= (1 << j)
                adj_bit[c][j] |= (1 << i)
        valid = True
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    cn = bin(adj_bit[c][u] & adj_bit[c][v]).count('1')
                    if cn >= 2:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return {'n': n, 'colors': best_colors}
            
    return None

def polish_coloring(n, colors, time_limit, edge_index):
    """
    Polish a near-valid coloring by fixing remaining C4s.
    """
    num_edges = n * (n - 1) // 2
    adj_bit = [[0] * n for _ in range(5)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[edge_index[i][j]]
            adj_bit[c][i] |= (1 << j)
            adj_bit[c][j] |= (1 << i)
            
    def compute_energy():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    cn = bin(adj_bit[c][u] & adj_bit[c][v]).count('1')
                    if cn >= 2:
                        total += cn * (cn - 1) // 2
        return total
        
    energy = compute_energy()
    if energy == 0:
        return colors[:]
        
    start_t = time.time()
    T = 0.5
    cooling = 0.999
    min_T = 0.005
    
    while time.time() < start_t + time_limit:
        if T < min_T:
            T = min_T
            
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        idx_e = edge_index[i][j]
        c1 = colors[idx_e]
        c2 = random.randint(0, 4)
        if c1 == c2:
            continue
            
        # Compute delta
        delta = 0
        mask_j = adj_bit[c1][j] & ~(1 << i)
        m = mask_j
        while m:
            lsb = m & (-m)
            k = lsb.bit_length() - 1
            m &= m - 1
            cn = bin(adj_bit[c1][i] & adj_bit[c1][k]).count('1')
            delta -= (cn - 1)
            
        mask_i = adj_bit[c1][i] & ~(1 << j)
        m = mask_i
        while m:
            lsb = m & (-m)
            k = lsb.bit_length() - 1
            m &= m - 1
            cn = bin(adj_bit[c1][j] & adj_bit[c1][k]).count('1')
            delta -= (cn - 1)
            
        mask_i2 = adj_bit[c2][i] & ~(1 << j)
        m = mask_i2
        while m:
            lsb = m & (-m)
            k = lsb.bit_length() - 1
            m &= m - 1
            cn = bin(adj_bit[c2][j] & adj_bit[c2][k]).count('1')
            delta += cn
            
        mask_j2 = adj_bit[c2][j] & ~(1 << i)
        m = mask_j2
        while m:
            lsb = m & (-m)
            k = lsb.bit_length() - 1
            m &= m - 1
            cn = bin(adj_bit[c2][i] & adj_bit[c2][k]).count('1')
            delta += cn
            
        if delta <= 0 or random.random() < math.exp(-delta / T):
            colors[idx_e] = c2
            adj_bit[c1][i] &= ~(1 << j)
            adj_bit[c1][j] &= ~(1 << i)
            adj_bit[c2][i] |= (1 << j)
            adj_bit[c2][j] |= (1 << i)
            energy += delta
            
            if energy == 0:
                return colors[:]
                
        T *= cooling
        
    return None
