import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Helper to count violations using bitmasks
    # colors: list of length n*(n-1)//2
    # n: number of vertices
    def count_violations(n, colors):
        # neighbors[c][v] is a bitmask of neighbors of v in color c
        neighbors = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
                
        violations = 0
        for c in range(5):
            # For each pair of vertices u, v, count common neighbors in color c
            # If k common neighbors, there are k*(k-1)//2 C4s
            for u in range(n):
                mask_u = neighbors[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    common = mask_u & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    # Local search for a specific n
    def local_search_n(n, time_limit):
        num_edges = n * (n - 1) // 2
        # Initialize with random colors
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Map (i, j) to index for fast access if needed, but we iterate sequentially in count_violations
        # For updates, we just pick a random index.
        
        current_violations = count_violations(n, colors)
        if current_violations == 0:
            return {'n': n, 'colors': colors}
            
        best_violations = current_violations
        best_colors = colors[:]
        
        start_ls = time.time()
        temperature = 1.0
        cooling_rate = 0.9995
        min_temp = 0.01
        
        while time.time() - start_ls < time_limit:
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            
            if new_color == old_color:
                continue
                
            colors[e_idx] = new_color
            new_violations = count_violations(n, colors)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return {'n': n, 'colors': best_colors}
            else:
                colors[e_idx] = old_color
                
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp
                
        # If we didn't find 0, return None or best found?
        # For n=26, we MUST find 0. If not, we failed.
        if best_violations == 0:
            return {'n': n, 'colors': best_colors}
        return None

    # 1. Try to find valid n=26
    result_26 = local_search_n(26, 30.0)
    
    if result_26 is None:
        # Fallback to n=25 if 26 fails (should be very unlikely with good SA)
        # Or just return a smaller valid graph.
        # Let's try n=25 quickly.
        result_25 = local_search_n(25, 10.0)
        if result_25 is not None:
            return result_25
        # Absolute fallback
        return {'n': 11, 'colors': [0]*55}

    # 2. Try to extend to n=27
    # We have a valid coloring for n=26.
    # We add vertex 26.
    # We need to assign colors to edges (26, i) for i in 0..25.
    # Let's use local search on these 26 variables.
    
    base_colors = result_26['colors']
    n_base = 26
    
    # Precompute base adjacencies for n=26
    adj = [[0] * n_base for _ in range(5)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    def get_violations_27(new_colors):
        # new_colors is a list of 26 ints, new_colors[i] is color of (26, i)
        # Group neighbors of 26 by color
        neighbors_by_color = [[] for _ in range(5)]
        for i in range(n_base):
            c = new_colors[i]
            neighbors_by_color[c].append(i)
            
        violations = 0
        for c in range(5):
            neighbors = neighbors_by_color[c]
            # For each pair of neighbors u, v in this color, check if they share a common neighbor in G_c (base)
            for u_idx in range(len(neighbors)):
                u = neighbors[u_idx]
                mask_u = adj[c][u]
                for v_idx in range(u_idx + 1, len(neighbors)):
                    v = neighbors[v_idx]
                    common = mask_u & adj[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    # Initialize new colors
    best_new_colors = [random.randint(0, 4) for _ in range(n_base)]
    best_violations = get_violations_27(best_new_colors)
    
    start_opt = time.time()
    limit_opt = 20.0 # 20s to try to extend
    temperature = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    while time.time() - start_opt < limit_opt:
        if best_violations == 0:
            break
            
        e_idx = random.randint(0, n_base - 1)
        old_c = best_new_colors[e_idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
            
        best_new_colors[e_idx] = new_c
        new_violations = get_violations_27(best_new_colors)
        
        delta = new_violations - best_violations
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            best_violations = new_violations
        else:
            best_new_colors[e_idx] = old_c
            
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp
            
    if best_violations == 0:
        # Construct the full coloring for n=27
        final_colors = [0] * (27 * 26 // 2)
        
        idx = 0
        for i in range(27):
            for j in range(i + 1, 27):
                if j < 26:
                    # Edge (i, j) is in K26
                    # Calculate index in K26 coloring
                    # Edges before i: sum_{k=0}^{i-1} (25-k)
                    # Edges from i to < j: j - i - 1
                    base_idx = i * 25 - i * (i - 1) // 2 + (j - i - 1)
                    final_colors[idx] = base_colors[base_idx]
                else:
                    # j == 26, edge is (i, 26)
                    final_colors[idx] = best_new_colors[i]
                idx += 1
                
        return {'n': 27, 'colors': final_colors}
        
    # If extension failed, return n=26
    return result_26

# To allow for some randomness in the seed, we can set the seed based on time or just let it be random
# The prompt says "search score is value/seed", implying the seed might be fixed or variable.
# We should not hardcode a specific seed that might be "bad" for the random walk, but rather a robust algorithm.

if __name__ == '__main__':
    # For testing
    res = construct()
    print(res['n'])
