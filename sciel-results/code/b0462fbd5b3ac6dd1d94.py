import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # We know n=26 is possible. Try to find it.
    # If we can't find it quickly, we might fall back to a known valid smaller n.
    # But let's give it a good chance.
    
    target_ns = [26, 25, 24, 23, 22, 21, 20]
    
    best_result = None
    
    for n in target_ns:
        if time.time() > deadline:
            break
            
        result = local_search(n, time_limit=15)
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        # Fallback: Return a trivially valid small n
        # n=11 coloring from seed
        best_result = {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
        
    return best_result

def local_search(n, time_limit=15):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Precompute edge indices
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    def get_violations(colors):
        # neighbors[c][v] is a bitmask of neighbors of v in color c
        neighbors = [[0] * n for _ in range(5)]
        
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    # Count bits
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    current_violations = get_violations(colors)
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    
    # Simulated Annealing / Hill Climbing
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        colors[e_idx] = new_color
        
        new_violations = get_violations(colors)
        
        delta = new_violations - current_violations
        
        # Accept if better, or with probability exp(-delta / temperature)
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            colors[e_idx] = old_color
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None
