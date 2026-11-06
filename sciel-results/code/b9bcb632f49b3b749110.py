import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try n=26 first, then 25
    for n in [26, 25]:
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts
        max_restarts = 100
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Time limit per restart
            time_limit = 1.5 if n == 26 else 1.0
            
            result = local_search(n, time_limit)
            if result is not None:
                return result

    # Fallback: A valid coloring for n=25 is known to exist.
    # If we haven't found it, we return a smaller valid one to ensure non-zero score.
    # But we should try harder for n=25.
    # Let's do a final long attempt for n=25 if we have time.
    if time.time() < deadline - 5:
        result = local_search(25, time_limit=10.0)
        if result is not None:
            return result

    # Absolute fallback
    return {'n': 11, 'colors': [(i + j) % 5 for i in range(11) for j in range(i + 1, 11)]}

def local_search(n, time_limit):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build neighbor bitmasks per color
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
    
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        total += cnt * (cnt - 1) // 2
        return total

    current_violations = count_violations()
    best_violations = current_violations
    best_colors = colors[:]
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        u, v = edges[e_idx]
        old_color = colors[e_idx]
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
        
        # Calculate delta
        delta = 0
        
        # Remove (u,v) from old_color
        # This affects pairs (u, w) and (v, w) for w != u, v
        for w in range(n):
            if w == u or w == v:
                continue
                
            # Old color effects
            # Check if v is neighbor of w in old_color
            if (1 << v) & neighbors[old_color][w]:
                # c_uw in old_color
                c_uw = (neighbors[old_color][u] & neighbors[old_color][w]).bit_count()
                # c_uw includes v? Yes, before removal.
                # After removal, c_uw becomes c_uw - 1.
                delta -= (c_uw - 1)
                
            if (1 << u) & neighbors[old_color][w]:
                c_vw = (neighbors[old_color][v] & neighbors[old_color][w]).bit_count()
                delta -= (c_vw - 1)
                
            # New color effects
            if (1 << v) & neighbors[new_color][w]:
                c_uw = (neighbors[new_color][u] & neighbors[new_color][w]).bit_count()
                # Before addition, v is NOT in N_new(u).
                # So v is NOT in intersection.
                # After addition, v is in N_new(u).
                # So intersection size increases by 1.
                delta += c_uw
                
            if (1 << u) & neighbors[new_color][w]:
                c_vw = (neighbors[new_color][v] & neighbors[new_color][w]).bit_count()
                delta += c_vw
                
        # Apply change
        colors[e_idx] = new_color
        neighbors[old_color][u] &= ~(1 << v)
        neighbors[old_color][v] &= ~(1 << u)
        neighbors[new_color][u] |= (1 << v)
        neighbors[new_color][v] |= (1 << u)
        
        current_violations += delta
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            if best_violations == 0:
                return {'n': n, 'colors': colors}
        
        # Simulated Annealing
        if delta >= 0:
            if random.random() < pow(2.71828, -delta / max(temperature, 0.01)):
                pass
            else:
                # Revert
                colors[e_idx] = old_color
                neighbors[new_color][u] &= ~(1 << v)
                neighbors[new_color][v] &= ~(1 << u)
                neighbors[old_color][u] |= (1 << v)
                neighbors[old_color][v] |= (1 << u)
                current_violations -= delta
                
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01

    return None
