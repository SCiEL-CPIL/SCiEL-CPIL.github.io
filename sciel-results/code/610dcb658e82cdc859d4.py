import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # We want to maximize n.
    # Known best is 26.
    # We will try to find a valid coloring for n=26, then 25, etc.
    # We use a fast local search with delta evaluation.
    
    target_ns = [26, 25, 24]
    
    best_result = None
    
    for n in target_ns:
        if time.time() > deadline - 5:
            break
            
        # Allocate time based on difficulty
        # n=26 is the hardest. Give it the most time.
        time_limit = 20 if n == 26 else 10
        
        # Try multiple initializations
        # 1. Random
        # 2. Structured (Algebraic-like)
        
        # Try structured initialization first as it often starts closer to a solution
        result = local_search_fast_structured(n, time_limit, structured=True)
        if result is not None:
            return result
        
        # If structured fails, try random initialization with remaining time
        if time.time() < deadline - 5:
            result = local_search_fast_structured(n, time_limit, structured=False)
            if result is not None:
                return result
                
    # Fallback to seed if all fails (should not happen if logic is correct, but safety first)
    # The seed n=11 is valid.
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def local_search_fast_structured(n, time_limit, structured=True):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Map edge to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Initialize colors
    colors = [0] * num_edges
    if structured:
        # Algebraic-like initialization
        # Try a few different structured patterns
        # Pattern 1: (i * j) % 5
        # Pattern 2: (i + j) % 5
        # Pattern 3: (i * j + i - j) % 5
        # We'll pick a random structured pattern from a small set to diversify
        pattern_id = random.randint(0, 2)
        for idx, (i, j) in enumerate(edges):
            if pattern_id == 0:
                colors[idx] = (i * j) % 5
            elif pattern_id == 1:
                colors[idx] = (i + j) % 5
            else:
                colors[idx] = (i * j + i - j) % 5
    else:
        for idx in range(num_edges):
            colors[idx] = random.randint(0, 4)
            
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    
    def update_neighbors(idx, old_color, new_color):
        i, j = edges[idx]
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    # Precompute common neighbor counts for each color and pair of vertices
    # k[c][u][v] = number of common neighbors of u and v in color c
    k = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    def calc_k_for_color(c):
        for u in range(n):
            for v in range(u + 1, n):
                common = neighbors[c][u] & neighbors[c][v]
                kc = common.bit_count()
                k[c][u][v] = kc
                k[c][v][u] = kc
                
    for c in range(5):
        calc_k_for_color(c)
        
    def total_violations():
        v = 0
        for c in range(5):
            for u in range(n):
                for v_idx in range(u + 1, n):
                    kc = k[c][u][v_idx]
                    if kc >= 2:
                        v += kc * (kc - 1) // 2
        return v
        
    current_violations = total_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.9995 # Slower cooling for larger n
    
    # Local search loop
    while time.time() < start_time + time_limit:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        # Calculate delta violations
        # We update k for pairs involving i or j.
        # Delta = NewViolations - OldViolations
        delta = 0
        
        # Effect on old_color: Removing edge (i, j) from old_color
        # For any w != i, j:
        # If j is in N_old(w), then j was in N_old(i) and N_old(w), so it contributed to k_old(i, w).
        # Removing j from N_old(i) reduces k_old(i, w) by 1.
        # If i is in N_old(w), then i was in N_old(j) and N_old(w), so it contributed to k_old(j, w).
        # Removing i from N_old(j) reduces k_old(j, w) by 1.
        
        for w in range(n):
            if w == i or w == j:
                continue
            # Update k for old_color
            # Check if j is neighbor of w in old_color
            if (neighbors[old_color][w] >> j) & 1:
                old_k = k[old_color][i][w]
                new_k = old_k - 1
                # Contribution to violations: C(old_k, 2) -> C(new_k, 2)
                delta -= (old_k * (old_k - 1) // 2) - (new_k * (new_k - 1) // 2)
                k[old_color][i][w] = new_k
                k[old_color][w][i] = new_k
                
            # Check if i is neighbor of w in old_color
            if (neighbors[old_color][w] >> i) & 1:
                old_k = k[old_color][j][w]
                new_k = old_k - 1
                delta -= (old_k * (old_k - 1) // 2) - (new_k * (new_k - 1) // 2)
                k[old_color][j][w] = new_k
                k[old_color][w][j] = new_k
                
        # Effect on new_color: Adding edge (i, j) to new_color
        # For any w != i, j:
        # If j is in N_new(w), then j is in N_new(i) (after add) and N_new(w), so k_new(i, w) increases by 1.
        # If i is in N_new(w), then i is in N_new(j) (after add) and N_new(w), so k_new(j, w) increases by 1.
        
        for w in range(n):
            if w == i or w == j:
                continue
            # Update k for new_color
            # Check if j is neighbor of w in new_color
            if (neighbors[new_color][w] >> j) & 1:
                old_k = k[new_color][i][w]
                new_k = old_k + 1
                delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                k[new_color][i][w] = new_k
                k[new_color][w][i] = new_k
                
            # Check if i is neighbor of w in new_color
            if (neighbors[new_color][w] >> i) & 1:
                old_k = k[new_color][j][w]
                new_k = old_k + 1
                delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                k[new_color][j][w] = new_k
                k[new_color][w][j] = new_k
                
        # Apply change if accepted
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_violations += delta
            colors[e_idx] = new_color
            update_neighbors(e_idx, old_color, new_color)
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert k changes
            # We need to undo the k updates.
            # Since we only changed k for pairs involving i or j, we can recalculate them.
            # This is O(n) per revert, which is acceptable.
            for w in range(n):
                if w == i or w == j:
                    continue
                # Recalculate k[old][i][w]
                common = neighbors[old_color][i] & neighbors[old_color][w]
                k[old_color][i][w] = common.bit_count()
                k[old_color][w][i] = common.bit_count()
                
                common = neighbors[old_color][j] & neighbors[old_color][w]
                k[old_color][j][w] = common.bit_count()
                k[old_color][w][j] = common.bit_count()
                
                common = neighbors[new_color][i] & neighbors[new_color][w]
                k[new_color][i][w] = common.bit_count()
                k[new_color][w][i] = common.bit_count()
                
                common = neighbors[new_color][j] & neighbors[new_color][w]
                k[new_color][j][w] = common.bit_count()
                k[new_color][w][j] = common.bit_count()
                
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None
