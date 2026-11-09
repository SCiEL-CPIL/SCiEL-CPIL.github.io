"""Bootstrap portfolio: SEARCH approach for 4-colourings of K_n without monochromatic triangles.
Strategy:
1. Start from known good constructions (32-vertex via GF(16) doubling, plus others).
2. Use vectorized local search / simulated annealing to extend beyond 32 vertices.
3. Bitset-based vertex addition with structured color assignment.
4. Try random restarts with different initial colorings.
5. Return the best valid coloring found within the time budget.
"""
import time
import random
import numpy as np

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 140.0  # seconds (10s safety margin)

def time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def time_up():
    return time.time() - START_TIME >= TIME_LIMIT - 2.0

def validate(colors, n):
    """Check that no color class contains a triangle. Returns True if valid."""
    if n < 1:
        return False
    expected = n * (n - 1) // 2
    if len(colors) != expected:
        return False
    # Build adjacency: for each color, list of edges
    from collections import defaultdict
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 3:
                return False
            adj[c].add((i, j))
            idx += 1
    # For each color, check no triangle
    for c in range(4):
        edges = adj[c]
        # Build adjacency dict for this color
        adj_dict = defaultdict(set)
        for (u, v) in edges:
            adj_dict[u].add(v)
            adj_dict[v].add(u)
        for (u, v) in edges:
            common = adj_dict[u] & adj_dict[v]
            if common:
                return False
    return True

def build_colors_matrix(colors, n):
    """Convert flat list to 2D numpy matrix for fast access."""
    mat = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = colors[idx]
            mat[j, i] = colors[idx]
            idx += 1
    return mat

def flatten_matrix(mat):
    n = mat.shape[0]
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(int(mat[i, j]))
    return colors

def has_mono_triangle(mat, i, j, k):
    """Check if triangle (i,j,k) is monochromatic."""
    c1 = mat[i, j]
    c2 = mat[i, k]
    c3 = mat[j, k]
    return c1 == c2 == c3

def count_mono_triangles(mat, n):
    """Count monochromatic triangles using vectorized approach."""
    count = 0
    for i in range(n):
        # For each i, count triangles (i,j,k) where j,k > i
        # mat[i, i+1:] gives colors of edges from i to others
        rest = mat[i, i+1:]  # shape (n-i-1,)
        if len(rest) < 2:
            continue
        sub = mat[i+1:, i+1:]  # shape (n-i-1, n-i-1)
        # Triangle (i,j,k) is mono if mat[i,j]==mat[i,k]==mat[j,k]
        # For each j (index in rest), check if rest[j] == sub[j,k] for any k>j
        # Vectorized: for each j, count k>j where rest[j]==sub[j,k] and mat[i,k]==rest[j]
        # Actually simpler: count triples where all three colors match
        for j in range(len(rest) - 1):
            cj = rest[j]
            # Need mat[i,k] == cj and mat[j,k] == cj for k > j (in sub indices)
            # mat[i, i+1+j+1:] = rest[j+1:]
            # sub[j, j+1:]
            matches = (rest[j+1:] == cj) & (sub[j, j+1:] == cj)
            count += matches.sum()
    return count

def count_mono_triangles_fast(mat, n):
    """Faster count using the structure: for each edge (i,j), count k where all three match."""
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            c = mat[i, j]
            # Count k where mat[i,k]==c and mat[j,k]==c, k != i,j
            mask_i = (mat[i, :] == c)
            mask_j = (mat[j, :] == c)
            mask_i[i] = 0
            mask_i[j] = 0
            mask_j[i] = 0
            mask_j[j] = 0
            count += (mask_i & mask_j).sum()
    # Each triangle counted 3 times (once per edge)
    return count // 3

def local_search(mat, n, max_iters=10000, temperature=1.0):
    """Vectorized local search to reduce monochromatic triangles to 0.
    Returns (mat, success) where success is True if 0 mono triangles."""
    mat = mat.copy()
    current_cost = count_mono_triangles_fast(mat, n)
    
    if current_cost == 0:
        return mat, True
    
    best_mat = mat.copy()
    best_cost = current_cost
    
    T = temperature
    cooling = 0.9999
    rng = np.random
    
    for it in range(max_iters):
        if time_up():
            break
        if current_cost == 0:
            return mat, True
        
        # Randomly pick an edge and try changing its color
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        
        old_color = mat[i, j]
        new_color = rng.randint(0, 3)
        if new_color == old_color:
            continue
        
        # Compute delta cost using vectorized operations
        # Triangles involving edge (i,j): (i,j,k) for all k != i,j
        # Old: triangle is mono if old_color == mat[i,k] == mat[j,k]
        # New: triangle is mono if new_color == mat[i,k] == mat[j,k]
        
        mi = mat[i, :]
        mj = mat[j, :]
        mi_i = mi[i]
        mi_j = mi[j]
        mj_i = mj[i]
        mj_j = mj[j]
        
        # For each k, old triangle (i,j,k) is mono if old_color==mi[k]==mj[k]
        # Exclude k=i and k=j
        old_mono = (mi == old_color) & (mj == old_color)
        old_mono[i] = False
        old_mono[j] = False
        old_count = old_mono.sum()
        
        new_mono = (mi == new_color) & (mj == new_color)
        new_mono[i] = False
        new_mono[j] = False
        new_count = new_mono.sum()
        
        delta = new_count - old_count
        
        new_cost = current_cost + delta
        
        if new_cost < current_cost:
            mat[i, j] = new_color
            mat[j, i] = new_color
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_mat = mat.copy()
        elif delta == 0:
            mat[i, j] = new_color
            mat[j, i] = new_color
            current_cost = new_cost
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < np.exp(-delta / max(T, 1e-10)):
                mat[i, j] = new_color
                mat[j, i] = new_color
                current_cost = new_cost
        
        T *= cooling
        T = max(T, 0.01)
    
    return best_mat, best_cost == 0

def try_extend(mat, n, target_n, max_iters=20000):
    """Try to extend a valid n-vertex coloring to target_n vertices.
    Uses structured initialization then local search.
    Returns (mat, success)."""
    if target_n <= n:
        return mat, True
    
    # Extend matrix
    new_mat = np.zeros((target_n, target_n), dtype=np.int8)
    new_mat[:n, :n] = mat
    
    # For new vertices, use structured initialization based on the existing coloring
    # Try to find good colors for edges between old and new vertices
    for i in range(n, target_n):
        # Initialize edges from new vertex i to old vertices
        for j in range(n):
            # Use a heuristic: pick the color that minimizes immediate mono triangles
            best_c = 0
            best_score = float('inf')
            for c in range(4):
                # Count how many k in [0,n) where mat[j,k]==c and mat[i,k] would be c
                # For now just use random but weighted
                score = 0
                for k in range(j):
                    if mat[j, k] == c:
                        score += 1
                # Also check edges to other new vertices (not set yet)
                if score < best_score:
                    best_score = score
                    best_c = c
            new_mat[i, j] = best_c
            new_mat[j, i] = best_c
        
        # Edges between new vertices
        for j in range(i + 1, target_n):
            new_mat[i, j] = random.randint(0, 3)
            new_mat[j, i] = new_mat[i, j]
    
    # Local search to eliminate mono triangles
    result, success = local_search(new_mat, target_n, max_iters=max_iters)
    return result, success

def try_extend_bitset(mat, n, target_n, max_iters=50000):
    """Try to extend using bitset-based approach: add vertices one at a time.
    For each new vertex, try all 4^n color assignments (or a subset) and pick the best.
    Returns (mat, success)."""
    if target_n <= n:
        return mat, True
    
    new_mat = mat.copy()
    current_n = n
    
    for new_v in range(n, target_n):
        if time_up():
            break
        
        # Extend matrix
        new_mat = np.zeros((new_mat.shape[0] + 1, new_mat.shape[0] + 1), dtype=np.int8)
        new_mat[:current_n, :current_n] = mat[:current_n, :current_n]
        
        # Try to find good colors for edges from new_v to all existing vertices
        # Use local search on just these edges
        best_assignment = None
        best_cost = float('inf')
        
        # Initialize with random
        np.random.seed(new_v * 7919 + 13)
        for attempt in range(20):
            if time_up():
                break
            
            assignment = np.random.randint(0, 4, current_n, dtype=np.int8)
            
            # Set up temporary matrix
            temp_mat = np.zeros((current_n + 1, current_n + 1), dtype=np.int8)
            temp_mat[:current_n, :current_n] = mat[:current_n, :current_n]
            temp_mat[current_n, :current_n] = assignment
            temp_mat[:current_n, current_n] = assignment
            # Edges to other new vertices (if any already added)
            # For simplicity, we add one vertex at a time so no other new vertices yet
            
            # Count mono triangles involving the new vertex
            cost = 0
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    c1 = assignment[i]
                    c2 = assignment[j]
                    c3 = mat[i, j]
                    if c1 == c2 == c3:
                        cost += 1
            
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment.copy()
            
            if best_cost == 0:
                break
        
        # Apply best assignment
        temp_mat = np.zeros((current_n + 1, current_n + 1), dtype=np.int8)
        temp_mat[:current_n, :current_n] = mat[:current_n, :current_n]
        temp_mat[current_n, :current_n] = best_assignment
        temp_mat[:current_n, current_n] = best_assignment
        
        # Now do local search on the full matrix to fix any remaining issues
        result, success = local_search(temp_mat, current_n + 1, max_iters=max_iters // 10)
        
        if success:
            mat = result
            current_n += 1
        else:
            # Use the result even if not perfect, continue trying
            mat = result
            current_n += 1
    
    # Final check
    final_mat = mat[:target_n, :target_n]
    cost = count_mono_triangles_fast(final_mat, target_n)
    return final_mat, cost == 0

def construct():
    global START_TIME
    START_TIME = time.time()
    
    random.seed(42)
    np.random.seed(42)
    
    best_n = 0
    best_colors = None
    
    # Start with the known 32-vertex construction
    def make_gf16_doubled():
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
        return 32, colors
    
    n0, colors0 = make_gf16_doubled()
    if validate(colors0, n0):
        best_n = n0
        best_colors = colors0
    
    # Try to extend from 32 to larger n
    mat32 = build_colors_matrix(colors0, 32)
    
    # Strategy 1: Try various target sizes with structured extension
    for target in range(33, 51):
        if time_up():
            break
        
        # Try multiple restarts with different approaches
        for restart in range(5):
            if time_up():
                break
            
            if restart < 2:
                ext_mat, success = try_extend(mat32, 32, target, max_iters=8000)
            else:
                ext_mat, success = try_extend_bitset(mat32, 32, target, max_iters=20000)
            
            if success:
                colors = flatten_matrix(ext_mat)
                if validate(colors, target):
                    best_n = target
                    best_colors = colors
                    break  # Try next target
    
    # Strategy 2: If we haven't exceeded 32, try random 4-colorings with local search for various n
    if best_n <= 32:
        for n_try in range(33, 48):
            if time_up():
                break
            
            for restart in range(8):
                if time_up():
                    break
                
                mat = np.random.randint(0, 4, (n_try, n_try), dtype=np.int8)
                # Make symmetric
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        mat[j, i] = mat[i, j]
                
                result, success = local_search(mat, n_try, max_iters=5000)
                if success:
                    colors = flatten_matrix(result)
                    if validate(colors, n_try):
                        if n_try > best_n:
                            best_n = n_try
                            best_colors = colors
                        break
    
    # Strategy 3: Aggressive local search from the best found so far
    if best_n > 0 and best_colors is not None and not time_up():
        mat_best = build_colors_matrix(best_colors, best_n)
        # Try to push further with more iterations
        for target in range(best_n + 1, min(best_n + 5, 51)):
            if time_up():
                break
            for restart in range(3):
                if time_up():
                    break
                ext_mat, success = try_extend(mat_best, best_n, target, max_iters=15000)
                if success:
                    colors = flatten_matrix(ext_mat)
                    if validate(colors, target):
                        best_n = target
                        best_colors = colors
                        mat_best = ext_mat
                        break
    
    # Final check and return
    if best_colors is None:
        # Fallback to 32-vertex construction
        best_n = 32
        best_colors = colors0
    
    return {'n': best_n, 'colors': best_colors}
