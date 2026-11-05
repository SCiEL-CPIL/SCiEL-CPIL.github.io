import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin
    
    # Target size. We aim for 50. If we fail, we might try 48, etc.
    # But the search is fast enough that we can just aim for 50.
    N = 50
    
    # We will run multiple restarts if we don't find a solution quickly.
    # Each restart is a fresh random initialization.
    
    for restart in range(50):  # Max 50 restarts, but time is the real limit
        if time.time() > DEADLINE - 5:
            break
            
        # 1. Initialization: Random 4-coloring
        # colors[i][j] for i < j
        # We store colors in a 2D array for easy access, or just a list of edges?
        # 2D array NxN is easier for neighbor logic.
        # colors[i][j] = c. Symmetric.
        colors = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            for j in range(i + 1, N):
                colors[i][j] = random.randint(0, 3)
                colors[j][i] = colors[i][j]
        
        # 2. Build neighbor lists
        # neighbors[v][c] = list of u such that colors[v][u] == c
        neighbors = [[[] for _ in range(4)] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                c = colors[i][j]
                neighbors[i][c].append(j)
                neighbors[j][c].append(i)
        
        # 3. Helper to count mono triangles for a specific edge (u, v) of color c
        # It is the size of intersection of neighbors[u][c] and neighbors[v][c]
        # Since lists are small (<50), we can use set intersection or just a boolean array.
        # For speed, we'll use a helper that assumes u, v are neighbors of each other in color c?
        # No, the intersection counts vertices w such that (u,w) is c and (v,w) is c.
        # Note: w cannot be u or v.
        
        def count_triangles_for_edge(u, v, c):
            # Count w in neighbors[u][c] that are also in neighbors[v][c]
            # Optimization: iterate over the smaller list
            list_u = neighbors[u][c]
            list_v = neighbors[v][c]
            if len(list_u) > len(list_v):
                list_u, list_v = list_v, list_u
            
            # Convert list_v to a set for O(1) lookup? Or just use in?
            # Lists are small, 'in' is O(N). Intersection is O(N^2) worst case, but N=50.
            # Actually, creating a set of the larger list might be faster if called often.
            # But we call this for every move evaluation.
            # Let's use a boolean array for the larger list?
            # Or just:
            count = 0
            set_v = set(list_v)
            for w in list_u:
                if w in set_v:
                    count += 1
            return count

        # 4. Calculate initial total triangle count
        # We need a list of "bad" edges to pick from.
        # An edge is "bad" if it is in at least one mono triangle.
        
        def get_bad_edges():
            bad = []
            for i in range(N):
                for j in range(i + 1, N):
                    c = colors[i][j]
                    if count_triangles_for_edge(i, j, c) > 0:
                        bad.append((i, j, c))
            return bad

        bad_edges = get_bad_edges()
        
        # Simulated Annealing parameters
        # T starts high, decays.
        current_T = 10.0
        cooling = 0.95
        
        # Local search loop
        max_iters = 50000
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            
            if not bad_edges:
                # Found a solution!
                # Format output
                res_colors = []
                for i in range(N):
                    for j in range(i + 1, N):
                        res_colors.append(int(colors[i][j]))
                return {'n': N, 'colors': res_colors}
            
            # Pick a random bad edge
            u, v, c_old = random.choice(bad_edges)
            
            # Try changing color to one of the other 3
            best_delta = float('inf')
            best_c = -1
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                
                # Calculate delta
                # Remove triangles of c_old: -count_triangles_for_edge(u, v, c_old)
                # Add triangles of c_new: +count_triangles_for_edge(u, v, c_new)
                # Note: The function count_triangles_for_edge depends on the current neighbors.
                # If we hypothetically changed the color, the neighbors would change.
                # However, the intersection count for the NEW color c_new is calculated based on
                # the CURRENT neighbors of u and v for color c_new.
                # Is this correct?
                # If we change (u, v) to c_new, u becomes a neighbor of v in c_new.
                # The triangles formed are (u, v, w) where w is a common neighbor in c_new.
                # The current neighbors[u][c_new] and neighbors[v][c_new] do NOT include each other.
                # So the count of w is exactly the number of common neighbors in c_new.
                # This is correct.
                
                k_old = count_triangles_for_edge(u, v, c_old)
                k_new = count_triangles_for_edge(u, v, c_new)
                delta = k_new - k_old
                
                if delta < best_delta:
                    best_delta = delta
                    best_c = c_new
            
            # Acceptance criterion: Simulated Annealing
            # If best_delta < 0, always accept.
            # If best_delta >= 0, accept with probability exp(-delta / current_T)
            
            accept = False
            if best_delta < 0:
                accept = True
            else:
                # Probability
                p = np.exp(-best_delta / current_T)
                if random.random() < p:
                    accept = True
            
            if accept:
                # Apply move
                # Update colors
                colors[u][v] = best_c
                colors[v][u] = best_c
                
                # Update neighbors
                # Remove u from v's old color list, and v from u's old color list
                neighbors[u][c_old].remove(v)
                neighbors[v][c_old].remove(u)
                
                # Add u to v's new color list, and v to u's new color list
                neighbors[u][best_c].append(v)
                neighbors[v][best_c].append(u)
                
                # We need to update bad_edges.
                # This is the expensive part if we recalculate from scratch.
                # O(N^2 * N) = O(N^3) = 125k ops.
                # 50000 iters * 125k = 6.25e9 ops. Too slow in Python.
                
                # Optimization: Only update bad_edges for edges incident to u or v?
                # An edge (a, b) is bad if it's in a mono triangle.
                # Changing (u, v) affects:
                # 1. The edge (u, v) itself.
                # 2. Edges (u, w) for w in neighbors[u][c_old] U neighbors[u][best_c]
                # 3. Edges (v, w) for w in neighbors[v][c_old] U neighbors[v][best_c]
                # 4. Edges (w, x) where w is connected to u and x to v? No, only if they form a triangle with the changed edge.
                # Actually, a triangle (a, b, c) is mono if all 3 edges have same color.
                # Changing (u, v) only affects triangles containing (u, v).
                # So only edges in triangles with (u, v) can change their "badness".
                # Triangles with (u, v) are (u, v, w) where w is a common neighbor.
                # So only edges (u, w) and (v, w) for such w might change status?
                # No, the edge (u, w) is in a triangle (u, v, w). If (u, v) changes color,
                # the triangle (u, v, w) might break or form.
                # Does this affect if edge (u, w) is "bad"?
                # Edge (u, w) is bad if it is in ANY mono triangle.
                # It might be in other triangles (u, w, z).
                # So we can't just update local edges easily without checking all triangles involving them.
                
                # Given N=50, full recalculation of bad_edges is:
                # 1225 edges. For each, count intersection.
                # Intersection takes ~50 ops.
                # 1225 * 50 = 61k ops.
                # 50000 iters * 61k = 3e9 ops. Still too slow for Python loops.
                
                # We need a faster bad_edge update or a different strategy.
                
                # Alternative: Don't maintain bad_edges list.
                # Just pick a random edge, check if it's bad. If yes, fix it. If no, skip?
                # If we pick a random edge, probability of it being bad is (bad_count / total_edges).
                # Initially, bad_count is high. Later, it's low.
                # If bad_count is low, we waste time picking good edges.
                
                # Better: Maintain a list of bad edges, but update it incrementally?
                # Or: Use NumPy to vectorize the triangle count for ALL edges?
                
                # Let's try a different approach:
                # Keep the list of bad edges.
                # When we move, we know which triangles were broken/formed.
                # Let $T_{broken}$ be the set of triangles that were mono and are no longer.
                # Let $T_{formed}$ be the set of triangles that are now mono.
                # For each triangle in $T_{broken}$, its 3 edges might no longer be bad (if they were only in this triangle).
                # For each triangle in $T_{formed}$, its 3 edges are now bad.
                
                # This requires knowing the exact sets.
                
                # Given the time constraints and Python speed, let's optimize the `count_triangles_for_edge` and `get_bad_edges` with NumPy.
                
                # NumPy approach for get_bad_edges:
                # M_c is NxN matrix.
                # Triangle count for edge (i, j) with color c is (M_c @ M_c)[i, j]?
                # No. (M_c @ M_c)[i, j] = sum_k M_c[i, k] M_c[k, j].
                # This counts paths of length 2.
                # If M_c[i, j] = 1, then a triangle (i, j, k) exists if M_c[i, k]=1 and M_c[j, k]=1.
                # So the number of triangles containing edge (i, j) of color c is (M_c @ M_c)[i, j] - 1?
                # Wait. (M_c @ M_c)[i, j] sums over k.
                # If k=i, M_c[i, i]=0.
                # If k=j, M_c[j, j]=0.
                # So it sums over k != i, j.
                # So yes, if M_c[i, j]=1, then (M_c @ M_c)[i, j] is the number of k such that (i, k) and (j, k) are color c.
                # This is exactly the number of mono triangles containing edge (i, j).
                
                # So we can compute all "badness" at once.
                # But we need to do this after every move?
                # 4 matrix multiplications of 50x50.
                # 50^3 = 125,000 flops per matmul.
                # 4 * 125,000 = 500,000 flops.
                # In NumPy, this is extremely fast (< 1ms).
                # 50,000 iterations * 1ms = 50 seconds.
                # This fits in the 115s budget!
                
                # Let's switch to NumPy-based bad edge detection.
                
                # We will break out of the inner loop if time is running out.
                # And we will restart the SA loop if we get stuck?
                # No, SA with cooling should escape.
                
                # Let's refactor the loop to use NumPy for bad edge detection.
                
                pass # Placeholder for refactor

            # Cooling
            current_T *= cooling
            if current_T < 0.01:
                current_T = 0.01

        # If we didn't return, we failed this restart. Try next.

    # Fallback: Return the best found? We didn't track best.
    # If we didn't find 50, maybe try smaller N?
    # But the task says maximize n.
    # If we fail 50, we should return whatever we have.
    # But we didn't keep the state.
    # We should keep the best state found in any restart.
    
    # Let's wrap this in a function that returns the best state.

    # Revised Plan:
    # 1. Define a function `solve(N)` that runs SA for N.
    # 2. Call `solve(50)`. If success, return.
    # 3. If fail, call `solve(48)`, etc.
    
    # But `solve` needs to return the colors.
    
    # Let's implement `solve(N)` properly with the NumPy optimization.

    def solve(N):
        t_start = time.time()
        # Time budget per restart
        # We have ~110s total.
        # Let's allow 10 restarts of 10s each? Or one long run?
        # SA is good with long runs. Let's do one long run with restarts if stuck.
        
        for restart in range(10):
            if time.time() > DEADLINE - 5:
                return None
            
            # Init
            colors = np.zeros((N, N), dtype=np.int8)
            for i in range(N):
                for j in range(i + 1, N):
                    colors[i][j] = random.randint(0, 3)
                    colors[j][i] = colors[i][j]
            
            # Matrices M_c
            M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
            for i in range(N):
                for j in range(i + 1, N):
                    c = colors[i][j]
                    M[c][i][j] = 1
                    M[c][j][i] = 1
            
            def get_bad_edges():
                # Returns list of (i, j, c) where triangle count > 0
                bad = []
                # Pre-compute M_c @ M_c
                M2 = [M[c] @ M[c] for c in range(4)]
                
                for i in range(N):
                    for j in range(i + 1, N):
                        c = colors[i][j]
                        if M2[c][i][j] > 0:
                            bad.append((i, j, c))
                return bad

            bad_edges = get_bad_edges()
            
            current_T = 5.0
            cooling = 0.999 # Slow cooling for long run?
            # Or adaptive?
            
            max_iters = 20000
            for it in range(max_iters):
                if time.time() > DEADLINE - 5:
                    return None
                
                if not bad_edges:
                    res_colors = []
                    for i in range(N):
                        for j in range(i + 1, N):
                            res_colors.append(int(colors[i][j]))
                    return res_colors
                
                # Pick random bad edge
                u, v, c_old = random.choice(bad_edges)
                
                # Evaluate moves
                best_delta = float('inf')
                best_c = -1
                
                # We need M2 for current state to evaluate k_old and k_new?
                # k_old = M2[c_old][u][v]
                # k_new = M2[c_new][u][v]
                # This is valid because M2 is computed from current M.
                
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    k_old = M2[c_old][u][v]
                    k_new = M2[c_new][u][v]
                    delta = k_new - k_old
                    if delta < best_delta:
                        best_delta = delta
                        best_c = c_new
                
                # Accept
                accept = False
                if best_delta < 0:
                    accept = True
                else:
                    p = np.exp(-best_delta / current_T)
                    if random.random() < p:
                        accept = True
                
                if accept:
                    # Update colors
                    colors[u][v] = best_c
                    colors[v][u] = best_c
                    
                    # Update M matrices
                    M[c_old][u][v] = 0
                    M[c_old][v][u] = 0
                    M[best_c][u][v] = 1
                    M[best_c][v][u] = 1
                    
                    # Update M2 matrices?
                    # M2 = M @ M.
                    # Recalculating all 4 M2 is 4 matmuls.
                    # This is the cost per step.
                    M2 = [M[c] @ M[c] for c in range(4)]
                    
                    # Update bad_edges
                    bad_edges = get_bad_edges()
                    
                    # Recalculate bad_edges is O(N^2) loop + 4 matmuls.
                    # The loop is 1225 iterations.
                    # Matmuls are fast.
                    # Total ~ 1-2 ms per step.
                    # 20000 steps * 2ms = 40s.
                    # Fits in budget.
                
                current_T *= cooling
                if current_T < 0.01:
                    current_T = 0.01

        return None

    # Try N=50
    result = solve(50)
    if result:
        return {'n': 50, 'colors': result}
    
    # Try N=48
    result = solve(48)
    if result:
        return {'n': 48, 'colors': result}
        
    # Try N=46
    result = solve(46)
    if result:
        return {'n': 46, 'colors': result}

    # Fallback: Return a small valid one?
    # We should return something valid.
    # The seed returned 32.
    # Let's just return the seed's 32 if we fail completely?
    # Or just return n=1?
    # Better to return a known valid small one.
    
    # Construct K_4? No, R(3,3)=6, so K_5 is valid for 2 colors.
    # For 4 colors, K_1 is always valid.
    # Let's return the seed's K_32 construction to be safe?
    # It's complex to re-implement.
    # Let's just return n=1.
    
    return {'n': 1, 'colors': []}

# Wait, the seed program returns a dict.
# The function signature is construct() -> dict.

# One issue: `get_bad_edges` inside `solve` uses `M2` from the outer scope?
# No, `M2` is local to the `if accept` block?
# No, `M2` is defined in `get_bad_edges`?
# In my draft, `M2` was computed in `get_bad_edges`.
# But I also use `M2` in the main loop for delta calculation.
# So `M2` should be a variable in the `solve` scope, updated after each move.

# Let's clean up the code structure.

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0
    
    def solve(N):
        t_start = time.time()
        
        for restart in range(10):
            if time.time() > DEADLINE - 5:
                return None
            
            # Init
            colors = np.zeros((N, N), dtype=np.int8)
            for i in range(N):
                for j in range(i + 1, N):
                    colors[i][j] = random.randint(0, 3)
                    colors[j][i] = colors[i][j]
            
            M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
            for i in range(N):
                for j in range(i + 1, N):
                    c = colors[i][j]
                    M[c][i][j] = 1
                    M[c][j][i] = 1
            
            # Compute initial M2
            M2 = [M[c] @ M[c] for c in range(4)]
            
            def get_bad_edges():
                bad = []
                for i in range(N):
                    for j in range(i + 1, N):
                        c = colors[i][j]
                        if M2[c][i][j] > 0:
                            bad.append((i, j, c))
                return bad

            bad_edges = get_bad_edges()
            
            current_T = 5.0
            cooling = 0.999
            
            max_iters = 20000
            for it in range(max_iters):
                if time.time() > DEADLINE - 5:
                    return None
                
                if not bad_edges:
                    res_colors = []
                    for i in range(N):
                        for j in range(i + 1, N):
                            res_colors.append(int(colors[i][j]))
                    return res_colors
                
                # Pick random bad edge
                u, v, c_old = random.choice(bad_edges)
                
                # Evaluate moves
                best_delta = float('inf')
                best_c = -1
                
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    k_old = M2[c_old][u][v]
                    k_new = M2[c_new][u][v]
                    delta = k_new - k_old
                    if delta < best_delta:
                        best_delta = delta
                        best_c = c_new
                
                # Accept
                accept = False
                if best_delta < 0:
                    accept = True
                else:
                    p = np.exp(-best_delta / current_T)
                    if random.random() < p:
                        accept = True
                
                if accept:
                    # Update colors
                    colors[u][v] = best_c
                    colors[v][u] = best_c
                    
                    # Update M matrices
                    M[c_old][u][v] = 0
                    M[c_old][v][u] = 0
                    M[best_c][u][v] = 1
                    M[best_c][v][u] = 1
                    
                    # Update M2
                    M2 = [M[c] @ M[c] for c in range(4)]
                    
                    # Update bad_edges
                    bad_edges = get_bad_edges()
                
                current_T *= cooling
                if current_T < 0.01:
                    current_T = 0.01

        return None

    # Try N=50
    result = solve(50)
    if result:
        return {'n': 50, 'colors': result}
    
    # Try N=48
    result = solve(48)
    if result:
        return {'n': 48, 'colors': result}
        
    # Try N=46
    result = solve(46)
    if result:
        return {'n': 46, 'colors': result}

    # Fallback
    return {'n': 1, 'colors': []}

# Check imports
import time
import random
import numpy as np

# The code looks solid.
# Complexity:
# N=50.
# Matmul 50x50x50.
# 4 matmuls per step.
# 20000 steps.
# 80,000 matmuls.
# Each matmul ~ 125,000 flops.
# Total ~ 10^10 flops.
# NumPy is C-optimized. 10^10 flops is ~ 10-50 seconds depending on CPU.
# Might be tight.
# Optimization: Only recalculate M2 for the changed colors?
# M[c_old] and M[best_c] change.
# So only M2[c_old] and M2[best_c] need update.
# That's 2 matmuls instead of 4.
# Also, `get_bad_edges` iterates N^2/2.
# This is fast.

# Let's optimize the M2 update.

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0
    
    def solve(N):
        for restart in range(10):
            if time.time() > DEADLINE - 5:
                return None
            
            colors = np.zeros((N, N), dtype=np.int8)
            for i in range(N):
                for j in range(i + 1, N):
                    colors[i][j] = random.randint(0, 3)
                    colors[j][i] = colors[i][j]
            
            M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
            for i in range(N):
                for j in range(i + 1, N):
                    c = colors[i][j]
                    M[c][i][j] = 1
                    M[c][j][i] = 1
            
            M2 = [M[c] @ M[c] for c in range(4)]
            
            def get_bad_edges():
                bad = []
                for i in range(N):
                    for j in range(i + 1, N):
                        c = colors[i][j]
                        if M2[c][i][j] > 0:
                            bad.append((i, j, c))
                return bad

            bad_edges = get_bad_edges()
            
            current_T = 5.0
            cooling = 0.999
            
            max_iters = 20000
            for it in range(max_iters):
                if time.time() > DEADLINE - 5:
                    return None
                
                if not bad_edges:
                    res_colors = []
                    for i in range(N):
                        for j in range(i + 1, N):
                            res_colors.append(int(colors[i][j]))
                    return res_colors
                
                u, v, c_old = random.choice(bad_edges)
                
                best_delta = float('inf')
                best_c = -1
                
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    k_old = M2[c_old][u][v]
                    k_new = M2[c_new][u][v]
                    delta = k_new - k_old
                    if delta < best_delta:
                        best_delta = delta
                        best_c = c_new
                
                accept = False
                if best_delta < 0:
                    accept = True
                else:
                    p = np.exp(-best_delta / current_T)
                    if random.random() < p:
                        accept = True
                
                if accept:
                    colors[u][v] = best_c
                    colors[v][u] = best_c
                    
                    M[c_old][u][v] = 0
                    M[c_old][v][u] = 0
                    M[best_c][u][v] = 1
                    M[best_c][v][u] = 1
                    
                    # Only update M2 for affected colors
                    M2[c_old] = M[c_old] @ M[c_old]
                    M2[best_c] = M[best_c] @ M[best_c]
                    
                    bad_edges = get_bad_edges()
                
                current_T *= cooling
                if current_T < 0.01:
                    current_T = 0.01

        return None

    result = solve(50)
    if result:
        return {'n': 50, 'colors': result}
    
    result = solve(48)
    if result:
        return {'n': 48, 'colors': result}
        
    result = solve(46)
    if result:
        return {'n': 46, 'colors': result}

    return {'n': 1, 'colors': []}

# Final check on imports and syntax.
# Looks good.
