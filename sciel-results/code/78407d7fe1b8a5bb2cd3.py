import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    def get_seed_solution():
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
        return {'n': 32, 'colors': colors}

    def count_triangles(mat):
        n = mat.shape[0]
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = mat[i][j]
                for k in range(j+1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        count += 1
        return count

    def find_mono_triangle(mat):
        n = mat.shape[0]
        # Return (i, j, k) of a mono triangle or None
        # Random sampling is faster if sparse
        for _ in range(100):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            k = random.randint(0, n-1)
            if i == j or j == k or i == k:
                continue
            c1 = mat[i][j]
            c2 = mat[i][k]
            c3 = mat[j][k]
            if c1 == c2 == c3:
                return (i, j, k)
        # Fallback to full scan if not found by sampling (should be rare if count is low)
        # But if count is high, sampling finds it quickly.
        # If count is 0, this returns None.
        # To be safe, if we suspect triangles exist but sampling failed, do a full scan?
        # For n <= 40, full scan is 64k ops, negligible.
        for i in range(n):
            for j in range(i+1, n):
                c = mat[i][j]
                for k in range(j+1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        return (i, j, k)
        return None

    def local_search_sa(mat, n, time_budget):
        current_cost = count_triangles(mat)
        if current_cost == 0:
            return mat, True
        
        best_mat = mat.copy()
        best_cost = current_cost
        
        T = 1.0
        cooling = 0.9999
        start = time.time()
        
        while time.time() - start < time_budget:
            if current_cost == 0:
                return mat, True
            
            # Determine move
            tri = find_mono_triangle(mat)
            if tri:
                i, j, k = tri
                # Pick random edge from the triangle
                edge_idx = random.randint(0, 2)
                if edge_idx == 0:
                    u, v = i, j
                elif edge_idx == 1:
                    u, v = i, k
                else:
                    u, v = j, k
            else:
                # No mono triangle found by sampling? 
                # If cost > 0, it must exist. 
                # If cost == 0, we break.
                # So if tri is None, it means cost is likely 0 or sampling failed.
                # Given we checked cost > 0, it's a sampling miss.
                # Just pick random edge.
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue

            old_color = mat[u][v]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
            
            # Calculate delta
            delta = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                # Old
                if mat[u][w] == old_color and mat[v][w] == old_color:
                    delta -= 1
                # New
                if mat[u][w] == new_color and mat[v][w] == new_color:
                    delta += 1
            
            new_cost = current_cost + delta
            
            # Acceptance
            if new_cost < current_cost:
                mat[u][v] = new_color
                mat[v][u] = new_color
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
            elif delta == 0:
                mat[u][v] = new_color
                mat[v][u] = new_color
                current_cost = new_cost
            else:
                # Metropolis criterion
                if random.random() < np.exp(-delta / max(T, 1e-10)):
                    mat[u][v] = new_color
                    mat[v][u] = new_color
                    current_cost = new_cost
            
            T *= cooling
            T = max(T, 0.01)
            
            if time.time() > deadline:
                break
        
        return best_mat, best_cost == 0

    # Main Logic
    # Start with n=32
    seed_res = get_seed_solution()
    n = 32
    colors = seed_res['colors']
    
    # Convert to matrix
    mat = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            mat[i][j] = colors[idx]
            mat[j][i] = colors[idx]
            idx += 1
    
    best_n = n
    best_colors = colors
    
    # Try to extend
    for target_n in range(33, 45): # Try up to 44
        if time.time() > deadline - 10:
            break
        
        # Extend matrix
        new_mat = np.zeros((target_n, target_n), dtype=np.int8)
        new_mat[:n, :n] = mat
        
        # Initialize new edges randomly
        for i in range(n, target_n):
            for j in range(i+1, target_n):
                c = random.randint(0, 3)
                new_mat[i][j] = c
                new_mat[j][i] = c
        for i in range(n):
            for j in range(n, target_n):
                c = random.randint(0, 3)
                new_mat[i][j] = c
                new_mat[j][i] = c
        
        # Run SA
        # Budget: 5 seconds per attempt
        time_budget = 5.0
        if time.time() + time_budget > deadline - 5:
             time_budget = max(0.1, deadline - 5 - time.time())
             
        res_mat, success = local_search_sa(new_mat, target_n, time_budget)
        
        if success:
            # Update best
            n = target_n
            mat = res_mat
            best_n = n
            # Convert back to colors list
            best_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    best_colors.append(int(mat[i][j]))
            # Continue to next target
        else:
            # If failed, we can stop trying larger n?
            # Or try again with a different random seed?
            # Given time constraints, if we fail n=33, n=34 is harder.
            # But maybe n=33 is easy and we just need more time?
            # Let's try one more restart for the current target_n if we have time.
            if time.time() < deadline - 10:
                 # One more attempt
                 new_mat2 = np.zeros((target_n, target_n), dtype=np.int8)
                 new_mat2[:n, :n] = mat
                 for i in range(n, target_n):
                     for j in range(i+1, target_n):
                         c = random.randint(0, 3)
                         new_mat2[i][j] = c
                         new_mat2[j][i] = c
                 for i in range(n):
                     for j in range(n, target_n):
                         c = random.randint(0, 3)
                         new_mat2[i][j] = c
                         new_mat2[j][i] = c
                 
                 res_mat2, success2 = local_search_sa(new_mat2, target_n, time_budget)
                 if success2:
                     n = target_n
                     mat = res_mat2
                     best_n = n
                     best_colors = []
                     for i in range(n):
                         for j in range(i+1, n):
                             best_colors.append(int(mat[i][j]))
                 else:
                     # Break if we failed twice
                     break
            else:
                break

    return {'n': best_n, 'colors': best_colors}
