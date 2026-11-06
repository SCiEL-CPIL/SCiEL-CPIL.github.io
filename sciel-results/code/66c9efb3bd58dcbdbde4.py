import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Generate the seed n=32 coloring
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
        
    seed_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            seed_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    # Build matrix for seed
    def colors_to_matrix(colors, n):
        mat = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                mat[i][j] = c
                mat[j][i] = c
                idx += 1
        return mat

    best_n = 32
    best_matrix = colors_to_matrix(seed_colors, 32)

    def matrix_to_colors(mat, n):
        cols = []
        for i in range(n):
            for j in range(i + 1, n):
                cols.append(mat[i][j])
        return cols

    def is_valid(mat, n):
        # Check for monochromatic triangles
        for i in range(n):
            for j in range(i + 1, n):
                c = mat[i][j]
                for k in range(j + 1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        return False
        return True

    def try_add_vertex(mat, n):
        """
        Try to add vertex n to the graph of size n.
        Returns the new color array x[0..n-1] if successful, else None.
        """
        # Local search for x
        # Cost: number of pairs (i, j) such that x[i] == x[j] == mat[i][j]
        
        def cost(x):
            c = 0
            for i in range(n):
                xi = x[i]
                for j in range(i + 1, n):
                    if xi == x[j] and xi == mat[i][j]:
                        c += 1
            return c

        # Initialize x randomly
        # We can try to initialize smartly:
        # For each i, pick the color that minimizes conflicts with existing neighbors?
        # Simple random is okay with enough restarts.
        
        best_x = None
        best_cost = float('inf')
        
        # Simulated Annealing
        temp = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        for restart in range(20):
            x = [random.randint(0, 3) for _ in range(n)]
            curr_cost = cost(x)
            if curr_cost == 0:
                return x
            
            temp = 1.0
            stuck_counter = 0
            
            while temp > min_temp:
                if time.time() - start_time > time_limit:
                    return None
                
                # Pick random i
                i = random.randint(0, n - 1)
                old_c = x[i]
                new_c = random.choice([c for c in range(4) if c != old_c])
                x[i] = new_c
                
                new_cost = cost(x)
                delta = new_cost - curr_cost
                
                if delta < 0:
                    curr_cost = new_cost
                    stuck_counter = 0
                    if curr_cost == 0:
                        return x
                else:
                    # Accept with probability exp(-delta / temp)
                    if random.random() < pow(2.71828, -delta / temp):
                        curr_cost = new_cost
                        stuck_counter += 1
                    else:
                        x[i] = old_c
                        stuck_counter += 1
                
                temp *= cooling
                
                if stuck_counter > 100:
                    break
                    
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_x = x[:]
                
        if best_x is not None and cost(best_x) == 0:
            return best_x
            
        return None

    def grow_from(mat, start_n, target_n):
        """
        Attempts to grow the graph from start_n to target_n.
        Returns the final matrix if successful, else None.
        """
        current_n = start_n
        current_mat = [row[:] for row in mat]
        
        while current_n < target_n:
            if time.time() - start_time > time_limit:
                return None
                
            # Try to add vertex current_n
            success = False
            # Try multiple times with different random seeds implicitly via internal restarts
            for attempt in range(5):
                x = try_add_vertex(current_mat, current_n)
                if x is not None:
                    # Commit
                    # Add row/col to matrix
                    current_mat.append([0] * (current_n + 1))
                    for i in range(current_n):
                        current_mat[i].append(x[i])
                        current_mat[current_n][i] = x[i]
                    current_mat[current_n][current_n] = 0 # Self loop, irrelevant
                    current_n += 1
                    success = True
                    break
            
            if not success:
                return None
                
        return current_mat

    # Strategy 1: Extend the seed n=32 to n=50
    # The seed is a specific structure. It might be hard to extend.
    # Let's try.
    
    # We will try to extend the seed.
    # Note: The seed is valid.
    
    # To increase chances, we can try to "shrink" the problem? No.
    # Let's just try to extend.
    
    # It is possible that the seed structure is "maximal" in some sense or hard to extend.
    # Chung's construction is different.
    # If extension fails, we might need to start from scratch.
    
    # Let's try extending the seed.
    target_n = 50
    result_mat = grow_from(best_matrix, 32, target_n)
    
    if result_mat is not None:
        best_n = target_n
        best_matrix = result_mat
    else:
        # If extending the seed failed, try growing from scratch?
        # Growing from scratch to 50 is very hard with simple greedy.
        # But we can try to grow to a smaller N, say 40, from scratch?
        # Or maybe the seed extension just needed more attempts?
        
        # Let's try a few more times to extend the seed with different random states.
        # The internal `try_add_vertex` already restarts.
        # Let's try restarting the whole `grow_from` loop a few times.
        
        for _ in range(3):
            if time.time() - start_time > time_limit:
                break
            result_mat = grow_from(best_matrix, 32, target_n)
            if result_mat is not None:
                best_n = target_n
                best_matrix = result_mat
                break

    # If we still only have 32, we return 32.
    # Is it possible to do better than 32?
    # Yes, 50 is the record.
    # If we can't reach 50, maybe 40?
    # The `grow_from` function stops if it fails at any step.
    # If it fails at step 33, it returns None.
    
    # Let's try to grow to 40 if 50 failed.
    if best_n == 32:
        target_n_40 = 40
        result_mat = grow_from(best_matrix, 32, target_n_40)
        if result_mat is not None:
            best_n = target_n_40
            best_matrix = result_mat
            
    if best_n == 32:
        target_n_36 = 36
        result_mat = grow_from(best_matrix, 32, target_n_36)
        if result_mat is not None:
            best_n = target_n_36
            best_matrix = result_mat

    # Final check and conversion
    colors = matrix_to_colors(best_matrix, best_n)
    
    # Verify validity one last time (safety check)
    # If invalid, return n=1 (trivially valid) or the seed?
    # The seed is known valid.
    if not is_valid(best_matrix, best_n):
        # Fallback to seed
        best_n = 32
        colors = seed_colors
        
    return {'n': best_n, 'colors': colors}

# EVOLVE-BLOCK-END
