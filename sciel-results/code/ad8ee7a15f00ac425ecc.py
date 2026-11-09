import time
import random
import copy

def construct():
    """
    Finds a 4-coloring of K_n with no monochromatic triangles, maximizing n.
    """
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Seed: Greenwood-Gleason coloring for n=32
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

    seed_sol = get_seed_solution()
    best_n = 32
    best_colors = seed_sol['colors']
    
    # Store the seed coloring in a 2D array
    N_seed = 32
    seed_c = [[0] * N_seed for _ in range(N_seed)]
    idx = 0
    for i in range(N_seed):
        for j in range(i + 1, N_seed):
            seed_c[i][j] = best_colors[idx]
            seed_c[j][i] = best_colors[idx]
            idx += 1

    def count_bad(c, N):
        """Count total number of monochromatic triangle contributions."""
        A = [[[[0] * N for _ in range(N)] for _ in range(4)]]
        total_bad = 0
        for col in range(4):
            for u in range(N):
                for v in range(u + 1, N):
                    if c[u][v] == col:
                        count = 0
                        for w in range(N):
                            if w == u or w == v:
                                continue
                            if (c[u][w] == col and c[v][w] == col):
                                count += 1
                        total_bad += count
        return total_bad

    def try_add_vertex(c, N, time_limit):
        """Try to add a new vertex at position N using backtracking."""
        # New vertex index is N
        # We need to assign colors to edges (i, N) for i in 0..N-1
        
        # Precompute constraints: for each i, which colors are forbidden for edge (i, N)
        # given the current assignments for edges (k, N) where k < i
        # Actually, we need to check triangles (k, i, N) for all k < i
        
        # Let's use a recursive backtracking with pruning
        assignments = [0] * N  # assignments[i] = color of edge (i, N)
        
        def backtrack(i):
            if i == N:
                return True
            
            # Try each color for edge (i, N)
            for col in range(4):
                assignments[i] = col
                # Check if this creates a monochromatic triangle with any k < i
                valid = True
                for k in range(i):
                    if c[k][i] == assignments[k] == col:
                        valid = False
                        break
                if valid:
                    if backtrack(i + 1):
                        return True
            return False
        
        # Randomize the order of colors to try different solutions
        colors_order = list(range(4))
        random.shuffle(colors_order)
        
        def backtrack_random(i):
            if i == N:
                return True
            
            for col in colors_order:
                assignments[i] = col
                valid = True
                for k in range(i):
                    if c[k][i] == assignments[k] == col:
                        valid = False
                        break
                if valid:
                    if backtrack_random(i + 1):
                        return True
            return False
        
        start = time.time()
        # Try a few random orders
        for attempt in range(10):
            if time.time() - start > time_limit:
                return None
            colors_order = list(range(4))
            random.shuffle(colors_order)
            if backtrack_random(0):
                # Success: extend the coloring
                new_c = [row[:] for row in c]
                for i in range(N):
                    new_c[i].append(assignments[i])
                    new_c[i][N] = assignments[i]
                new_c.append([0] * (N + 1))
                for i in range(N):
                    new_c[N][i] = assignments[i]
                return new_c
        return None

    def local_search(c, N, time_limit):
        """Local search to reduce monochromatic triangles."""
        # Build A matrix: A[col][u][v] = number of w such that c[u][w]==col and c[v][w]==col
        A = [[[[0] * N for _ in range(N)] for _ in range(4)]]
        for col in range(4):
            for u in range(N):
                for v in range(N):
                    if u == v:
                        continue
                    count = 0
                    for w in range(N):
                        if w == u or w == v:
                            continue
                        if c[u][w] == col and c[v][w] == col:
                            count += 1
                    A[col][u][v] = count
        
        total_bad = sum(A[col][u][v] for col in range(4) for u in range(N) for v in range(u + 1, N))
        
        if total_bad == 0:
            flat = []
            for i in range(N):
                for j in range(i + 1, N):
                    flat.append(c[i][j])
            return flat
        
        start_time = time.time()
        
        while time.time() < start_time + time_limit:
            best_delta = 1000000
            best_u, best_v, best_col = -1, -1, -1
            
            for u in range(N):
                for v in range(u + 1, N):
                    old_col = c[u][v]
                    for new_col in range(4):
                        if new_col == old_col:
                            continue
                        delta = A[new_col][u][v] - A[old_col][u][v]
                        if delta < best_delta:
                            best_delta = delta
                            best_u, best_v, best_col = u, v, new_col
            
            if best_u == -1:
                break

            if best_delta > 0:
                # Local minimum: make a random move
                if random.random() < 0.7:
                    u = random.randint(0, N - 1)
                    v = random.randint(0, N - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    old_col = c[u][v]
                    new_col = random.choice([0, 1, 2, 3])
                    if new_col == old_col:
                        continue
                    best_delta = A[new_col][u][v] - A[old_col][u][v]
                    best_col = new_col
                    best_u, best_v = u, v
                else:
                    continue

            # Apply move
            old_col = c[best_u][best_v]
            new_col = best_col
            c[best_u][best_v] = new_col
            c[best_v][best_u] = new_col
            
            u, v = best_u, best_v
            
            for w in range(N):
                if w == u or w == v:
                    continue
                
                col_vw = c[v][w] if v < w else c[w][v]
                
                if col_vw == old_col:
                    A[old_col][u][w] -= 1
                    A[old_col][w][u] -= 1
                if col_vw == new_col:
                    A[new_col][u][w] += 1
                    A[new_col][w][u] += 1
                    
                col_uw = c[u][w] if u < w else c[w][u]
                
                if col_uw == old_col:
                    A[old_col][v][w] -= 1
                    A[old_col][w][v] -= 1
                if col_uw == new_col:
                    A[new_col][v][w] += 1
                    A[new_col][w][v] += 1
            
            total_bad += best_delta
            
            if total_bad == 0:
                flat = []
                for i in range(N):
                    for j in range(i + 1, N):
                        flat.append(c[i][j])
                return flat

        return None

    # Strategy: Start from seed, try to extend by adding vertices
    # Then use local search to fix any issues
    
    current_n = N_seed
    current_c = [row[:] for row in seed_c]
    
    # Try to extend from 32 to higher values
    for target_n in range(33, 63):
        if time.time() > deadline - 5.0:
            break
        
        # Try to add vertex at position current_n
        remaining_time = deadline - time.time() - 5.0
        if remaining_time < 2.0:
            break
        
        # Allocate time: more time for lower target_n
        if target_n <= 40:
            t_add = min(5.0, remaining_time * 0.3)
            t_search = min(10.0, remaining_time * 0.5)
        elif target_n <= 45:
            t_add = min(3.0, remaining_time * 0.2)
            t_search = min(8.0, remaining_time * 0.4)
        else:
            t_add = min(2.0, remaining_time * 0.1)
            t_search = min(5.0, remaining_time * 0.3)
        
        # Try to add vertex
        new_c = try_add_vertex(current_c, current_n, t_add)
        if new_c is not None:
            current_n = target_n
            current_c = new_c
            # Update best
            flat = []
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    flat.append(current_c[i][j])
            best_n = current_n
            best_colors = flat
            continue
        
        # If adding failed, try local search on current coloring to see if we can fix it
        # But we need to first create a candidate with the new vertex (randomly)
        # Then use local search to fix it
        
        # Create a candidate with the new vertex randomly colored
        candidate_c = [row[:] for row in current_c]
        for i in range(current_n):
            candidate_c[i].append(random.randint(0, 3))
            candidate_c[i][current_n] = random.randint(0, 3)
        candidate_c.append([0] * (current_n + 1))
        for i in range(current_n):
            candidate_c[current_n][i] = candidate_c[i][current_n]
        
        # Run local search
        sol = local_search(candidate_c, target_n, t_search)
        if sol is not None:
            current_n = target_n
            # Rebuild current_c from sol
            current_c = [[0] * target_n for _ in range(target_n)]
            idx = 0
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    current_c[i][j] = sol[idx]
                    current_c[j][i] = sol[idx]
                    idx += 1
            best_n = target_n
            best_colors = sol
            continue
        
        # If we can't extend to this target, stop
        break
    
    # Final: try local search on the current best to see if we can improve
    if time.time() < deadline - 5.0:
        t_final = min(30.0, deadline - time.time() - 2.0)
        # Try local search on a copy of the best coloring
        test_c = [[0] * best_n for _ in range(best_n)]
        idx = 0
        for i in range(best_n):
            for j in range(i + 1, best_n):
                test_c[i][j] = best_colors[idx]
                test_c[j][i] = best_colors[idx]
                idx += 1
        
        sol = local_search(test_c, best_n, t_final)
        if sol is not None:
            # Verify it's valid (should be, but double-check)
            best_colors = sol
    
    return {'n': best_n, 'colors': best_colors}
