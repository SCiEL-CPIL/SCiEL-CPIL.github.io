import time
import random

def construct():
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16) (the Clebsch graph three times),
    # doubled: two copies with colour 3 on all edges between them (a bipartite colour class).
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
    
    # Build initial n=32 coloring
    n = 32
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    # Convert to matrix for easy access
    # M[i][j] = color
    M = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            M[i][j] = c
            M[j][i] = c
            idx += 1

    def try_extend(M, n, max_n=50, time_limit=10.0):
        start_time = time.time()
        # Try to extend to n+1
        # We need to assign colors to new vertex n
        # Let col[i] be the color of edge (n, i)
        
        # Backtracking
        # State: col[0..n-1]
        # We can use a simple recursive search
        
        def backtrack(i, col):
            if time.time() - start_time > time_limit:
                return False
            if i == n:
                return True
            for c in range(4):
                # Check if color c is valid for vertex i
                valid = True
                for j in range(i):
                    if col[j] == c and M[i][j] == c:
                        valid = False
                        break
                if valid:
                    col[i] = c
                    if backtrack(i+1, col):
                        return True
            return False

        col = [0]*n
        if backtrack(0, col):
            # Success, update M
            # Add row/col
            new_M = [row[:] for row in M]
            new_M.append([0]*(n+1))
            for i in range(n):
                new_M[i].append(col[i])
                new_M[n][i] = col[i]
            new_M[n][n] = 0
            return True, new_M, n+1
        return False, M, n

    # Try to extend up to 50 or time limit
    best_n = n
    current_M = M
    current_n = n
    
    # We have 150s total. Reserve 10s for each extension attempt?
    # Or just try a few.
    # Let's try to extend one by one.
    
    while current_n < 50:
        # Time check
        if time.time() > 140:
            break
            
        success, new_M, new_n = try_extend(current_M, current_n, max_n=current_n+1, time_limit=5.0)
        if success:
            current_M = new_M
            current_n = new_n
            best_n = new_n
        else:
            # If we fail to extend, we might try to repair or just stop.
            # For now, stop.
            break
            
    # Convert back to list format
    final_colors = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            final_colors.append(current_M[i][j])
            
    return {'n': best_n, 'colors': final_colors}
