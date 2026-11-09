import numpy as np
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Generate the seed n=32 coloring (Greenwood-Gleason / Clebsch)
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
        
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    # State:
    # n: current number of vertices
    # colors: list of colors for edges (i,j) i<j
    # A: list of 4 numpy arrays (n x n)
    
    def init_state(n, colors):
        A = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                A[c][i, j] = 1
                A[c][j, i] = 1
                idx += 1
        return A

    def count_violations(A, n):
        # Sum of triangles in each color
        # Trace(A^3) / 6
        # For speed, we can just sum A[i,j]*A[j,k]*A[k,i]
        # But NumPy trace is fast enough for n=50
        total = 0
        for c in range(4):
            # A[c] is symmetric
            # Triangles = trace(A^3) / 6
            # A^3 = A @ A @ A
            # Using einsum for speed?
            # np.einsum('ij,jk,ki->', A[c], A[c], A[c]) / 6
            t = np.einsum('ij,jk,ki->', A[c], A[c], A[c]) / 6
            total += t
        return total

    def get_edge_idx(i, j, n):
        # i < j
        # Index in flat array
        # Rows: 0..n-1
        # For row i, edges to i+1..n-1
        # Count edges before row i: sum_{k=0}^{i-1} (n-1-k)
        # This is (i-1)*(2*n-i)/2 ?
        # Let's just precompute or use a map
        # Or simply: idx = i*n - i*(i+1)//2 + (j-i-1)
        return i*n - i*(i+1)//2 + (j - i - 1)

    def set_edge_color(A, colors, i, j, new_c, n):
        old_c = colors[get_edge_idx(i, j, n)]
        if old_c == new_c:
            return 0 # No change
        
        # Update A
        A[old_c][i, j] = 0
        A[old_c][j, i] = 0
        A[new_c][i, j] = 1
        A[new_c][j, i] = 1
        colors[get_edge_idx(i, j, n)] = new_c
        
        # Calculate change in violations
        # Only triangles involving edge (i,j) are affected
        # Triangles (i, j, k) for k != i, j
        delta = 0
        for k in range(n):
            if k == i or k == j:
                continue
            
            # Check triangle (i, j, k)
            # Old state:
            # c_ij = old_c
            # c_ik = colors[get_edge_idx(i, k, n)] if i<k else colors[get_edge_idx(k, i, n)]
            # c_jk = colors[get_edge_idx(j, k, n)] if j<k else colors[get_edge_idx(k, j, n)]
            
            # Helper to get color
            def get_c(u, v):
                if u > v: u, v = v, u
                return colors[get_edge_idx(u, v, n)]
                
            c_ik = get_c(i, k)
            c_jk = get_c(j, k)
            
            # Was it mono?
            was_mono = (old_c == c_ik == c_jk)
            
            # Is it mono now?
            is_mono = (new_c == c_ik == c_jk)
            
            if was_mono and not is_mono:
                delta -= 1
            elif is_mono and not was_mono:
                delta += 1
                
        return delta

    # Local Search Loop
    def local_search(n, colors, A, max_steps=5000):
        current_v = count_violations(A, n)
        if current_v == 0:
            return True
        
        steps = 0
        while steps < max_steps and time.time() < deadline:
            # Find a monochromatic triangle
            # We can pick a random edge and check if it's in a mono triangle?
            # Or just pick a random vertex and look at its edges?
            # Efficient way: Pick a random edge (i,j).
            # If it's in a mono triangle, change it.
            
            # To ensure we find one if exists:
            # We can maintain a list of violating edges?
            # For simplicity, random sampling.
            
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i, j = j, i
            
            c_ij = colors[get_edge_idx(i, j, n)]
            
            # Check if (i,j) is in a mono triangle
            # Find k such that c_ik == c_jk == c_ij
            found_k = -1
            for k in range(n):
                if k == i or k == j: continue
                c_ik = colors[get_edge_idx(i, k, n)] if i < k else colors[get_edge_idx(k, i, n)]
                c_jk = colors[get_edge_idx(j, k, n)] if j < k else colors[get_edge_idx(k, j, n)]
                if c_ik == c_ij and c_jk == c_ij:
                    found_k = k
                    break
            
            if found_k == -1:
                # This edge is not in a mono triangle.
                # Try another edge.
                # To avoid infinite loop if no violations exist (should be caught by current_v==0)
                # But current_v might be stale? No, we update it.
                # If no violations, we return True.
                # So if we loop here, there are violations, but we missed them.
                # Increase max_steps or break.
                continue
            
            # Change color of (i,j)
            # Try other 3 colors
            new_c = random.choice([c for c in range(4) if c != c_ij])
            
            delta = set_edge_color(A, colors, i, j, new_c, n)
            
            if delta < 0:
                current_v += delta
                if current_v == 0:
                    return True
            elif delta == 0:
                # Accept to move around
                current_v += delta
            else:
                # Revert
                set_edge_color(A, colors, i, j, c_ij, n)
                
            steps += 1
            
        return current_v == 0

    # Extension Strategy
    best_n = 32
    best_colors = colors_32[:]
    
    current_n = 32
    current_colors = colors_32[:]
    current_A = init_state(current_n, current_colors)
    
    for target_n in range(33, 51):
        if time.time() > deadline:
            break
            
        # Try to add vertex target_n - 1
        success = False
        for attempt in range(50): # 50 restarts for this vertex
            if time.time() > deadline:
                break
                
            # Copy state
            trial_colors = current_colors[:]
            # Add new edges
            for u in range(current_n):
                trial_colors.append(random.randint(0, 3))
                
            trial_n = target_n
            trial_A = init_state(trial_n, trial_colors)
            
            # Run local search
            if local_search(trial_n, trial_colors, trial_A, max_steps=2000):
                current_n = trial_n
                current_colors = trial_colors
                current_A = trial_A
                success = True
                break
        
        if not success:
            # Failed to add this vertex
            break
            
        # Update best
        if current_n > best_n:
            best_n = current_n
            best_colors = current_colors[:]

    # Return best found
    return {'n': best_n, 'colors': best_colors}

