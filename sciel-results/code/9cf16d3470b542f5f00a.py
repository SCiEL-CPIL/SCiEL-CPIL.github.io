"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    random.seed(42)
    
    # Build the known K16 3-coloring (Clebsch graph) as a baseline
    def _mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    _log, _x = {}, 1
    for _e in range(15):
        _log[_x] = _e
        _x = _mul(_x, 2)
    
    # K16 3-coloring from Clebsch graph
    k16_mat = [[0] * 16 for _ in range(16)]
    for i in range(16):
        for j in range(i + 1, 16):
            c = _log[i ^ j] % 3
            k16_mat[i][j] = k16_mat[j][i] = c
    
    # Double to K32: 4-coloring
    def _double_3to4(mat, m):
        n = 2 * m
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // m == j // m:
                    colors.append(mat[i % m][j % m])
                else:
                    colors.append(3)
        return colors
    
    best_colors = _double_3to4(k16_mat, 16)
    best_n = 32
    
    # Direct 4-coloring search for n > 32
    # Use efficient triangle counting and simulated annealing
    
    def _count_mono_triangles(n, col_mat):
        """Count monochromatic triangles efficiently."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                cij = col_mat[i][j]
                # For each k > j, check if (i,k) and (j,k) both have color cij
                for k in range(j + 1, n):
                    if col_mat[i][k] == cij and col_mat[j][k] == cij:
                        count += 1
        return count
    
    def _delta_for_edge_change(n, col_mat, i, j, old_c, new_c):
        """Compute change in mono-triangle count when edge (i,j) changes from old_c to new_c."""
        delta = 0
        for k in range(n):
            if k == i or k == j:
                continue
            cik = col_mat[i][k]
            cjk = col_mat[j][k]
            # Before: triangle (i,j,k) is mono if all three edges = old_c
            before = 1 if (old_c == cik == cjk) else 0
            # After: triangle (i,j,k) is mono if all three edges = new_c
            after = 1 if (new_c == cik == cjk) else 0
            delta += after - before
        return delta
    
    def _sa_search(n, timeout):
        """Simulated annealing to find triangle-free 4-coloring of K_n."""
        start = time.time()
        
        # Greedy initialization: assign colors to minimize immediate conflicts
        col_mat = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                # Try each color, pick the one that creates fewest mono triangles
                best_c = 0
                best_delta = float('inf')
                for c in range(4):
                    # Count how many k < i already form a mono triangle with (i,j)=c
                    cnt = 0
                    for k in range(i):
                        if col_mat[i][k] == c and col_mat[j][k] == c:
                            cnt += 1
                    if cnt < best_delta:
                        best_delta = cnt
                        best_c = c
                    if cnt == 0:
                        break
                col_mat[i][j] = col_mat[j][i] = best_c
        
        cur_count = _count_mono_triangles(n, col_mat)
        if cur_count == 0:
            return col_mat
        
        best_col = [row[:] for row in col_mat]
        best_count = cur_count
        
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        while time.time() - start < timeout and T > T_min:
            T *= cooling
            
            # Pick a random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = col_mat[i][j]
            new_c = (old_c + random.randint(1, 3)) % 4
            if new_c == old_c:
                continue
            
            delta = _delta_for_edge_change(n, col_mat, i, j, old_c, new_c)
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                col_mat[i][j] = col_mat[j][i] = new_c
                cur_count += delta
                if cur_count < best_count:
                    best_count = cur_count
                    best_col = [row[:] for row in col_mat]
                    if best_count == 0:
                        return best_col
        
        return best_col if best_count == 0 else None
    
    # Try to find 4-colorings for n = 33, 34, ..., up to 50
    for target_n in range(33, 51):
        if time.time() > deadline - 5.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        # Allocate time for this n: more time for larger n
        alloc = min(remaining * 0.6, 30.0)
        
        found = None
        # Multiple restarts
        max_restarts = 3
        for restart in range(max_restarts):
            t_left = alloc / (max_restarts - restart)
            if time.time() > deadline - 5.0:
                break
            result = _sa_search(target_n, t_left)
            if result is not None:
                found = result
                break
        
        if found is not None:
            # Verify
            m = target_n
            valid = True
            for i in range(m):
                for j in range(i + 1, m):
                    cij = found[i][j]
                    for k in range(j + 1, m):
                        if found[i][k] == cij and found[j][k] == cij:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Convert to lexicographic edge list
                best_colors = []
                for i in range(m):
                    for j in range(i + 1, m):
                        best_colors.append(found[i][j])
                best_n = m
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
