import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    def time_remaining():
        return deadline - (time.time() - start_time)
    
    def time_up():
        return time.time() - start_time >= deadline - 2.0
    
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
    
    def build_matrix(colors, n):
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
    
    def count_mono_triangles(mat, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if mat[i, j] == mat[i, k] == mat[j, k]:
                        count += 1
        return count
    
    def local_search_sa(mat, n, max_iters=50000, initial_temp=2.0):
        """Simulated annealing local search."""
        mat = mat.copy()
        current_cost = count_mono_triangles(mat, n)
        
        if current_cost == 0:
            return mat, True
        
        best_mat = mat.copy()
        best_cost = current_cost
        
        T = initial_temp
        cooling = 0.9995
        min_T = 0.01
        
        for it in range(max_iters):
            if time_up():
                break
            if current_cost == 0:
                return mat, True
            
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            old_color = int(mat[i, j])
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
            
            # Compute delta efficiently
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                c_ik = int(mat[i, k])
                c_jk = int(mat[j, k])
                # Old triangle (i,j,k)
                if old_color == c_ik == c_jk:
                    delta -= 1
                # New triangle (i,j,k)
                if new_color == c_ik == c_jk:
                    delta += 1
            
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
            
            T = max(T * cooling, min_T)
        
        return best_mat, best_cost == 0
    
    def try_solve(n, num_restarts=5):
        """Try to find valid coloring for K_n using multiple SA restarts."""
        best_result = None
        
        for restart in range(num_restarts):
            if time_up():
                break
            
            random.seed(42 + restart * 1000)
            
            # Initialize with random colors
            mat = np.random.randint(0, 4, (n, n), dtype=np.int8)
            # Make symmetric
            for i in range(n):
                for j in range(i + 1, n):
                    mat[j, i] = mat[i, j]
            
            result_mat, success = local_search_sa(mat, n, max_iters=30000, initial_temp=2.0)
            
            if success:
                colors = flatten_matrix(result_mat)
                return {'n': n, 'colors': colors}
        
        return None
    
    def try_extend_from_seed(target_n, num_restarts=3):
        """Try to extend the 32-vertex seed solution to target_n."""
        seed_sol = get_seed_solution()
        mat32 = build_matrix(seed_sol['colors'], 32)
        
        for restart in range(num_restarts):
            if time_up():
                break
            
            random.seed(999 + restart * 777)
            
            # Extend matrix
            new_mat = np.zeros((target_n, target_n), dtype=np.int8)
            new_mat[:32, :32] = mat32
            
            # Fill new edges with random colors
            for i in range(32, target_n):
                for j in range(i + 1, target_n):
                    c = random.randint(0, 3)
                    new_mat[i, j] = c
                    new_mat[j, i] = c
            for i in range(32):
                for j in range(32, target_n):
                    c = random.randint(0, 3)
                    new_mat[i, j] = c
                    new_mat[j, i] = c
            
            result_mat, success = local_search_sa(new_mat, target_n, max_iters=25000, initial_temp=1.5)
            
            if success:
                colors = flatten_matrix(result_mat)
                return {'n': target_n, 'colors': colors}
        
        return None
    
    # Strategy: Try to find valid colorings for increasing n
    # Start with known 32, try to go higher
    
    best_solution = get_seed_solution()
    
    # Try extending from 32 to larger sizes
    for target in range(33, 40):
        if time_up():
            break
        
        result = try_extend_from_seed(target, num_restarts=2)
        if result is not None:
            best_solution = result
            # Continue trying to go higher
            continue
    
    # If we got a good extension, try even higher
    current_n = best_solution['n']
    if current_n > 32:
        for target in range(current_n + 1, 45):
            if time_up():
                break
            result = try_extend_from_seed(target, num_restarts=1)
            if result is not None:
                best_solution = result
                current_n = target
    
    # Also try pure random restarts for various sizes
    for n in [34, 35, 36, 37]:
        if time_up():
            break
        if n <= best_solution['n']:
            continue
        result = try_solve(n, num_restarts=2)
        if result is not None:
            best_solution = result
    
    return best_solution
