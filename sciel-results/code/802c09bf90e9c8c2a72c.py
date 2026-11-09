import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    
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

    def local_search_sa(n, max_iters=200000, initial_temp=5.0):
        # Initialize
        mat = np.random.randint(0, 4, (n, n), dtype=np.int8)
        np.fill_diagonal(mat, -1)
        mat = np.triu(mat, 1) + np.triu(mat, 1).T
        
        # Initial cost calculation
        def calc_cost(m):
            cost = 0
            for c in range(4):
                A = (m == c).astype(np.int8)
                np.fill_diagonal(A, 0)
                cost += np.trace(np.dot(np.dot(A, A), A)) // 6
            return int(cost)
            
        current_cost = calc_cost(mat)
        best_cost = current_cost
        best_mat = mat.copy()
        
        T = initial_temp
        cooling = 0.99995
        min_T = 0.01
        
        for it in range(max_iters):
            if time_up():
                break
                
            if current_cost == 0:
                return mat, True
            
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            old_color = int(mat[i, j])
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
            
            # Vectorized delta
            c_ik = mat[i, :]
            c_jk = mat[j, :]
            
            count_old = np.sum((c_ik == old_color) & (c_jk == old_color))
            count_new = np.sum((c_ik == new_color) & (c_jk == new_color))
            
            delta = count_new - count_old
            
            new_cost = current_cost + delta
            
            if new_cost < current_cost:
                mat[i, j] = new_color
                mat[j, i] = new_color
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_mat = mat.copy()
            elif delta == 0:
                if random.random() < 0.5:
                    mat[i, j] = new_color
                    mat[j, i] = new_color
                    current_cost = new_cost
            else:
                if random.random() < np.exp(-delta / max(T, 1e-10)):
                    mat[i, j] = new_color
                    mat[j, i] = new_color
                    current_cost = new_cost
            
            T = max(T * cooling, min_T)
            
        return best_mat, best_cost == 0

    def try_solve(n, num_restarts=5):
        for restart in range(num_restarts):
            if time_up():
                break
            random.seed(42 + restart * 123)
            np.random.seed(42 + restart * 123)
            
            result_mat, success = local_search_sa(n, max_iters=100000, initial_temp=5.0)
            if success:
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        colors.append(int(result_mat[i, j]))
                return {'n': n, 'colors': colors}
        return None

    best_solution = get_seed_solution()
    
    # Try to find larger n
    for n in [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]:
        if time_up():
            break
        
        # More restarts for higher n
        restarts = 5 if n >= 48 else 10
        
        result = try_solve(n, num_restarts=restarts)
        if result is not None:
            best_solution = result
            break # Found the max n in this descending order
            
    return best_solution
