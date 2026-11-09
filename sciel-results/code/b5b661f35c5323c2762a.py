import numpy as np
from scipy.optimize import linprog
import time

def squares_in_square10() -> tuple:
    n = 10
    
    def solve_lp(sep_matrix):
        c = np.zeros(3*n)
        c[:n] = -1
        
        A_ub = []
        b_ub = []
        
        for i in range(n):
            row = np.zeros(3*n)
            row[i] = 0.5
            row[n+i] = -1
            A_ub.append(row)
            b_ub.append(0)
            
            row = np.zeros(3*n)
            row[i] = 0.5
            row[n+i] = 1
            A_ub.append(row)
            b_ub.append(1)
            
            row = np.zeros(3*n)
            row[i] = 0.5
            row[2*n+i] = -1
            A_ub.append(row)
            b_ub.append(0)
            
            row = np.zeros(3*n)
            row[i] = 0.5
            row[2*n+i] = 1
            A_ub.append(row)
            b_ub.append(1)
            
        for i in range(n):
            for j in range(i+1, n):
                sep = sep_matrix[i][j]
                row = np.zeros(3*n)
                row[i] = 0.5
                row[j] = 0.5
                
                if sep == 0: # i left of j
                    row[n+i] = 1
                    row[n+j] = -1
                elif sep == 1: # i right of j
                    row[n+i] = -1
                    row[n+j] = 1
                elif sep == 2: # i below j
                    row[2*n+i] = 1
                    row[2*n+j] = -1
                elif sep == 3: # i above j
                    row[2*n+i] = -1
                    row[2*n+j] = 1
                    
                A_ub.append(row)
                b_ub.append(0)
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None)] * n + [(None, None)] * (2*n)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            s = res.x[:n]
            x = res.x[n:2*n]
            y = res.x[2*n:3*n]
            centers = np.column_stack((x, y))
            sides = s
            return centers, sides, res.fun * -1
        else:
            return None, None, -np.inf

    sep = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            sep[i][j] = np.random.randint(0, 4)
            
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    start_time = time.time()
    time_limit = 300
    
    centers, sides, score = solve_lp(sep)
    if centers is not None and score > best_score:
        best_centers = centers
        best_sides = sides
        best_score = score
        
    while time.time() - start_time < time_limit:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
            
        if i > j:
            i, j = j, i
            
        current = sep[i][j]
        # Try flipping
        if current == 0: new_val = 1
        elif current == 1: new_val = 0
        elif current == 2: new_val = 3
        else: new_val = 2
        
        sep[i][j] = new_val
        centers, sides, score = solve_lp(sep)
        
        if centers is not None and score > best_score:
            best_centers = centers.copy()
            best_sides = sides.copy()
            best_score = score
        else:
            sep[i][j] = current
            
    if best_centers is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(10, side)
        return centers, sides
        
    return best_centers, best_sides
