import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 3
    
    def total_error_fast(U, V, W, rank):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j][r] * V[j*p+k][r]
                    err += (s - 1)**2
        return err
    
    def check_fast(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j][r] * V[j*p+k][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find rank 17
    if time.time() < deadline - 5:
        random.seed(42)
        best_err = float('inf')
        best_sol = None
        
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            rank = 17
            U = [[0.0]*rank for _ in range(6)]
            V = [[0.0]*rank for _ in range(9)]
            W = [[0.0]*rank for _ in range(6)]
            
            # Random initialization
            for idx in range(6):
                for r in range(rank):
                    U[idx][r] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5])
            for idx in range(9):
                for r in range(rank):
                    V[idx][r] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5])
            for idx in range(6):
                for r in range(rank):
                    W[idx][r] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5])
            
            # Local search
            current_err = total_error_fast(U, V, W, rank)
            for step in range(5000):
                if time.time() > deadline - 5:
                    break
                
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randint(0, 5)
                    r = random.randint(0, 16)
                    old = U[idx][r]
                    delta = random.choice([-1, -0.5, 0.5, 1])
                    U[idx][r] = old + delta
                    new_err = total_error_fast(U, V, W, rank)
                    if new_err < current_err:
                        current_err = new_err
                        if new_err < best_err:
                            best_err = new_err
                            best_sol = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
                            if best_err < 1e-12:
                                break
                    else:
                        U[idx][r] = old
                elif mat == 'V':
                    idx = random.randint(0, 8)
                    r = random.randint(0, 16)
                    old = V[idx][r]
                    delta = random.choice([-1, -0.5, 0.5, 1])
                    V[idx][r] = old + delta
                    new_err = total_error_fast(U, V, W, rank)
                    if new_err < current_err:
                        current_err = new_err
                        if new_err < best_err:
                            best_err = new_err
                            best_sol = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
                            if best_err < 1e-12:
                                break
                    else:
                        V[idx][r] = old
                else:
                    idx = random.randint(0, 5)
                    r = random.randint(0, 16)
                    old = W[idx][r]
                    delta = random.choice([-1, -0.5, 0.5, 1])
                    W[idx][r] = old + delta
                    new_err = total_error_fast(U, V, W, rank)
                    if new_err < current_err:
                        current_err = new_err
                        if new_err < best_err:
                            best_err = new_err
                            best_sol = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
                            if best_err < 1e-12:
                                break
                    else:
                        W[idx][r] = old
            
            if best_err < 1e-12:
                break
        
        if best_sol is not None and best_err < 1e-12:
            U, V, W = best_sol
            if check_fast(U, V, W, 17):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    rank = 18
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
