import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    start_time = time.time()
    deadline = start_time + 130
    
    def verify_float(U, V, W, atol=1e-9):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > atol:
                        return False
        return True
    
    def verify_exact(U, V, W):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != Fraction(1):
                        return False
        return True
    
    # Schoolbook fallback (rank 18)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(U_size)]
    V_school = [[0] * rank_school for _ in range(V_size)]
    W_school = [[0] * rank_school for _ in range(W_size)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def compute_error(U, V, W):
        R = len(U[0])
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    err += (s - 1.0) ** 2
        return err
    
    def local_search(U, V, W, R, max_steps=100, deadline=None):
        """Local search on half-integer lattice"""
        best_err = compute_error(U, V, W)
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        
        rng = random.Random(int(time.time() * 1000) % 100000)
        
        for step in range(max_steps):
            if deadline and time.time() > deadline:
                break
            improved = False
            for _ in range(40):
                if deadline and time.time() > deadline:
                    break
                mat = rng.randint(3)
                if mat == 0:
                    row = rng.randint(U_size)
                    col = rng.randint(R)
                    orig = best_U[row][col]
                    for delta in [-0.5, 0.5, -1.0, 1.0, -1.5, 1.5]:
                        new_val = orig + delta
                        if abs(new_val) > 5:
                            continue
                        best_U[row][col] = new_val
                        err = compute_error(best_U, best_V, best_W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                            break
                        else:
                            best_U[row][col] = orig
                elif mat == 1:
                    row = rng.randint(V_size)
                    col = rng.randint(R)
                    orig = best_V[row][col]
                    for delta in [-0.5, 0.5, -1.0, 1.0, -1.5, 1.5]:
                        new_val = orig + delta
                        if abs(new_val) > 5:
                            continue
                        best_V[row][col] = new_val
                        err = compute_error(best_U, best_V, best_W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                            break
                        else:
                            best_V[row][col] = orig
                else:
                    row = rng.randint(W_size)
                    col = rng.randint(R)
                    orig = best_W[row][col]
                    for delta in [-0.5, 0.5, -1.0, 1.0, -1.5, 1.5]:
                        new_val = orig + delta
                        if abs(new_val) > 5:
                            continue
                        best_W[row][col] = new_val
                        err = compute_error(best_U, best_V, best_W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                            break
                        else:
                            best_W[row][col] = orig
            if not improved:
                break
        
        return best_U, best_V, best_W, best_err
    
    def try_rank(R, num_restarts=10):
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            rng = random.Random(restart * 12345 + R)
            U = [[rng.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(R)] for _ in range(U_size)]
            V = [[rng.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(R)] for _ in range(V_size)]
            W = [[rng.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(R)] for _ in range(W_size)]
            
            U, V, W, err = local_search(U, V, W, R, max_steps=80, deadline=deadline)
            
            if err < 1e-8:
                # Snap to half-integers
                U_snap = [[round(x * 2) / 2 for x in row] for row in U]
                V_snap = [[round(x * 2) / 2 for x in row] for row in V]
                W_snap = [[round(x * 2) / 2 for x in row] for row in W]
                
                if verify_float(U_snap, V_snap, W_snap, atol=1e-9):
                    if verify_exact(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
                    else:
                        return U_snap, V_snap, W_snap
        
        return None
    
    # Try rank 15
    result = try_rank(15, num_restarts=8)
    if result:
        return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Try rank 16
    result = try_rank(16, num_restarts=6)
    if result:
        return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Try rank 17
    result = try_rank(17, num_restarts=4)
    if result:
        return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fallback to rank 18
    return {"U": U_school, "V": V_school, "W": W_school}

