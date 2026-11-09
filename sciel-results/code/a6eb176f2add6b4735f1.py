"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def time_left():
        return deadline - time.time()

    # Schoolbook solution (always valid)
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Fast verification using numpy
    def verify_np(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U_np[i * m + j]
                    vj = V_np[j * p + k]
                    wk = W_np[k * n + i]
                    s = np.dot(ui * vj, wk)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Try Kronecker product structure: 3x4x6 = (1x4x3) ⊗ (3x1x2)
    # 1x4x3 multiplication: rank 12 (schoolbook for 1*4*3)
    # 3x1x2 multiplication: rank 6 (schoolbook for 3*1*2)
    # Total rank: 12 * 6 = 72 (same as schoolbook, not helpful)
    # 
    # Alternative: Try to find a lower rank decomposition using structured search
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # 
    # Let's try a more systematic approach: use the fact that we can exploit
    # the structure of the tensor to find linear dependencies
    
    # Method 1: Try to find a rank-r decomposition by solving a system of equations
    # For each rank r, we need to find U (12 x r), V (24 x r), W (18 x r)
    # such that for all (i,j,k): sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    
    # This is a nonlinear system. Let's use a greedy approach with structured initialization.
    
    # Method 2: Use block structure. Split the middle dimension 4 into 2+2
    # A is 3x4, split into A1 (3x2) and A2 (3x2)
    # B is 4x6, split into B1 (2x6) and B2 (2x6)
    # C = A1*B1 + A2*B2
    # Each 3x2 * 2x6 multiplication has rank at most 3*2*6 = 36
    # But we can do better by sharing structure
    
    # Let's try a direct search for rank 36, 30, 27, 24
    
    def try_structured_search(target_rank, num_trials=50):
        """Try to find a decomposition of given rank using structured random search."""
        if time_left() < 5:
            return None
        
        for trial in range(num_trials):
            if time_left() < 3:
                return None
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Use a structured pattern based on the block decomposition
            # Split middle dimension into two halves of size 2
            mid1 = m // 2  # 2
            mid2 = m - mid1  # 2
            
            for r in range(target_rank):
                # Assign values with some structure
                # Use a pattern that respects the block structure
                block = r % 2  # 0 for first half, 1 for second half
                
                for i in range(n):
                    for j in range(m):
                        # For U, use a pattern based on i, j, and block
                        if block == 0:
                            if j < mid1:
                                U[i * m + j][r] = random.choice([-1, 1])
                            else:
                                U[i * m + j][r] = 0
                        else:
                            if j >= mid1:
                                U[i * m + j][r] = random.choice([-1, 1])
                            else:
                                U[i * m + j][r] = 0
                    
                    for k in range(p):
                        # For V, similar pattern
                        if block == 0:
                            if k < p // 2:
                                V[j * p + k][r] = random.choice([-1, 1])
                            else:
                                V[j * p + k][r] = 0
                        else:
                            if k >= p // 2:
                                V[j * p + k][r] = random.choice([-1, 1])
                            else:
                                V[j * p + k][r] = 0
                    
                    for i2 in range(n):
                        # For W, similar pattern
                        W[k * n + i2][r] = random.choice([-1, 1])
            
            if verify_np(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None

    # Try different target ranks
    for target_rank in [36, 30, 27, 24]:
        if time_left() < 10:
            break
        if target_rank >= best_rank:
            continue
        
        result = try_structured_search(target_rank, num_trials=30)
        if result:
            best_rank = target_rank
            best_result = result
            break

    # If we haven't improved, try a more aggressive random search
    if best_rank == n * m * p and time_left() > 20:
        for target_rank in [48, 42, 36, 30]:
            if time_left() < 8:
                break
            if target_rank >= best_rank:
                continue
            
            found = False
            for _ in range(20):
                if time_left() < 3:
                    break
                
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                
                # Fill with sparse random values
                for r in range(target_rank):
                    for i in range(n * m):
                        if random.random() < 0.5:
                            U[i][r] = random.choice([-1, 1])
                    for i in range(m * p):
                        if random.random() < 0.5:
                            V[i][r] = random.choice([-1, 1])
                    for i in range(p * n):
                        if random.random() < 0.5:
                            W[i][r] = random.choice([-1, 1])
                
                if verify_np(U, V, W):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    found = True
                    break
            
            if found:
                break

    # Final verification
    if not verify_np(best_result["U"], best_result["V"], best_result["W"]):
        best_result = make_schoolbook()

    return best_result
# EVOLVE-BLOCK-END
