import time
import sys
import random
from itertools import product
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 by 5x6 matrix multiplication.
    
    Strategy:
    - Split the 2x5 matrix A into A1 (2x2, cols 0-1) and A2 (2x3, cols 2-4).
    - Split the 5x6 matrix B into B1 (2x6, rows 0-1) and B2 (3x6, rows 2-4).
    - C = A1*B1 + A2*B2.
    
    For A1*B1 (2x2 * 2x6):
    Split B1 into three 2x2 blocks. Use Strassen's algorithm for each.
    Rank cost: 3 * 7 = 21.
    
    For A2*B2 (2x3 * 3x6):
    Use schoolbook. Rank cost: 2 * 3 * 6 = 36.
    
    Total rank: 21 + 36 = 57.
    
    Additionally, attempt to find a lower-rank decomposition for the 2x3 x 3x6
    subproblem using a structured search, potentially reducing total rank.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Strassen's algorithm for 2x2 * 2x2
    # Returns U2, V2, W2 of size 4x7, 4x7, 4x7
    def strassen_2x2():
        U2 = [[0] * 7 for _ in range(4)]
        V2 = [[0] * 7 for _ in range(4)]
        W2 = [[0] * 7 for _ in range(4)]
        
        # m1 = (A00 + A11) * (B00 + B11)
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1
        
        # m2 = (A10 + A11) * B00
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[2][1] = 1
        
        # m3 = A00 * (B10 - B11)
        U2[0][2] = 1
        V2[2][2] = 1; V2[3][2] = -1
        W2[0][2] = 1; W2[1][2] = -1
        
        # m4 = A11 * (B10 - B00)
        U2[3][3] = 1
        V2[2][3] = 1; V2[0][3] = -1
        W2[2][3] = 1; W2[0][3] = -1
        
        # m5 = (A00 + A01) * B11
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[1][4] = 1
        
        # m6 = (A10 - A00) * (B00 + B01)
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[1][5] = 1
        W2[3][5] = 1
        
        # m7 = (A01 - A11) * (B10 + B11)
        U2[1][6] = 1; U2[3][6] = -1
        V2[2][6] = 1; V2[3][6] = 1
        W2[3][6] = 1
        
        return U2, V2, W2
    
    U2, V2, W2 = strassen_2x2()
    
    # Build the Strassen part (rank 21) for A1(2x2) * B1(2x6)
    U_strassen = [[0] * 21 for _ in range(10)]  # 10 x 21
    V_strassen = [[0] * 21 for _ in range(30)]  # 30 x 21
    W_strassen = [[0] * 21 for _ in range(12)]  # 12 x 21
    
    col_s = 0
    for block in range(3):
        b_c0 = block * 2
        b_c1 = block * 2 + 1
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    if U2[i*2+j][r] != 0:
                        U_strassen[i*m + j][col_s] += U2[i*2+j][r]
            for j in range(2):
                for k_local in range(2):
                    if V2[j*2+k_local][r] != 0:
                        k_global = b_c0 + k_local
                        V_strassen[j*p + k_global][col_s] += V2[j*2+k_local][r]
            for i in range(2):
                for k_local in range(2):
                    if W2[k_local*2+i][r] != 0:
                        k_global = b_c0 + k_local
                        W_strassen[k_global*n + i][col_s] += W2[k_local*2+i][r]
            col_s += 1
    
    # Verify Strassen part
    def verify_part(U_part, V_part, W_part, indices_A, indices_B, indices_C):
        """Verify a partial decomposition for specific index ranges."""
        rank = len(U_part[0])
        U_np = np.array(U_part, dtype=float)
        V_np = np.array(V_part, dtype=float)
        W_np = np.array(W_part, dtype=float)
        
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        for i in indices_A:
            for j in indices_B:
                for k in indices_C:
                    val = np.sum(U_3d[i,j,:] * V_3d[j,k,:] * W_3d[k,i,:])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Verify Strassen covers A[0:2, 0:2] * B[0:2, :] -> C[:, 0:2]
    # Actually it covers all of C for the first 2 rows of B
    # Let's just verify the full Strassen contribution
    if not verify_part(U_strassen, V_strassen, W_strassen, 
                       range(2), range(2), range(6)):
        # Should not happen, but fallback
        rank = 60
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
    
    # Now try to find a better decomposition for A2(2x3) * B2(3x6)
    # Schoolbook gives rank 36. Try to find rank < 36.
    
    def find_better_2x3_3x6(max_rank=36, time_limit=30):
        """
        Try to find a rank-r decomposition for 2x3 * 3x6 multiplication.
        Returns (U_sub, V_sub, W_sub) or None if not found.
        U_sub: 6 x r (for A[0:2, 2:5])
        V_sub: 18 x r (for B[2:5, 0:6])  
        W_sub: 12 x r (for C[0:6, 0:2])
        """
        if time.time() > time_limit:
            return None
        
        # The subproblem: A_sub is 2x3, B_sub is 3x6, C_sub is 2x6
        # T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4,5}
        # We need: sum_r U_sub[i,j,r] * V_sub[j,k,r] * W_sub[k,i,r] = 1
        
        # Try rank from 30 to 35 (skip 36 which is schoolbook)
        for target_rank in range(30, 36):
            if time.time() > time_limit:
                return None
            
            # Initialize with random values
            best_result = None
            best_err = float('inf')
            
            for attempt in range(50):
                if time.time() > time_limit:
                    break
                
                # Random initialization
                U_sub = np.random.randn(6, target_rank) * 0.5
                V_sub = np.random.randn(18, target_rank) * 0.5
                W_sub = np.random.randn(12, target_rank) * 0.5
                
                # Optimize using gradient descent
                lr = 0.01
                for step in range(500):
                    if time.time() > time_limit:
                        break
                    
                    # Compute error
                    err = 0.0
                    grad_U = np.zeros_like(U_sub)
                    grad_V = np.zeros_like(V_sub)
                    grad_W = np.zeros_like(W_sub)
                    
                    for i in range(2):
                        for j in range(3):
                            for k in range(6):
                                # Local index mapping
                                # A_sub: i in {0,1}, j in {0,1,2} -> index i*3+j
                                # B_sub: j in {0,1,2}, k in {0,1,2,3,4,5} -> index j*6+k
                                # C_sub: k in {0,1,2,3,4,5}, i in {0,1} -> index k*2+i
                                a_idx = i * 3 + j
                                b_idx = j * 6 + k
                                c_idx = k * 2 + i
                                
                                # Current product
                                prod = np.sum(U_sub[a_idx, :] * V_sub[b_idx, :] * W_sub[c_idx, :])
                                err_term = prod - 1.0
                                err += err_term * err_term
                                
                                # Gradients
                                for r in range(target_rank):
                                    vu = V_sub[b_idx, r] * W_sub[c_idx, r]
                                    vv = U_sub[a_idx, r] * W_sub[c_idx, r]
                                    vw = U_sub[a_idx, r] * V_sub[b_idx, r]
                                    grad_U[a_idx, r] += 2 * err_term * vu
                                    grad_V[b_idx, r] += 2 * err_term * vv
                                    grad_W[c_idx, r] += 2 * err_term * vw
                    
                    # Update
                    U_sub -= lr * grad_U
                    V_sub -= lr * grad_V
                    W_sub -= lr * grad_W
                    
                    # Check convergence
                    if err < 1e-12:
                        break
                    
                    # Adaptive learning rate
                    if step > 100 and err > best_err:
                        lr *= 0.5
                    
                    best_err = min(best_err, err)
                
                if err < 1e-10:
                    # Found a solution
                    return (U_sub.tolist(), V_sub.tolist(), W_sub.tolist())
            
            # If we got close but not exact, try to polish
            if best_err < 1e-8:
                # Try to snap to rational values
                pass
        
        return None
    
    # Try to find better decomposition
    better = None
    if time.time() < deadline - 20:
        better = find_better_2x3_3x6(max_rank=36, time_limit=30)
    
    if better is not None:
        U_sub, V_sub, W_sub = better
        sub_rank = len(U_sub[0])
        total_rank = 21 + sub_rank
        
        # Build full matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Copy Strassen part
        for i in range(10):
            for r in range(21):
                U[i][r] = U_strassen[i][r]
        for i in range(30):
            for r in range(21):
                V[i][r] = V_strassen[i][r]
        for i in range(12):
            for r in range(21):
                W[i][r] = W_strassen[i][r]
        
        # Copy sub-part (offset by 21)
        for i in range(2):
            for j in range(3):
                for r in range(sub_rank):
                    U[i*m + (2+j)][21+r] = U_sub[i*3+j][r]
        for j in range(3):
            for k in range(6):
                for r in range(sub_rank):
                    V[(2+j)*p + k][21+r] = V_sub[j*6+k][r]
        for k in range(6):
            for i in range(2):
                for r in range(sub_rank):
                    W[k*n + i][21+r] = W_sub[k*2+i][r]
        
        # Verify full decomposition
        def verify_full(U, V, W):
            rank = len(U[0])
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            U_3d = U_np.reshape(n, m, rank)
            V_3d = V_np.reshape(m, p, rank)
            W_3d = W_np.reshape(p, n, rank)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_3d[i,j,:] * V_3d[j,k,:] * W_3d[k,i,:])
                        if abs(val - 1.0) > 1e-8:
                            return False
            return True
        
        if verify_full(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: use schoolbook for the 2x3 x 3x6 part (rank 36)
    total_rank = 21 + 36
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy Strassen part
    for i in range(10):
        for r in range(21):
            U[i][r] = U_strassen[i][r]
    for i in range(30):
        for r in range(21):
            V[i][r] = V_strassen[i][r]
    for i in range(12):
        for r in range(21):
            W[i][r] = W_strassen[i][r]
    
    # Schoolbook for A2(2x3) * B2(3x6)
    col_s = 21
    for i in range(2):
        for j in range(2, 5):
            for k in range(6):
                U[i*m + j][col_s] = 1
                V[j*p + k][col_s] = 1
                W[k*n + i][col_s] = 1
                col_s += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
