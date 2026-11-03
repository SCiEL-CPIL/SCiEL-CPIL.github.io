"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    Strategy: Decompose into two 3x3x3 blocks. Find a low-rank decomposition for 3x3x3.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    def validate_3x3x3(U, V, W):
        """Check if U, V, W (3 x r) form a valid rank-r decomposition of 3x3x3."""
        r = len(U[0])
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for c in range(r):
                        s += U[i][c] * V[j][c] * W[k][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def find_3x3x3_decomposition(target_rank, max_attempts=500, timeout=20):
        """
        Try to find a rank-target_rank decomposition for 3x3x3 matrix multiplication.
        Uses random integer initialization and gradient descent polishing.
        """
        start_time = time.time()
        
        # Target tensor T_{i,j,k} = 1
        # We want sum_c U[i,c] V[j,c] W[k,c] = 1
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            
            # Initialize with random integers -1, 0, 1
            U = np.random.randint(-1, 2, (3, target_rank))
            V = np.random.randint(-1, 2, (3, target_rank))
            W = np.random.randint(-1, 2, (3, target_rank))
            
            # Gradient descent to minimize error
            lr = 0.05
            for it in range(200):
                if time.time() - start_time > timeout:
                    break
                
                # Compute error
                error = 0
                grad_U = np.zeros_like(U, dtype=float)
                grad_V = np.zeros_like(V, dtype=float)
                grad_W = np.zeros_like(W, dtype=float)
                
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = np.sum(U[i, :] * V[j, :] * W[k, :])
                            diff = s - 1.0
                            error += diff * diff
                            
                            # Gradients
                            # d/dU[i,c] = 2 * diff * V[j,c] * W[k,c]
                            term = 2.0 * diff * V[j, :] * W[k, :]
                            grad_U[i, :] += term
                            grad_V[j, :] += 2.0 * diff * U[i, :] * W[k, :]
                            grad_W[k, :] += 2.0 * diff * U[i, :] * V[j, :]
                
                if error < 1e-6:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if validate_3x3x3(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return U_r, V_r, W_r
                    
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                lr *= 0.99
                
        return None

    def schoolbook_3x3x6():
        """Return the schoolbook rank 54 decomposition."""
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

    def assemble_3x3x6_from_3x3x3(U3, V3, W3):
        """
        Assemble 3x3x6 decomposition from two copies of 3x3x3 decomposition.
        U3, V3, W3 are (3, r) matrices.
        """
        r3 = len(U3[0])
        total_rank = 2 * r3
        
        # U for 3x3x6 is (3*3, 2*r3) = (9, 2*r3)
        # First r3 columns correspond to k=0,1,2 (first half of output)
        # Second r3 columns correspond to k=3,4,5 (second half of output)
        # Wait, the index k in 3x3x6 goes from 0 to 5.
        # The tensor is T_{i,j,k}.
        # For k in 0..2, it's one 3x3x3.
        # For k in 3..5, it's another 3x3x3.
        
        U = [[0] * total_rank for _ in range(9)]
        V = [[0] * total_rank for _ in range(18)] # m*p = 3*6 = 18
        W = [[0] * total_rank for _ in range(18)] # p*n = 6*3 = 18
        
        # Fill first half (k=0,1,2)
        for i in range(3):
            for j in range(3):
                for c in range(r3):
                    U[i * 3 + j][c] = U3[i][c]
                    # V index: j * p + k. Here k is 0,1,2.
                    # But V3 is for 3x3x3, so V3[j][c] corresponds to j in 0..2.
                    # In the 3x3x6 V matrix, the row index is j * 6 + k.
                    # For the first block (k=0,1,2), we need to map V3[j][c] to V[j*6+k][c]?
                    # No, V3[j][c] is the coefficient for B_{j,k}.
                    # In the decomposition, V_{j,k,c} is the coefficient.
                    # For the first 3x3x3 block, k_local = 0,1,2.
                    # So V_{j, k_local, c} = V3[j][c].
                    # In the 3x3x6 V matrix, the index is (j * 6 + k_local).
                    # So we fill V[j * 6 + k_local][c] = V3[j][c].
                    
                    # Similarly for W. W_{k, i, c} = W3[k][c].
                    # In 3x3x6 W matrix, index is (k * 3 + i).
                    # So W[k_local * 3 + i][c] = W3[k_local][c].
                    
        # This logic is getting complex to assemble manually. 
        # Let's use a simpler mapping.
        
        # Reset U, V, W
        U = np.zeros((9, total_rank))
        V = np.zeros((18, total_rank))
        W = np.zeros((18, total_rank))
        
        # First block: k in 0..2
        for i in range(3):
            for j in range(3):
                for c in range(r3):
                    U[i * 3 + j, c] = U3[i][c]
                    # V row index: j * 6 + k. k is 0,1,2.
                    # But V3[j][c] is the same for all k? No.
                    # In 3x3x3, V is (3, r). V[j][c] is the coefficient for B_{j,k}.
                    # Wait, the tensor is T_{i,j,k}.
                    # The decomposition is T = sum U_i V_j W_k.
                    # For 3x3x3, V has 3 rows (j=0..2).
                    # For 3x3x6, V has 18 rows (j=0..2, k=0..5).
                    # The entry V_{j,k,c} in 3x3x6 corresponds to the coefficient of B_{j,k}.
                    # In the first block (k=0..2), the coefficient is V3[j][c].
                    # So V[j * 6 + k][c] = V3[j][c] for k=0,1,2.
                    # This means V[0..5, c] are filled with V3[0][c], V3[1][c], V3[2][c] repeated?
                    # No.
                    # For fixed j, V3[j][c] is the coefficient for B_{j, k_local}.
                    # So for k_local=0, V[j*6+0][c] = V3[j][c].
                    # For k_local=1, V[j*6+1][c] = V3[j][c].
                    # For k_local=2, V[j*6+2][c] = V3[j][c].
                    # This is correct because the tensor T_{i,j,k} is 1 for all k.
                    # So the coefficient for B_{j,k} is the same for all k in the block?
                    # No, the tensor is T_{i,j,k} = 1.
                    # The decomposition is sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1.
                    # If we use two independent 3x3x3 decompositions:
                    # For k=0,1,2: sum_c U1_{i,j,c} V1_{j,k,c} W1_{k,i,c} = 1.
                    # Here V1_{j,k,c} depends on k.
                    # In the 3x3x3 problem, V is (3, r). The index is j.
                    # Wait, in 3x3x3, the second dimension of V is j (0..2).
                    # The third dimension of W is i (0..2).
                    # The first dimension of U is i (0..2)? No, U is (n, r) = (3, r).
                    # The tensor is T_{i,j,k}.
                    # U_{i,c}, V_{j,c}, W_{k,c}.
                    # So V_{j,c} is the coefficient for the j-th row of B?
                    # In matrix multiplication C = AB, C_{i,k} = sum_j A_{i,j} B_{j,k}.
                    # The tensor T_{i,j,k} multiplies A_{i,j} and B_{j,k}.
                    # So the decomposition is sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} ?
                    # No, the standard tensor decomposition for matrix multiplication is:
                    # T_{i,j,k} = sum_c U_{i,c} V_{j,c} W_{k,c} ?
                    # This would imply U depends on i, V on j, W on k.
                    # But the task defines U as (n*m, rank), V as (m*p, rank), W as (p*n, rank).
                    # So U_{i*m+j, c}, V_{j*p+k, c}, W_{k*n+i, c}.
                    # This matches the schoolbook implementation in the prompt.
                    
                    # So for the first block (k=0,1,2):
                    # We need U_{i*3+j, c}, V_{j*6+k, c}, W_{k*3+i, c}.
                    # If we use a 3x3x3 decomposition with vectors u_i, v_j, w_k:
                    # We map u_i -> U_{i*3+j, c}? No, U depends on i and j.
                    # In the 3x3x3 problem, U is (3, r). It depends on i?
                    # In the 3x3x3 problem, the tensor is T_{i,j,k}.
                    # The factors are U_{i,c}, V_{j,c}, W_{k,c}.
                    # So U depends only on i?
                    # But in the 3x3x6 problem, U depends on i and j.
                    # This implies the 3x3x3 decomposition I'm thinking of (U_i, V_j, W_k) is NOT directly compatible with the U_{i,j}, V_{j,k}, W_{k,i} structure of the 3x3x6 problem unless we expand.
                    
                    # In 3x3x3:
                    # T_{i,j,k} = sum_c u_i^{(c)} v_j^{(c)} w_k^{(c)} = 1.
                    # In 3x3x6:
                    # T_{i,j,k} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)} = 1.
                    
                    # If we have a 3x3x3 solution (u, v, w), can we build a 3x3x6 solution?
                    # For k=0,1,2:
                    # We want sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)} = 1.
                    # If we set U_{i,j}^{(c)} = u_i^{(c)} (independent of j?),
                    # V_{j,k}^{(c)} = v_j^{(c)} (independent of k?),
                    # W_{k,i}^{(c)} = w_k^{(c)} (independent of i?),
                    # Then sum_c u_i v_j w_k = 1. This works for the first block.
                    # But we need to define U, V, W for the second block (k=3,4,5) as well.
                    # We can use the same u, v, w for the second block, but with different indices in U, V, W.
                    
                    # So:
                    # For c in 0..r3-1:
                    #   For i in 0..2, j in 0..2:
                    #     U[i*3+j, c] = u_i[c]
                    #     U[i*3+j, r3+c] = u_i[c]  # for second block
                    #   
                    #   For j in 0..2, k in 0..2:
                    #     V[j*6+k, c] = v_j[c]
                    #     V[j*6+(3+k), r3+c] = v_j[c] # for second block
                    #     
                    #   For k in 0..2, i in 0..2:
                    #     W[k*3+i, c] = w_k[c]
                    #     W[(3+k)*3+i, r3+c] = w_k[c] # for second block
                    
        # Let's implement this assembly.
        U = np.zeros((9, total_rank))
        V = np.zeros((18, total_rank))
        W = np.zeros((18, total_rank))
        
        for c in range(r3):
            for i in range(3):
                for j in range(3):
                    U[i * 3 + j, c] = U3[i][c]
                    U[i * 3 + j, r3 + c] = U3[i][c]
            
            for j in range(3):
                for k in range(3):
                    V[j * 6 + k, c] = V3[j][c]
                    V[j * 6 + 3 + k, r3 + c] = V3[j][c]
            
            for k in range(3):
                for i in range(3):
                    W[k * 3 + i, c] = W3[k][c]
                    W[(3 + k) * 3 + i, r3 + c] = W3[k][c]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find 3x3x3 decomposition
    best_3x3x3 = None
    for r in range(23, 28):
        result = find_3x3x3_decomposition(r, max_attempts=100, timeout=10)
        if result is not None:
            best_3x3x3 = result
            break
            
    if best_3x3x3 is not None:
        return assemble_3x3x6_from_3x3x3(*best_3x3x3)
        
    return schoolbook_3x3x6()
# EVOLVE-BLOCK-END
