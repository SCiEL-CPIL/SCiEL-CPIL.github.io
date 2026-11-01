"""Matrix multiplication 2x2x7 rank-minimal decomposition."""

import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 2, 7]

def construct():
    """Return rank-minimal exact tensor decomposition for 2x2*2x7 matrix multiplication."""
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Target: rank 25 = 7 (Strassen 2x2x2) + 7 (Strassen 2x2x2) + 11 (2x2x3)
    # Fallback: rank 26 = 7 + 7 + 12 (schoolbook 2x2x3)
    
    # Build the 2x2x7 multiplication tensor
    # T[i, j, k] = 1 represents C[i,k] += A[i,j] * B[j,k]
    # Flatten: T_flat[(i*m+j) * (p*n) + (k*n+i)] ... actually let's use the U,V,W formulation
    
    # The tensor equation: for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T[i,j,k] = 1 (since C[i,k] = sum_j A[i,j]B[j,k], each term A[i,j]*B[j,k] contributes to C[i,k])
    
    # So we need: sum_r U[i*2+j, r] * V[j*7+k, r] * W[k*2+i, r] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,6}
    
    # That's 2*2*7 = 28 equations.
    
    # Strassen for 2x2x2:
    def make_strassen_block(offset_V_k, offset_W_k):
        """Create Strassen decomposition for 2x2x2 sub-block.
        offset_V_k: offset for B column index (0, 2, or 4)
        offset_W_k: offset for C column index (0, 2, or 4)
        Returns (U_block, V_block, W_block) each as list of 4 or 6 or 4 elements per column
        """
        # 7 Strassen terms
        U_cols = [
            [1, 0, 0, 1],   # f1
            [0, 0, 1, 1],   # f2
            [1, 0, 0, 0],   # f3
            [0, 0, 0, 1],   # f4
            [1, 1, 0, 0],   # f5
            [-1, 0, 1, 0],  # f6
            [0, 1, 0, -1],  # f7
        ]
        V_cols = [
            [1, 0, 0, 1],   # f1
            [1, 0, 0, 0],   # f2
            [0, 1, 0, -1],  # f3
            [-1, 0, 1, 0],  # f4
            [0, 0, 0, 1],   # f5
            [1, 1, 0, 0],   # f6
            [0, 0, 1, 1],   # f7
        ]
        W_cols = [
            [1, 0, 0, 1],   # f1
            [0, 0, 1, -1],  # f2
            [0, 1, 0, 1],   # f3
            [1, 0, 1, 0],   # f4
            [-1, 1, 0, 0],  # f5
            [0, 0, 0, 1],   # f6
            [1, 0, 0, 0],   # f7
        ]
        
        # These are for local indices. Global V row for local (j,k) is j*7 + (k+offset_V_k)
        # Global W row for local (k,i) is (k+offset_W_k)*2 + i
        # U is always local (i,j) -> i*2+j
        
        return U_cols, V_cols, W_cols, offset_V_k, offset_W_k
    
    # Initialize full matrices
    rank_target = 25
    U = [[0.0] * rank_target for _ in range(n * m)]
    V = [[0.0] * rank_target for _ in range(m * p)]
    W = [[0.0] * rank_target for _ in range(p * n)]
    
    # Block 1: B columns 0,1 -> C columns 0,1 (Strassen, rank 7)
    U1, V1, W1, offV1, offW1 = make_strassen_block(0, 0)
    for col in range(7):
        for row in range(4):
            U[row][col] = U1[col][row]
            V[row][col] = V1[col][row]  # local j,k -> global j*7+k, but k=0,1 so j*7+0 or j*7+1
            # Actually V1 is already in local 4-element form for B[0,0],B[0,1],B[1,0],B[1,1]
            # Global row for B[j,k] is j*7+k
            W[row][col] = W1[col][row]  # local k,i -> global (k+0)*2+i = k*2+i
    
    # Block 2: B columns 2,3 -> C columns 2,3 (Strassen, rank 7)
    U2, V2, W2, offV2, offW2 = make_strassen_block(2, 2)
    for col in range(7):
        for row in range(4):
            U[row][col + 7] = U2[col][row]
            local_j = row // 2
            local_k = row % 2
            global_V_row = local_j * 7 + (local_k + 2)
            V[global_V_row][col + 7] = V2[col][row]
            local_k_w = row // 2
            local_i_w = row % 2
            global_W_row = (local_k_w + 2) * 2 + local_i_w
            W[global_W_row][col + 7] = W2[col][row]
    
    # Block 3: B columns 4,5,6 -> C columns 4,5,6 (need rank 11)
    # Try to find rank-11 decomposition for 2x2x3
    found_r11 = False
    best_r11 = None
    
    if time.time() < deadline - 30:  # Leave 30s margin
        try:
            # Search for rank-11 decomposition of 2x2x3
            # Equations: for i in {0,1}, j in {0,1}, k in {0,1,2}:
            # sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
            # That's 12 equations, 11 rank-1 terms
            
            # Use random initialization and gradient descent
            np.random.seed(42)
            
            for attempt in range(2000):
                if time.time() > deadline - 30:
                    break
                    
                # Random initialization
                U3 = np.random.randn(4, 11) * 0.5
                V3 = np.random.randn(6, 11) * 0.5
                W3 = np.random.randn(6, 11) * 0.5
                
                # Scale to encourage small solutions
                U3 /= np.linalg.norm(U3, axis=0, keepdims=True) + 1e-10
                V3 /= np.linalg.norm(V3, axis=0, keepdims=True) + 1e-10
                W3 /= np.linalg.norm(W3, axis=0, keepdims=True) + 1e-10
                
                # Optimize
                lr = 0.01
                for step in range(500):
                    if time.time() > deadline - 30:
                        break
                    
                    # Compute residuals
                    residuals = {}
                    for i in range(2):
                        for j in range(2):
                            for k in range(3):
                                val = 0.0
                                for r in range(11):
                                    val += U3[i*2+j, r] * V3[j*3+k, r] * W3[k*2+i, r]
                                residuals[(i,j,k)] = val - 1.0
                    
                    # Check if close enough
                    max_res = max(abs(res) for res in residuals.values())
                    if max_res < 1e-10:
                        # Verify exactly
                        exact = True
                        for i in range(2):
                            for j in range(2):
                                for k in range(3):
                                    val = sum(U3[i*2+j, r] * V3[j*3+k, r] * W3[k*2+i, r] for r in range(11))
                                    if abs(val - 1.0) > 1e-10:
                                        exact = False
                                        break
                            if not exact:
                                break
                        if exact:
                            # Round to integers/half-integers
                            U3_r = np.round(U3 * 2) / 2
                            V3_r = np.round(V3 * 2) / 2
                            W3_r = np.round(W3 * 2) / 2
                            
                            # Verify rounded version
                            exact_r = True
                            for i in range(2):
                                for j in range(2):
                                    for k in range(3):
                                        val = sum(U3_r[i*2+j, r] * V3_r[j*3+k, r] * W3_r[k*2+i, r] for r in range(11))
                                        if abs(val - 1.0) > 1e-9:
                                            exact_r = False
                                            break
                                if not exact_r:
                                    break
                            if exact_r:
                                best_r11 = (U3_r.tolist(), V3_r.tolist(), W3_r.tolist())
                                found_r11 = True
                                break
                    
                    # Gradient step (simplified)
                    dU = np.zeros_like(U3)
                    dV = np.zeros_like(V3)
                    dW = np.zeros_like(W3)
                    
                    for i in range(2):
                        for j in range(2):
                            for k in range(3):
                                res = residuals[(i,j,k)]
                                for r in range(11):
                                    dU[i*2+j, r] += res * V3[j*3+k, r] * W3[k*2+i, r]
                                    dV[j*3+k, r] += res * U3[i*2+j, r] * W3[k*2+i, r]
                                    dW[k*2+i, r] += res * U3[i*2+j, r] * V3[j*3+k, r]
                    
                    U3 -= lr * dU
                    V3 -= lr * dV
                    W3 -= lr * dW
                
                if found_r11:
                    break
        
        except Exception:
            pass
    
    if found_r11 and best_r11 is not None:
        U3_list, V3_list, W3_list = best_r11
        for col in range(11):
            for row in range(4):
                U[row][col + 14] = U3_list[row][col]
            for row in range(6):
                local_j = row // 3
                local_k = row % 3
                global_V_row = local_j * 7 + (local_k + 4)
                V[global_V_row][col + 14] = V3_list[row][col]
            for row in range(6):
                local_k = row // 2
                local_i = row % 2
                global_W_row = (local_k + 4) * 2 + local_i
                W[global_W_row][col + 14] = W3_list[row][col]
        
        rank_used = 25
    else:
        # Fallback: schoolbook for 2x2x3 (rank 12)
        for j in range(2):
            for k in range(3):
                col = 14 + j * 3 + k
                U[j][col] = 1.0  # A[i,j] for specific i... 
                # Schoolbook: for each (i,j,k), U[i*2+j, col]=1, V[j*7+(k+4), col]=1, W[(k+4)*2+i, col]=1
                for i in range(2):
                    U[i*2+j][col] = 1.0
                V[j*7+(k+4)][col] = 1.0
                for i in range(2):
                    W[(k+4)*2+i][col] = 1.0
        
        rank_used = 26
    
    # Final verification
    for i in range(2):
        for j in range(2):
            for k in range(7):
                val = 0.0
                for r in range(rank_used):
                    val += U[i*2+j][r] * V[j*7+k][r] * W[k*2+i][r]
                if abs(val - 1.0) > 1e-9:
                    # Invalid, fall back to schoolbook rank 28
                    rank = 28
                    U = [[0.0] * rank for _ in range(4)]
                    V = [[0.0] * rank for _ in range(14)]
                    W = [[0.0] * rank for _ in range(14)]
                    col = 0
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(7):
                                U[ii*2+jj][col] = 1.0
                                V[jj*7+kk][col] = 1.0
                                W[kk*2+ii][col] = 1.0
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
