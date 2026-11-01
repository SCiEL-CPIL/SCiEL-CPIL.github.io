import time
import numpy as np
import random

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6

    # Strategy: Kronecker product decomposition
    # T_{2,4,6} = T_{1,2,3} x T_{2,2,2}
    # T_{2,2,2} is Strassen (rank 7)
    # T_{1,2,3}: find minimum rank via numerical search
    # Total rank = 7 * rank(T_{1,2,3})

    # Strassen 2x2 @ 2x2 decomposition
    # Local indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    #                B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    #                C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    strassen_U = [
        [1, 0, 0, 1],    # P1: A00 + A11
        [0, 0, 1, 1],    # P2: A10 + A11
        [1, 0, 0, 0],    # P3: A00
        [0, 0, 0, 1],    # P4: A11
        [1, 1, 0, 0],    # P5: A00 + A01
        [-1, 0, 1, 0],   # P6: A10 - A00
        [0, 1, 0, -1],   # P7: A01 - A11
    ]

    strassen_V = [
        [1, 0, 0, 1],    # P1: B00 + B11
        [1, 0, 0, 0],    # P2: B00
        [0, 1, 0, -1],   # P3: B01 - B11
        [0, 0, 1, -1],   # P4: B10 - B00
        [0, 0, 0, 1],    # P5: B11
        [1, 1, 0, 0],    # P6: B00 + B01
        [0, 0, 1, 1],    # P7: B10 + B11
    ]

    strassen_W = [
        [1, 0, 0, 1],    # P1: C00 + C11
        [0, 0, 1, -1],   # P2: C10 - C11
        [0, 1, 0, 1],    # P3: C01 + C11
        [1, 0, 1, 0],    # P4: C00 + C10
        [-1, 1, 0, 0],   # P5: -C00 + C01
        [0, 0, 0, 1],    # P6: C11
        [1, 0, 0, 0],    # P7: C00
    ]

    # Solve T_{1,2,3} decomposition
    # n=1, m=2, p=3
    # U shape (2, R), V shape (6, R), W shape (3, R)
    # Condition: sum_r W[k,r] * U[j,r] * V[j'*3+k, r] = delta_{j,j'} for k in {0,1,2}, j,j' in {0,1}

    def solve_T123(R, max_iters=3000, seed=0):
        np.random.seed(seed)
        U = np.random.randn(2, R) * 0.5
        V = np.random.randn(6, R) * 0.5
        W = np.random.randn(3, R) * 0.5
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.05
        for it in range(max_iters):
            if time.time() - start_time > 90:
                break
            
            # Compute F[k,j,jp] = sum_r W[k,r] * U[j,r] * V[jp*3+k, r]
            F = np.zeros((3, 2, 2))
            for k in range(3):
                for j in range(2):
                    for jp in range(2):
                        F[k, j, jp] = np.dot(W[k], U[j] * V[jp*3+k])
            
            target = np.zeros((3, 2, 2))
            target[:, 0, 0] = 1.0
            target[:, 1, 1] = 1.0
            
            loss = np.sum((F - target)**2)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
            
            # Gradients
            dU = np.zeros((2, R))
            dV = np.zeros((6, R))
            dW = np.zeros((3, R))
            
            for k in range(3):
                for j in range(2):
                    for jp in range(2):
                        err = F[k, j, jp] - target[k, j, jp]
                        dW[k] += 2 * err * U[j] * V[jp*3+k]
                        dU[j] += 2 * err * W[k] * V[jp*3+k]
                        dV[jp*3+k] += 2 * err * W[k] * U[j]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if loss > best_loss * 1.5:
                lr *= 0.5
            elif loss < best_loss * 0.5:
                lr = min(lr * 1.2, 0.5)
        
        return best_loss, best_U, best_V, best_W

    # Try to find T_{1,2,3} with rank 2, then 3
    best_rank_123 = None
    best_sol_123 = None
    
    for R in [2, 3]:
        if time.time() - start_time > 70:
            break
        found = False
        for trial in range(20):
            if time.time() - start_time > 70:
                break
            loss, U123, V123, W123 = solve_T123(R, max_iters=2000, seed=trial * 100 + 42)
            if loss < 1e-8:
                best_rank_123 = R
                best_sol_123 = (U123, V123, W123)
                found = True
                break
        if found:
            break

    if best_sol_123 is not None:
        R123 = best_rank_123
        U123, V123, W123 = best_sol_123
        total_rank = 7 * R123
        
        # Kronecker product construction
        # n=2 = 1*2, m=4 = 2*2, p=6 = 3*2
        # A[i,j]: i = i2 (since i1=0), j = j1*2 + j2, i2 in {0,1}, j1 in {0,1}, j2 in {0,1}
        # B[j,k]: j = j1*2 + j2, k = k1*2 + k2, k1 in {0,1,2}, k2 in {0,1}
        # C[i,k]: i = i2, k = k1*2 + k2
        # W index: k*n + i = (k1*2+k2)*2 + i2
        # U index: i*m + j = i2*4 + j1*2 + j2
        # V index: j*p + k = (j1*2+j2)*6 + k1*2 + k2
        
        U = np.zeros((8, total_rank), dtype=np.float64)
        V = np.zeros((24, total_rank), dtype=np.float64)
        W = np.zeros((12, total_rank), dtype=np.float64)
        
        for r1 in range(R123):
            for r2 in range(7):
                r = r1 * 7 + r2
                for j1 in range(2):
                    for k1 in range(3):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    # U[i2*4 + j1*2 + j2, r] = U123[j1, r1] * strassen_U[r2][i2*2 + j2]
                                    U[i2*4 + j1*2 + j2, r] = U123[j1, r1] * strassen_U[r2][i2*2 + j2]
                                    # V[(j1*2+j2)*6 + k1*2 + k2, r] = V123[j1*3 + k1, r1] * strassen_V[r2][j2*2 + k2]
                                    V[(j1*2 + j2)*6 + k1*2 + k2, r] = V123[j1*3 + k1, r1] * strassen_V[r2][j2*2 + k2]
                                    # W[(k1*2+k2)*2 + i2, r] = W123[k1, r1] * strassen_W[r2][k2*2 + i2]
                                    W[(k1*2 + k2)*2 + i2, r] = W123[k1, r1] * strassen_W[r2][k2*2 + i2]
        
        # Verify
        max_err = 0.0
        for i in range(2):
            for k in range(6):
                for j in range(4):
                    for jp in range(4):
                        val = 0.0
                        for r in range(total_rank):
                            val += W[k*2+i, r] * U[i*4+j, r] * V[jp*6+k, r]
                        expected = 1.0 if j == jp else 0.0
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
        
        if max_err > 1e-5:
            # Verification failed, fall through to fallback
            pass
        else:
            # Try to convert to integers/half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if np.allclose(U, U_rounded, atol=1e-6) and np.allclose(V, V_rounded, atol=1e-6) and np.allclose(W, W_rounded, atol=1e-6):
                U_out = U_rounded.astype(np.float64)
                V_out = V_rounded.astype(np.float64)
                W_out = W_rounded.astype(np.float64)
            else:
                U_out = U
                V_out = V
                W_out = W
            
            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

    # Fallback: block Strassen rank 42
    rank = 42
    U = np.zeros((8, rank), dtype=np.int64)
    V = np.zeros((24, rank), dtype=np.int64)
    W = np.zeros((12, rank), dtype=np.int64)

    col = 0
    for j_block in range(2):  # j1 = 0, 1
        for k_block in range(3):  # k1 = 0, 1, 2
            for prod_idx in range(7):
                r = col + prod_idx
                for local_idx in range(4):
                    i2 = local_idx // 2  # 0 or 1
                    j2 = local_idx % 2   # 0 or 1
                    U[i2*4 + j_block*2 + j2, r] = strassen_U[prod_idx][local_idx]
                    V[(j_block*2 + j2)*6 + k_block*2 + (local_idx % 2), r] = strassen_V[prod_idx][local_idx]
                    W[(k_block*2 + (local_idx % 2))*2 + i2, r] = strassen_W[prod_idx][local_idx]
                col += 7

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
