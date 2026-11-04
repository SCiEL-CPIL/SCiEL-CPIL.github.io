"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    def verify(U, V, W, tol=1e-10):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Schoolbook decomposition (Rank 24) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, idx] = 1
                V_school[j * p + k, idx] = 1
                W_school[k * n + i, idx] = 1
                idx += 1

    # Rank-22 decomposition:
    # A is 2x3, B is 3x4
    # Split A = [A_left (2x2) | A_right (2x1)]
    # Split B = [B_top (2x4); B_bottom (1x4)]
    # C = A_left * B_top + A_right * B_bottom
    # A_left * B_top: 2x2 * 2x4, decompose as two 2x2*2x2 Strassen blocks (rank 7+7=14)
    # A_right * B_bottom: 2x1 * 1x4 (rank 8)
    # Total: 22

    rank_22 = 22
    U22 = np.zeros((n * m, rank_22))
    V22 = np.zeros((m * p, rank_22))
    W22 = np.zeros((p * n, rank_22))

    def add_u(i, j, c, val):
        U22[i * m + j, c] += val
    def add_v(j, k, c, val):
        V22[j * p + k, c] += val
    def add_w(k, i, c, val):
        W22[k * n + i, c] += val

    def add_strassen_block(ai, aj, bi, kj, offset):
        """
        Add Strassen decomposition for:
        A[ai:ai+2, aj:aj+2] * B[bi:bi+2, kj:kj+2] -> C[ai:ai+2, kj:kj+2]
        
        A elements:
          a00 = A[ai, aj],     a01 = A[ai, aj+1]
          a10 = A[ai+1, aj],   a11 = A[ai+1, aj+1]
        
        B elements:
          b00 = B[bi, kj],     b01 = B[bi, kj+1]
          b10 = B[bi+1, kj],   b11 = B[bi+1, kj+1]
        
        C elements (W index = k*n + i):
          c00 = C[ai, kj]     -> W[kj*n + ai]
          c01 = C[ai, kj+1]   -> W[(kj+1)*n + ai]
          c10 = C[ai+1, kj]   -> W[kj*n + ai+1]
          c11 = C[ai+1, kj+1] -> W[(kj+1)*n + ai+1]
        
        Strassen:
          m1 = (a00+a11)(b00+b11)
          m2 = (a10+a11)(b00)
          m3 = a00(b01-b11)
          m4 = a11(b10-b00)
          m5 = (a00+a01)(b11)
          m6 = (a10-a00)(b00+b10)
          m7 = (a01-a11)(b10+b11)
          
          C00 = m1 - m5 + m7
          C01 = m3 + m5
          C10 = m2 + m4
          C11 = m1 - m2 + m3 + m6
        """
        # m1 = (a00+a11)(b00+b11) -> C00 +1, C11 +1
        add_u(ai, aj, offset + 0, 1)
        add_u(ai+1, aj+1, offset + 0, 1)
        add_v(bi, kj, offset + 0, 1)
        add_v(bi+1, kj+1, offset + 0, 1)
        add_w(kj, ai, offset + 0, 1)
        add_w(kj+1, ai+1, offset + 0, 1)
        
        # m2 = (a10+a11)(b00) -> C10 +1, C11 -1
        add_u(ai+1, aj, offset + 1, 1)
        add_u(ai+1, aj+1, offset + 1, 1)
        add_v(bi, kj, offset + 1, 1)
        add_w(kj, ai+1, offset + 1, 1)
        add_w(kj+1, ai+1, offset + 1, -1)
        
        # m3 = a00(b01-b11) -> C01 +1, C11 +1
        add_u(ai, aj, offset + 2, 1)
        add_v(bi, kj+1, offset + 2, 1)
        add_v(bi+1, kj+1, offset + 2, -1)
        add_w(kj+1, ai, offset + 2, 1)
        add_w(kj+1, ai+1, offset + 2, 1)
        
        # m4 = a11(b10-b00) -> C10 +1
        add_u(ai+1, aj+1, offset + 3, 1)
        add_v(bi+1, kj, offset + 3, 1)
        add_v(bi, kj, offset + 3, -1)
        add_w(kj, ai+1, offset + 3, 1)
        
        # m5 = (a00+a01)(b11) -> C00 -1, C01 +1
        add_u(ai, aj, offset + 4, 1)
        add_u(ai, aj+1, offset + 4, 1)
        add_v(bi+1, kj+1, offset + 4, 1)
        add_w(kj, ai, offset + 4, -1)
        add_w(kj+1, ai, offset + 4, 1)
        
        # m6 = (a10-a00)(b00+b10) -> C11 +1
        add_u(ai+1, aj, offset + 5, 1)
        add_u(ai, aj, offset + 5, -1)
        add_v(bi, kj, offset + 5, 1)
        add_v(bi+1, kj, offset + 5, 1)
        add_w(kj+1, ai+1, offset + 5, 1)
        
        # m7 = (a01-a11)(b10+b11) -> C00 +1
        add_u(ai, aj+1, offset + 6, 1)
        add_u(ai+1, aj+1, offset + 6, -1)
        add_v(bi+1, kj, offset + 6, 1)
        add_v(bi+1, kj+1, offset + 6, 1)
        add_w(kj, ai, offset + 6, 1)

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # ai=0, aj=0, bi=0, kj=0
    add_strassen_block(0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # ai=0, aj=0, bi=0, kj=2
    add_strassen_block(0, 0, 0, 2, 7)
    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # This is 2x1 * 1x4, rank 8
    offset = 14
    for i in range(2):
        for k in range(4):
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1

    # Verify the rank-22 decomposition
    if verify(U22, V22, W22):
        return {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}

    # If rank-22 failed, try to find a lower rank via optimization
    if time.time() - t0 > 130:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Try rank-21 via gradient descent with multiple restarts
    def try_optimize_rank(target_rank, n_restarts=200, steps_per_restart=300):
        """Try to find a decomposition of given rank using gradient descent"""
        for restart in range(n_restarts):
            if time.time() - t0 > 140:
                break
            
            np.random.seed(restart * 7 + 13)
            U_r = np.random.randn(n * m, target_rank) * 0.5
            V_r = np.random.randn(m * p, target_rank) * 0.5
            W_r = np.random.randn(p * n, target_rank) * 0.5
            
            lr = 0.05
            best_loss = float('inf')
            
            for step in range(steps_per_restart):
                if time.time() - t0 > 140:
                    break
                
                loss = 0.0
                grad_U = np.zeros_like(U_r)
                grad_V = np.zeros_like(V_r)
                grad_W = np.zeros_like(W_r)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U_r[i*m+j, :]
                            v_val = V_r[j*p+k, :]
                            w_val = W_r[k*n+i, :]
                            pred = np.dot(u_val, v_val * w_val)
                            err = pred - 1.0
                            loss += err * err
                            
                            grad_U[i*m+j, :] += 2 * err * v_val * w_val
                            grad_V[j*p+k, :] += 2 * err * u_val * w_val
                            grad_W[k*n+i, :] += 2 * err * u_val * v_val
                
                U_r -= lr * grad_U
                V_r -= lr * grad_V
                W_r -= lr * grad_W
                
                if loss < 1e-12:
                    if verify(U_r, V_r, W_r, tol=1e-8):
                        return U_r, V_r, W_r
                    break
                
                if loss < best_loss:
                    best_loss = loss
            
            # Try a few more steps with smaller learning rate if close
            if best_loss < 1e-6:
                for step in range(200):
                    if time.time() - t0 > 140:
                        break
                    loss = 0.0
                    grad_U = np.zeros_like(U_r)
                    grad_V = np.zeros_like(V_r)
                    grad_W = np.zeros_like(W_r)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_val = U_r[i*m+j, :]
                                v_val = V_r[j*p+k, :]
                                w_val = W_r[k*n+i, :]
                                pred = np.dot(u_val, v_val * w_val)
                                err = pred - 1.0
                                loss += err * err
                                
                                grad_U[i*m+j, :] += 2 * err * v_val * w_val
                                grad_V[j*p+k, :] += 2 * err * u_val * w_val
                                grad_W[k*n+i, :] += 2 * err * u_val * v_val
                    
                    U_r -= 0.01 * grad_U
                    V_r -= 0.01 * grad_V
                    W_r -= 0.01 * grad_W
                    
                    if loss < 1e-12:
                        if verify(U_r, V_r, W_r, tol=1e-8):
                            return U_r, V_r, W_r
                        break
        
        return None, None, None

    # Try rank 21
    if time.time() - t0 < 135:
        U21, V21, W21 = try_optimize_rank(21, n_restarts=100, steps_per_restart=200)
        if U21 is not None and verify(U21, V21, W21):
            return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}

    # Return schoolbook as final fallback
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
