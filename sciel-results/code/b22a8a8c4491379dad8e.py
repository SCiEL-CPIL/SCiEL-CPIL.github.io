import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Construct the 3x3x1 part (Matrix-Vector multiplication)
    # A (3x3) * b (3x1) -> c (3x1)
    # c_i = sum_j A_ij b_j
    # This is a tensor of size 3x3x1.
    # Rank is 9 (schoolbook).
    # U_mv: (3*3) x 9
    # V_mv: (3*1) x 9
    # W_mv: (1*3) x 9
    
    n1, m1, p1 = 3, 3, 1
    R1 = 9
    U1 = np.zeros((n1 * m1, R1))
    V1 = np.zeros((m1 * p1, R1))
    W1 = np.zeros((p1 * n1, R1))
    col = 0
    for i in range(n1):
        for j in range(m1):
            U1[i * m1 + j, col] = 1
            V1[j * p1, col] = 1 # b_j
            W1[0 * n1 + i, col] = 1 # c_i
            col += 1

    # 2. Search for the 3x3x3 part (Matrix-Matrix multiplication)
    # Target rank 23.
    n2, m2, p2 = 3, 3, 3
    R2_target = 23
    
    # Fallback: Schoolbook 3x3x3 (Rank 27)
    R2_school = 27
    U2_school = np.zeros((n2 * m2, R2_school))
    V2_school = np.zeros((m2 * p2, R2_school))
    W2_school = np.zeros((p2 * n2, R2_school))
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2_school[i * m2 + j, col] = 1
                V2_school[j * p2 + k, col] = 1
                W2_school[k * n2 + i, col] = 1
                col += 1
                
    best_U2, best_V2, best_W2, best_R2 = U2_school, V2_school, W2_school, R2_school

    # Try to find rank 23 decomposition for 3x3x3
    # Using scipy.optimize.least_squares
    try:
        import scipy.optimize
        
        def get_tensor_333(U, V, W):
            T = np.zeros((n2, m2, p2))
            R = U.shape[1]
            for k in range(p2):
                for i in range(n2):
                    for j in range(m2):
                        u_val = U[i * m2 + j, :]
                        v_val = V[j * p2 + k, :]
                        w_val = W[k * n2 + i, :]
                        T[i, j, k] = np.dot(u_val, v_val * w_val)
            return T

        def residuals_333(x):
            U_r = x[:n2*m2*R2_target].reshape(n2*m2, R2_target)
            V_r = x[n2*m2*R2_target : n2*m2*R2_target + m2*p2*R2_target].reshape(m2*p2, R2_target)
            W_r = x[n2*m2*R2_target + m2*p2*R2_target :].reshape(p2*n2, R2_target)
            T_r = get_tensor_333(U_r, V_r, W_r)
            return (T_r - np.ones((n2,m2,p2))).flatten()

        # Try a few restarts
        for _ in range(10):
            if time.time() > deadline - 20:
                break
            U_init = np.random.randn(n2 * m2, R2_target) * 0.1
            V_init = np.random.randn(m2 * p2, R2_target) * 0.1
            W_init = np.random.randn(p2 * n2, R2_target) * 0.1
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = scipy.optimize.least_squares(residuals_333, x0, method='lm', max_nfev=200)
                if result.cost < 1e-8:
                    U_sol = result.x[:n2*m2*R2_target].reshape(n2*m2, R2_target)
                    V_sol = result.x[n2*m2*R2_target : n2*m2*R2_target + m2*p2*R2_target].reshape(m2*p2, R2_target)
                    W_sol = result.x[n2*m2*R2_target + m2*p2*R2_target :].reshape(p2*n2, R2_target)
                    
                    # Snap to half-integers?
                    # Strassen coefficients are integers or half-integers.
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify
                    T_check = get_tensor_333(U_snap, V_snap, W_snap)
                    if np.allclose(T_check, np.ones((n2,m2,p2)), atol=1e-6):
                        best_U2, best_V2, best_W2, best_R2 = U_snap, V_snap, W_snap, R2_target
                        break
            except Exception:
                pass
    except ImportError:
        pass

    # 3. Combine the parts
    # U_total: (3*3) x (R1 + R2)
    # V_total: (3*4) x (R1 + R2)
    #    Note: V for 3x3x3 is (3*3) x R2. V for 3x3x1 is (3*1) x R1.
    #    The full V is (3*4) x (R1+R2).
    #    The first 3 columns of V (for the 3x3x3 part) correspond to B's first 3 columns?
    #    No, the decomposition is on the tensor T.
    #    T[i,j,k] for k=0,1,2 is from part 2.
    #    T[i,j,k] for k=3 is from part 1.
    #    
    #    U is (n*m) x R. Same n,m for both.
    #    U_total = [U2, U1]
    #    
    #    V is (m*p) x R.
    #    Part 2 uses B[:, 0:3] (indices 0..8 in m*p=9).
    #    Part 1 uses B[:, 3] (index 9 in m*p=12? No, m=3, p=4. m*p=12.
    #    Indices for V:
    #    j in 0..2, k in 0..3.
    #    Part 2 (3x3x3): k in 0..2. V2 is (3*3) x R2.
    #    Part 1 (3x3x1): k in 3. V1 is (3*1) x R1.
    #    We need to map V1 to the full V dimensions.
    #    V1[j, col] corresponds to B[j, 3].
    #    In full V, index is j*p + k = j*4 + 3.
    #    
    #    W is (p*n) x R.
    #    Part 2 (3x3x3): k in 0..2. W2 is (3*3) x R2.
    #    Part 1 (3x3x1): k in 3. W1 is (1*3) x R1.
    #    W1[0, i] corresponds to C[i, 3].
    #    In full W, index is k*n + i = 3*3 + i = 9 + i.

    R_total = R1 + best_R2
    U_total = np.hstack([best_U2, U1])
    
    V_total = np.zeros((m * p, R_total))
    # Fill V_total with V2 for k=0,1,2
    for j in range(m):
        for k in range(3):
            V_total[j * p + k, :best_R2] = best_V2[j * 3 + k, :]
    # Fill V_total with V1 for k=3
    for j in range(m):
        V_total[j * p + 3, best_R2:] = U1[j * 1, :] # Wait, V1 is (3*1) x 9. V1[j, col].
        # U1 was (3*3) x 9.
        # V1 was (3*1) x 9.
        # So V1[j, col] is correct.
        
    W_total = np.zeros((p * n, R_total))
    # Fill W_total with W2 for k=0,1,2
    for k in range(3):
        for i in range(n):
            W_total[k * n + i, :best_R2] = best_W2[k * 3 + i, :]
    # Fill W_total with W1 for k=3
    for i in range(n):
        W_total[3 * n + i, best_R2:] = W1[0 * n + i, :] # W1 is (1*3) x 9. W1[0, i].

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
