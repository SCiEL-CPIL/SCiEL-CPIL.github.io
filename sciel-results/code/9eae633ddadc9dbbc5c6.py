import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 4
    deadline = time.time() + 140.0
    
    # Target tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # Unfoldings
    # T1: (n, m*p)
    T1 = T.reshape(n, m * p)
    # T2: (m, n*p)
    T2 = T.transpose(1, 0, 2).reshape(m, n * p)
    # T3: (p, n*m)
    T3 = T.transpose(2, 0, 1).reshape(p, n * m)
    
    def khatri_rao(A, B):
        # A: (I, R), B: (J, R) -> (I*J, R)
        return np.kron(A, B).reshape(A.shape[0] * B.shape[0], -1).T # Wait, np.kron(A, B) is (I*J, R*1)? No.
        # np.kron(A, B) where A is (I, R) and B is (J, R) is not standard Khatri-Rao.
        # Khatri-Rao is column-wise Kronecker.
        # A_{:,r} \otimes B_{:,r}.
        # Result shape (I*J, R).
        I, R = A.shape
        J, _ = B.shape
        KR = np.zeros((I * J, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # T_calc[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        R = U.shape[1]
        T_calc = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    T_calc[i, j, k] = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
        return np.allclose(T_calc, T, atol=1e-9)

    def to_half_int(X):
        return np.round(X * 2) / 2

    best_U, best_V, best_W = None, None, None
    best_rank = 24 # Schoolbook
    
    # Schoolbook
    U_sb = np.zeros((n * m, 24), dtype=float)
    V_sb = np.zeros((m * p, 24), dtype=float)
    W_sb = np.zeros((p * n, 24), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try Rank 19
    target_rank = 19
    max_iters = 50
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        np.random.seed(restart)
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Update U
            M_VW = khatri_rao(V, W) # (m*p * p*n, R) -> (12*8, 19) = (96, 19)
            # T1 is (n, m*p). Wait.
            # T1[u, v*w] = T[u, v, w].
            # Equation: T1 = U * (V \odot W)^T ?
            # T1_{u, (v,w)} = sum_r U_{u,r} V_{v,r} W_{w,r}.
            # Let M_{(v,w), r} = V_{v,r} W_{w,r}. This is the Khatri-Rao product of V and W?
            # V is (m*p, R). W is (p*n, R).
            # We need to pair V_{j*p+k, r} and W_{k*n+i, r} for fixed i?
            # No, the index of T1 is (u, v, w).
            # u corresponds to i*m+j.
            # v corresponds to j*p+k.
            # w corresponds to k*n+i.
            # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
            # This is NOT a standard CP unfolding because the indices i, j, k are coupled in the flattening.
            # In standard CP, T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
            # Here, U is indexed by (i,j), V by (j,k), W by (k,i).
            # This is a "structured" tensor.
            # We cannot use standard CP-ALS directly on the flattened matrices without respecting the structure.
            
            # We must solve the linear system for the specific structure.
            # For fixed V, W, find U.
            # Minimize sum_{i,j,k} (T_{i,j,k} - sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r})^2.
            # This is linear in U.
            # Let's construct the design matrix for U.
            # Variables: U_{i,j,r} for i in 0..1, j in 0..2, r in 0..18.
            # Total 6 * 19 = 114 variables.
            # Equations: 24.
            
            # We can use scipy.linalg.lstsq.
            # A * x = b
            # x is flattened U.
            
            if time.time() > deadline - 5:
                break

            # Construct A for U
            # A is 24 x (n*m*R)
            A_U = np.zeros((n * m * p, n * m * target_rank))
            # Equation index e = (i, j, k) -> i*m*p + j*p + k
            # Variable index v = (u, r) -> u*R + r, where u = i*m+j
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = i * m * p + j * p + k
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        for r in range(target_rank):
                            var_idx = u * target_rank + r
                            A_U[e, var_idx] = V[v, r] * W[w, r]
                        
            b = T.flatten()
            
            # Solve A_U * x_U = b
            # x_U is (n*m*R, 1)
            try:
                x_U, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                U = x_U.reshape(n * m, target_rank)
            except:
                pass

            # Construct A for V
            A_V = np.zeros((n * m * p, m * p * target_rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = i * m * p + j * p + k
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        for r in range(target_rank):
                            var_idx = v * target_rank + r
                            A_V[e, var_idx] = U[u, r] * W[w, r]
            try:
                x_V, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                V = x_V.reshape(m * p, target_rank)
            except:
                pass

            # Construct A for W
            A_W = np.zeros((n * m * p, p * n * target_rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = i * m * p + j * p + k
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        for r in range(target_rank):
                            var_idx = w * target_rank + r
                            A_W[e, var_idx] = U[u, r] * V[v, r]
            try:
                x_W, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                W = x_W.reshape(p * n, target_rank)
            except:
                pass

            # Check error
            err = np.linalg.norm(T.flatten() - (A_U @ x_U))
            if err < 1e-9:
                break

        # After ALS, check if we can round to half-integers
        U_hi = to_half_int(U)
        V_hi = to_half_int(V)
        W_hi = to_half_int(W)
        
        if verify(U_hi, V_hi, W_hi):
            best_U, best_V, best_W = U_hi, V_hi, W_hi
            best_rank = target_rank
            break

    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
