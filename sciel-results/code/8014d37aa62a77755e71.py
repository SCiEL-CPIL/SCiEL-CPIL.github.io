import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0
    
    def build_2x2_block_strassen():
        """3x3 blocks of 2x2, rank = 27*7 = 189"""
        rank = 189
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        idx = 0
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for s in range(7):
                        a00 = (2*i) * 6 + 2*k
                        a01 = (2*i) * 6 + 2*k + 1
                        a10 = (2*i+1) * 6 + 2*k
                        a11 = (2*i+1) * 6 + 2*k + 1
                        
                        b00 = (2*k) * 6 + 2*j
                        b01 = (2*k) * 6 + 2*j + 1
                        b10 = (2*k+1) * 6 + 2*j
                        b11 = (2*k+1) * 6 + 2*j + 1
                        
                        c00 = (2*j) * 6 + 2*i
                        c01 = (2*j+1) * 6 + 2*i
                        c10 = (2*j) * 6 + (2*i+1)
                        c11 = (2*j+1) * 6 + (2*i+1)
                        
                        if s == 0:
                            U[a00, idx] = 1; U[a11, idx] = 1
                            V[b00, idx] = 1; V[b11, idx] = 1
                            W[c00, idx] = 1; W[c11, idx] = 1
                        elif s == 1:
                            U[a10, idx] = 1; U[a11, idx] = 1
                            V[b00, idx] = 1
                            W[c10, idx] = 1; W[c11, idx] = -1
                        elif s == 2:
                            U[a00, idx] = 1
                            V[b01, idx] = 1; V[b11, idx] = -1
                            W[c01, idx] = 1; W[c11, idx] = 1
                        elif s == 3:
                            U[a11, idx] = 1
                            V[b10, idx] = 1; V[b00, idx] = -1
                            W[c00, idx] = 1; W[c10, idx] = 1
                        elif s == 4:
                            U[a00, idx] = 1; U[a01, idx] = 1
                            V[b11, idx] = 1
                            W[c00, idx] = -1; W[c01, idx] = 1
                        elif s == 5:
                            U[a10, idx] = 1; U[a00, idx] = -1
                            V[b00, idx] = 1; V[b01, idx] = 1
                            W[c11, idx] = 1
                        elif s == 6:
                            U[a01, idx] = 1; U[a11, idx] = -1
                            V[b10, idx] = 1; V[b11, idx] = 1
                            W[c00, idx] = 1
                        idx += 1
        return U, V, W, rank
    
    # Primary: 189-rank Strassen block decomposition
    U, V, W, rank = build_2x2_block_strassen()
    
    # Verify the decomposition
    if time.time() < deadline - 10:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Fast verification using matrix operations
        # C[i,j] = sum_t W[j*n+i, t] * (sum_k U[i*m+k, t] * V[k*p+j, t])
        # = sum_t W[j*n+i, t] * (U[:,t] * V[:,t])
        # where U[:,t] is a vector in R^(n*m) reshaped to (n,m)
        # and V[:,t] is a vector in R^(m*p) reshaped to (m,p)
        
        C_computed = np.zeros((n, p))
        for t in range(rank):
            u_t = U_arr[:, t].reshape(n, m)
            v_t = V_arr[:, t].reshape(m, p)
            # The product u_t * v_t is an outer-product-like contribution
            # Actually, the bilinear form is: sum_k U[i*m+k, t] * V[k*p+j, t]
            # This is NOT u_t @ v_t. Let's compute it properly.
            # For each t, the contribution to C[i,j] is:
            # W[j*n+i, t] * sum_k U[i*m+k, t] * V[k*p+j, t]
            
            # We can compute the "partial product" for term t:
            # partial_t[i,j] = sum_k U[i*m+k, t] * V[k*p+j, t]
            # This is equivalent to: 
            # Let U_t be n x m matrix with U_t[i,k] = U[i*m+k, t]
            # Let V_t be m x p matrix with V_t[k,j] = V[k*p+j, t]
            # Then partial_t = ??? 
            # sum_k U_t[i,k] * V_t[k,j] is just (U_t @ V_t)[i,j]
            
            u_mat = U_arr[:, t].reshape(n, m)
            v_mat = V_arr[:, t].reshape(m, p)
            partial = u_mat @ v_mat
            C_computed += W_arr[:, t] * partial.reshape(p * n, order='F')[:0] + (W_arr[:, t].reshape(p, n, order='F') * partial)
        
        # Wait, W is indexed as W[j*n + i, t], so W[:, t] reshaped to (p, n) with F-order gives W[j*n+i, t]
        # Let's redo this more carefully.
        
        C_computed = np.zeros((n, p))
        for t in range(rank):
            u_mat = U_arr[:, t].reshape(n, m)
            v_mat = V_arr[:, t].reshape(m, p)
            partial = u_mat @ v_mat  # (n, p)
            w_mat = W_arr[:, t].reshape(p, n, order='F')  # w_mat[j, i] = W[j*n+i, t]
            C_computed += w_mat.T * partial  # (n, p) element-wise
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=int)
            V_s = np.zeros((m * p, rank_school), dtype=int)
            W_s = np.zeros((p * n, rank_school), dtype=int)
            idx_s = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_s[i * m + k, idx_s] = 1
                        V_s[k * p + j, idx_s] = 1
                        W_s[j * n + i, idx_s] = 1
                        idx_s += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
