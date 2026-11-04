"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    def verify_numpy(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Verify: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_mat[i*m+j, :]
                for k in range(p):
                    v_row = V_mat[j*p+k, :]
                    w_row = W_mat[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                    if err > tol:
                        return False, max_err
        return True, max_err

    def clean(mat):
        """Snap values to nearest integer or half-integer."""
        out = np.zeros_like(mat)
        for idx in np.ndindex(mat.shape):
            v = mat[idx]
            if abs(v) < 1e-10:
                out[idx] = 0.0
            elif abs(v - round(v)) < 1e-9:
                out[idx] = float(round(v))
            elif abs(v - round(v*2)/2) < 1e-9:
                out[idx] = round(v*2)/2
            else:
                out[idx] = float(v)
        return out

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Construction A: Group over middle index j.
    # For each j, the j-slice is an n x p all-ones matrix.
    # Factor J_{n,p} = A_j * B_j^T where A_j is n x r, B_j is p x r.
    # Simplest: r = 1, A_j = ones(n,1), B_j = ones(p,1), scale by 1/1.
    # But that gives rank = m * 1 = m. Let's check:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # Then sum_r U*V*W = 1*1*1 = 1. Yes!
    # Wait, but we need distinct r for each j. So rank = m.
    
    # Actually, let me reconsider. The tensor is T[i,j,k] = 1 for all i,j,k.
    # We want T = sum_r U_r ⊗ V_r ⊗ W_r where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i*m+j]*V_r[j*p+k]*W_r[k*n+i]
    
    # Construction 1: One term per j.
    # For term r corresponding to j: U_r[i*m+j'] = 1 if j'=j else 0; V_r[j'*p+k] = 1 for all k if j'=j else 0; W_r[k*n+i] = 1 for all i,k.
    # Then T[i,j,k] = sum_r U_r[i*m+j]*V_r[j*p+k]*W_r[k*n+i] = U_j[i*m+j]*V_j[j*p+k]*W_j[k*n+i] = 1*1*1 = 1. ✓
    # Rank = m = 5.
    
    rank1 = m
    U1 = np.zeros((n*m, rank1))
    V1 = np.zeros((m*p, rank1))
    W1 = np.zeros((p*n, rank1))
    for j in range(m):
        for i in range(n):
            U1[i*m+j, j] = 1.0
        for k in range(p):
            V1[j*p+k, j] = 1.0
        for k in range(p):
            for i in range(n):
                W1[k*n+i, j] = 1.0
    
    ok1, _ = verify_numpy(U1, V1, W1, n, m, p)
    best_U, best_V, best_W, best_rank = None, None, None, None
    if ok1:
        best_U, best_V, best_W, best_rank = U1.copy(), V1.copy(), W1.copy(), rank1

    # Construction 2: One term per i.
    # For term r corresponding to i: U_r[i'*m+j] = 1 if i'=i else 0; V_r[j*p+k] = 1 for all j,k; W_r[k*n+i'] = 1 if i'=i else 0.
    # Then T[i,j,k] = U_i[i*m+j]*V_i[j*p+k]*W_i[k*n+i] = 1*1*1 = 1. ✓
    # Rank = n = 4.
    
    rank2 = n
    U2 = np.zeros((n*m, rank2))
    V2 = np.zeros((m*p, rank2))
    W2 = np.zeros((p*n, rank2))
    for i in range(n):
        for j in range(m):
            U2[i*m+j, i] = 1.0
        for j in range(m):
            for k in range(p):
                V2[j*p+k, i] = 1.0
        for k in range(p):
            W2[k*n+i, i] = 1.0
    
    ok2, _ = verify_numpy(U2, V2, W2, n, m, p)
    if ok2 and (best_rank is None or rank2 < best_rank):
        best_U, best_V, best_W, best_rank = U2.copy(), V2.copy(), W2.copy(), rank2

    # Construction 3: One term per k.
    # For term r corresponding to k: U_r[i*m+j] = 1 for all i,j; V_r[j'*p+k'] = 1 if k'=k else 0; W_r[k'*n+i] = 1 if k'=k else 0.
    # Then T[i,j,k] = U_k[i*m+j]*V_k[j*p+k]*W_k[k*n+i] = 1*1*1 = 1. ✓
    # Rank = p = 6.
    
    rank3 = p
    U3 = np.zeros((n*m, rank3))
    V3 = np.zeros((m*p, rank3))
    W3 = np.zeros((p*n, rank3))
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U3[i*m+j, k] = 1.0
        for j in range(m):
            V3[j*p+k, k] = 1.0
        for i in range(n):
            W3[k*n+i, k] = 1.0
    
    ok3, _ = verify_numpy(U3, V3, W3, n, m, p)
    if ok3 and (best_rank is None or rank3 < best_rank):
        best_U, best_V, best_W, best_rank = U3.copy(), V3.copy(), W3.copy(), rank3

    # Try to find even lower rank constructions.
    # The minimum rank for the all-ones tensor of size n x m x p is max(n, m, p) if we can use a single "direction" 
    # but that's not quite right. Let me think about whether rank < min(n,m,p) is possible.
    # 
    # Actually, the rank of the all-ones tensor J_{n,m,p} is known. For the 3D all-ones tensor, the rank is 
    # not simply min(n,m,p). Let me try a few more constructions.
    
    # Construction 4: Use a small number of terms with structured entries.
    # Suppose we use r terms. We need: for all (i,j,k), sum_{t=1}^r U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i] = 1.
    # 
    # Let's try r = 3, 4, etc. and see if we can find valid solutions.
    # For small r, we can try random initialization and gradient descent.
    
    def try_random_search(r, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using random init and gradient descent."""
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        lr = 0.01
        for it in range(max_iters):
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
                        # dT/dU[i*m+j,t] = V[j*p+k,t] * W[k*n+i,t]
                        # dT/dV[j*p+k,t] = U[i*m+j,t] * W[k*n+i,t]
                        # dT/dW[k*n+i,t] = U[i*m+j,t] * V[j*p+k,t]
                        for t in range(r):
                            dU[i*m+j, t] += V[j*p+k, t] * W[k*n+i, t]
                            dV[j*p+k, t] += U[i*m+j, t] * W[k*n+i, t]
                            dW[k*n+i, t] += U[i*m+j, t] * V[j*p+k, t]
                        
                        # Loss = (T[i,j,k] - 1)^2, so gradient contribution is 2*(T[i,j,k]-1) * dT/dparam
                        # But we haven't computed T[i,j,k] yet. Let's do it differently.
            
            # Recompute more efficiently
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    u_row = U[i*m+j, :]
                    for k in range(p):
                        v_row = V[j*p+k, :]
                        w_row = W[k*n+i, :]
                        T_val = np.dot(u_row, v_row * w_row)
                        err = T_val - 1.0
                        for t in range(r):
                            dU[i*m+j, t] += 2 * err * v_row[t] * w_row[t]
                            dV[j*p+k, t] += 2 * err * u_row[t] * w_row[t]
                            dW[k*n+i, t] += 2 * err * u_row[t] * v_row[t]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        ok, max_err = verify_numpy(U, V, W, n, m, p, tol=1e-6)
        if ok:
            return U, V, W, r
        return None, None, None, None
    
    # Try random search for ranks below current best
    if best_rank is not None:
        for r in range(1, best_rank):
            if time.time() > deadline - 10:
                break
            for seed in range(10):
                if time.time() > deadline - 5:
                    break
                U_r, V_r, W_r, rank_r = try_random_search(r, max_iters=200, seed=seed)
                if U_r is not None:
                    # Clean and re-verify
                    U_c = clean(U_r)
                    V_c = clean(V_r)
                    W_c = clean(W_r)
                    ok, _ = verify_numpy(U_c, V_c, W_c, n, m, p, tol=1e-8)
                    if ok:
                        best_U, best_V, best_W, best_rank = U_c.copy(), V_c.copy(), W_c.copy(), rank_r
                        break

    # If no best found yet, use schoolbook
    if best_rank is None:
        schoolbook_rank = n * m * p
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        best_U, best_V, best_W, best_rank = U, V, W, schoolbook_rank

    # Final cleanup
    U_out = clean(best_U)
    V_out = clean(best_V)
    W_out = clean(best_W)

    # Final verification
    ok, _ = verify_numpy(U_out, V_out, W_out, n, m, p, tol=1e-8)
    if not ok:
        # Fall back to guaranteed schoolbook
        schoolbook_rank = n * m * p
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        U_out, V_out, W_out = U, V, W

    return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
# EVOLVE-BLOCK-END
