import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 42

    # Build schoolbook decomposition as fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def to_half_int_list(matrix):
        """Convert numpy matrix to list of lists with half-integer entries"""
        return [[float(matrix[i, j]) for j in range(matrix.shape[1])] for i in range(matrix.shape[0])]

    def verify_exact(U, V, W, rank, n, m, p):
        """Verify exact decomposition using integer arithmetic."""
        U2 = np.round(U * 2).astype(np.int64)
        V2 = np.round(V * 2).astype(np.int64)
        W2 = np.round(W * 2).astype(np.int64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U2[i*m+j, r] * V2[j*p+k, r] * W2[k*n+i, r]
                    if total != 8:
                        return False
        return True

    # Build the dense matmul tensor
    T_dense = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_dense[i * m + j, j * p + k, k * n + i] = 1.0

    def peel_rank1_hosvd(T_flat, n, m, p):
        """Peel off best rank-1 component using HOSVD."""
        nm, mp, pn = n * m, m * p, p * n
        T3 = T_flat.reshape(nm, mp, pn)
        
        T1 = T3.reshape(nm, mp * pn)
        U1, S1, Vt1 = np.linalg.svd(T1, full_matrices=False)
        
        T2 = T3.transpose(1, 0, 2).reshape(mp, nm * pn)
        U2, S2, Vt2 = np.linalg.svd(T2, full_matrices=False)
        
        T3u = T3.transpose(2, 0, 1).reshape(pn, nm * mp)
        U3, S3, Vt3 = np.linalg.svd(T3u, full_matrices=False)
        
        u = U1[:, 0]
        v = U2[:, 0]
        w = U3[:, 0]
        
        alpha = np.einsum('i,j,k,ijk->', u, v, w, T3)
        
        T_residual = T3 - alpha * np.einsum('i,j,k->ijk', u, v, w)
        
        return u, v, w, alpha, T_residual.reshape(nm * mp * pn)

    def local_refine_halfint(U, V, W, rank, n, m, p, max_iters=500):
        """Local search to adjust to half-integers while maintaining exactness."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for _ in range(max_iters):
            U_half = np.round(U * 2) / 2.0
            V_half = np.round(V * 2) / 2.0
            W_half = np.round(W * 2) / 2.0
            if np.allclose(U, U_half, atol=1e-10) and np.allclose(V, V_half, atol=1e-10) and np.allclose(W, W_half, atol=1e-10):
                if verify_exact(U_half, V_half, W_half, rank, n, m, p):
                    return U_half, V_half, W_half
                else:
                    return None
            
            errors = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for r in range(rank):
                            total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        errors[i, j, k] = total - 1.0
            
            if np.max(np.abs(errors)) < 1e-12:
                U_half = np.round(U * 2) / 2.0
                V_half = np.round(V * 2) / 2.0
                W_half = np.round(W * 2) / 2.0
                if verify_exact(U_half, V_half, W_half, rank, n, m, p):
                    return U_half, V_half, W_half
                else:
                    return None
            
            idx = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
            i, j, k = idx
            err = errors[i, j, k]
            
            r = np.random.randint(rank)
            if V[j*p+k, r] != 0 and W[k*n+i, r] != 0:
                delta = err / (V[j*p+k, r] * W[k*n+i, r])
                U[i*m+j, r] += 0.05 * delta
            
            if _ % 50 == 49:
                U_half = np.round(U * 2) / 2.0
                V_half = np.round(V * 2) / 2.0
                W_half = np.round(W * 2) / 2.0
                if verify_exact(U_half, V_half, W_half, rank, n, m, p):
                    return U_half, V_half, W_half
        
        return None

    T_flat = T_dense.reshape(nm * mp * pn)
    
    best_result = None
    
    for target_rank in range(21, 43):
        found = False
        for restart in range(3):
            np.random.seed(restart * 100 + target_rank)
            T_remaining = T_flat.copy()
            U_cols = []
            V_cols = []
            W_cols = []
            
            for r in range(target_rank):
                u, v, w, alpha, T_remaining = peel_rank1_hosvd(T_remaining, n, m, p)
                U_cols.append(alpha * u)
                V_cols.append(v)
                W_cols.append(w)
            
            U = np.column_stack(U_cols)
            V = np.column_stack(V_cols)
            W = np.column_stack(W_cols)
            
            result = local_refine_halfint(U, V, W, target_rank, n, m, p)
            if result is not None:
                U_half, V_half, W_half = result
                best_result = (U_half, V_half, W_half)
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": to_half_int_list(U), "V": to_half_int_list(V), "W": to_half_int_list(W)}
    
    return {"U": to_half_int_list(U_school), "V": to_half_int_list(V_school), "W": to_half_int_list(W_school)}
