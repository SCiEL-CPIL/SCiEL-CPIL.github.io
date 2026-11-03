"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Fast verification using tensor contraction
    def verify_fast(U, V, W):
        """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
        rank = U.shape[1]
        # Reshape for batch computation
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # Compute sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c]
        # Use einsum: sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        result = np.einsum('ijc,jkc,kic->', U_r, V_r, W_r)  # This gives scalar sum over ALL i,j,k
        # We need per-(i,j,k) values
        # Better: compute T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # Step 1: A[i,j,k] = sum_c U[i,j,c]*V[j,k,c]  -> shape (n,m,p)
        A = np.einsum('ijc,jkc->ijk', U_r, V_r)
        # Step 2: T[i,j,k] = sum_c A[i,j,k_c]*W[k,i,c]... no that's wrong
        # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # = sum_c (U[i,j,c]*W[k,i,c]) * V[j,k,c]
        # Let B[i,j,k] = sum_c U[i,j,c]*W[k,i,c]  -> shape (n,m,p)
        B = np.einsum('ijc,kic->ijk', U_r, W_r)
        # Then T[i,j,k] = sum_c B[i,j,k_c]*V[j,k,c]... no
        # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # = sum_c V[j,k,c] * (U[i,j,c]*W[k,i,c])
        # For fixed i,j,k: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # Let's do it directly with einsum
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        # T should be all ones
        return np.allclose(T, 1.0, atol=1e-9)
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    def add_strassen_2x2x2(i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen terms for a 2x2x2 block."""
        # M1 = (A00+A11)(B00+B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1; w[k1*n+i1] = 1
        add_term(u, v, w)
        
        # M2 = (A10+A11)B00
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1
        w[k0*n+i1] = 1; w[k1*n+i1] = -1
        add_term(u, v, w)
        
        # M3 = A00(B01-B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1
        v[j0*p+k1] = 1; v[j1*p+k1] = -1
        w[k1*n+i0] = 1; w[k1*n+i1] = 1
        add_term(u, v, w)
        
        # M4 = A11(B10-B00)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j1] = 1
        v[j1*p+k0] = 1; v[j0*p+k0] = -1
        w[k0*n+i0] = 1; w[k0*n+i1] = 1
        add_term(u, v, w)
        
        # M5 = (A00+A01)B11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i0*m+j1] = 1
        v[j1*p+k1] = 1
        w[k0*n+i0] = 1; w[k1*n+i0] = -1
        add_term(u, v, w)
        
        # M6 = (A10-A00)(B00+B01)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j0] = 1; u[i0*m+j0] = -1
        v[j0*p+k0] = 1; v[j0*p+k1] = 1
        w[k1*n+i1] = 1
        add_term(u, v, w)
        
        # M7 = (A01-A11)(B10+B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j1] = 1; u[i1*m+j1] = -1
        v[j1*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1
        add_term(u, v, w)
    
    def add_schoolbook_term(i, j, k):
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i*m+j] = 1
        v[j*p+k] = 1
        w[k*n+i] = 1
        add_term(u, v, w)
    
    # Split: i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5,6}
    # i: 2+1 -> {0,1} and {2}
    # j: 2+2+1 -> {0,1}, {2,3}, {4}
    # k: 2+2+2+1 -> {0,1}, {2,3}, {4,5}, {6}
    
    i_sets = [{0, 1}, {2}]
    j_sets = [{0, 1}, {2, 3}, {4}]
    k_sets = [{0, 1}, {2, 3}, {4, 5}, {6}]
    
    for I in i_sets:
        for J in j_sets:
            for K in k_sets:
                if len(I) == 2 and len(J) == 2 and len(K) == 2:
                    i0, i1 = sorted(I)
                    j0, j1 = sorted(J)
                    k0, k1 = sorted(K)
                    add_strassen_2x2x2(i0, i1, j0, j1, k0, k1)
                else:
                    for i in I:
                        for j in J:
                            for k in K:
                                add_schoolbook_term(i, j, k)
    
    U_mat = np.array(U_list).T
    V_mat = np.array(V_list).T
    W_mat = np.array(W_list).T
    
    if not verify_fast(U_mat, V_mat, W_mat):
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Fast greedy term removal using incremental verification
    # Precompute contributions for speed
    def can_remove(idx, U_m, V_m, W_m):
        """Check if term idx can be removed."""
        U_new = np.delete(U_m, idx, axis=1)
        V_new = np.delete(V_m, idx, axis=1)
        W_new = np.delete(W_m, idx, axis=1)
        return verify_fast(U_new, V_new, W_new)
    
    # Greedy removal: repeatedly try to remove each term
    removed = True
    while removed and time.time() < deadline - 15:
        removed = False
        for idx in range(U_mat.shape[1]):
            if time.time() >= deadline - 15:
                break
            if can_remove(idx, U_mat, V_mat, W_mat):
                U_mat = np.delete(U_mat, idx, axis=1)
                V_mat = np.delete(V_mat, idx, axis=1)
                W_mat = np.delete(W_mat, idx, axis=1)
                removed = True
                break
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}

# EVOLVE-BLOCK-END
