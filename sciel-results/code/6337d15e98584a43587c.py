import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, p*n
    
    start = time.time()
    deadline = start + 140
    
    # Build the full multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # The 3x5x5 tensor is block-diagonal in the first index (i):
    # For each i in {0,1,2}, the sub-tensor is a 1x5x5 multiplication tensor.
    # Specifically, T[i*m+j, j*p+k, k*n+i] = 1 for j in 0..4, k in 0..4.
    # So we can decompose each 1x5x5 block independently and concatenate.
    
    # For 1x5x5 matrix multiplication, we need to decompose the tensor
    # T_block[j, k, k] = 1 for j in 0..4, k in 0..4
    # This is the tensor for multiplying a 1x5 row vector by a 5x5 matrix.
    # Wait, let me re-examine. For n=1, m=5, p=5:
    # T[i*m+j, j*p+k, k*n+i] with i=0: T[j, j*p+k, k] = 1
    # So T_block[j, j*p+k, k] = 1 for j,k in 0..4
    # This means output index is k (0..4), first index is j (0..4), second is j*p+k
    
    # Actually, let me think about this differently.
    # The 3x5x5 multiplication: C = A * B where A is 3x5, B is 5x5, C is 3x5.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # First index: i*m+j (row of A, col of A)
    # Second index: j*p+k (row of B, col of B)  
    # Third index: k*n+i (col of C, row of C)
    
    # For fixed i, the sub-tensor involves:
    # First indices: i*m + j for j=0..4 (5 values)
    # Second indices: j*p + k for j=0..4, k=0..4 (25 values)
    # Third indices: k*n + i for k=0..4 (5 values)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # So for each i, we have a 5x25x5 tensor. This is the tensor for
    # multiplying a 1x5 vector (row i of A) by a 5x5 matrix (B).
    # Output is 1x5 vector (row i of C).
    
    # The rank of 1x5x5 matrix multiplication:
    # This is equivalent to: given a 5-vector a and 5x5 matrix B, compute a*B.
    # (a*B)[k] = sum_j a[j] * B[j,k]
    # This can be done with rank 5 (just 5 scalar multiplications per output, but shared)
    # Actually, the tensor is sum over j,k of a_j * B_jk * e_k
    # So T = sum_{j,k} a_j * B_jk * e_k
    # This is a sum of 25 rank-1 tensors, but we can do better.
    
    # Actually, for 1xm x mxp multiplication, the rank is at most m*p (schoolbook)
    # but can be lower. For 1x5x5, schoolbook rank is 25.
    
    # But wait - we can use the fact that the output dimension is only 5.
    # Let me think: (a*B)[k] = sum_j a[j]*B[j,k]
    # This is like computing 5 dot products, each of length 5.
    # The tensor is: sum_{j=0}^{4} sum_{k=0}^{4} a_j * B_{j,k} * c_k
    # where c_k is the k-th basis vector.
    
    # A rank-1 decomposition would be: T = sum_r u_r * v_r * w_r
    # where u is 5-dim, v is 25-dim, w is 5-dim.
    
    # The minimal rank for 1x5x5... let me think.
    # We need: sum_r u_r[j] * v_r[j*p+k] * w_r[k] = 1 if j=k... no.
    # T[j, j*p+k, k] = 1 means for each (j,k) pair, there's a 1.
    # So T = sum_{j,k} e_j * e_{j*p+k} * e_k
    # This is already a sum of 25 rank-1 terms.
    
    # Can we do better? The tensor has rank at most 25 (schoolbook).
    # For 1xm x mxp, the rank is actually m*p in general, but for specific cases...
    
    # Hmm, actually for 1x5x5, I think the rank might be 25 or possibly lower.
    # Let me just use the schoolbook decomposition for each block and concatenate.
    # That gives rank 3*25 = 75, which is the same as the full schoolbook.
    
    # Let me try a different approach: use the full tensor and try to find a lower rank decomposition.
    
    # Actually, let me reconsider. The key insight is that for the 3x5x5 case,
    # we have 3 independent 1x5x5 sub-problems. If we can find a rank-r decomposition
    # for 1x5x5, the total rank would be 3*r.
    
    # For 1x5x5, let's try to find the minimal rank.
    # The tensor T_block[j, j*p+k, k] = 1 for j,k in 0..4.
    # This can be written as: T_block = sum_{j,k} e_j ⊗ e_{jp+k} ⊗ e_k
    
    # One approach: use the identity that sum_k e_k ⊗ e_k = I (in some sense)
    # But that doesn't directly help.
    
    # Let me just try the greedy SVD approach on the full tensor, but with better initialization.
    
    # Alternative: try to find a decomposition using random initialization and gradient descent.
    
    # For now, let's use the schoolbook rank of 75 as a baseline and try to improve.
    
    # Try greedy decomposition with SVD
    def greedy_decompose(T, nm, mp, np_, max_terms=60, deadline=None):
        U_list = []
        V_list = []
        W_list = []
        R = T.copy()
        
        for r in range(max_terms):
            if deadline is not None and time.time() > deadline - 10:
                break
            
            # Matricize along first mode
            R_mat = R.reshape(nm, mp * np_)
            try:
                U_svd, s, Vt_svd = np.linalg.svd(R_mat, full_matrices=False)
            except:
                break
            
            if len(s) == 0 or s[0] < 1e-12:
                break
            
            u = U_svd[:, 0]
            vw = Vt_svd[0].reshape(mp, np_)
            
            try:
                U_vw, s_vw, Vt_vw = np.linalg.svd(vw, full_matrices=False)
            except:
                break
            
            if len(s_vw) == 0 or s_vw[0] < 1e-12:
                break
            
            v = U_vw[:, 0]
            w = Vt_vw[0]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            
            R -= np.einsum('a,b,c->abc', u, v, w)
            
            if np.max(np.abs(R)) < 1e-12:
                break
        
        if len(U_list) == 0:
            return None
        
        U_arr = np.array(U_list).T
        V_arr = np.array(V_list).T
        W_arr = np.array(W_list).T
        
        # Check exactness
        T_check = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        if np.max(np.abs(T_check - T)) > 1e-8:
            return None
        
        # Try to round to half-integers
        U_r = np.round(U_arr * 2) / 2
        V_r = np.round(V_arr * 2) / 2
        W_r = np.round(W_arr * 2) / 2
        
        T_check2 = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.max(np.abs(T_check2 - T)) < 1e-6:
            return U_r, V_r, W_r
        
        # Try integer rounding
        U_i = np.round(U_arr)
        V_i = np.round(V_arr)
        W_i = np.round(W_arr)
        T_check3 = np.einsum('ar,br,cr->abc', U_i, V_i, W_i)
        if np.max(np.abs(T_check3 - T)) < 1e-6:
            return U_i.astype(float), V_i.astype(float), W_i.astype(float)
        
        return None
    
    result = greedy_decompose(T, nm, mp, np_, max_terms=60, deadline=deadline)
    
    if result is not None:
        U, V, W = result
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(np_)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
