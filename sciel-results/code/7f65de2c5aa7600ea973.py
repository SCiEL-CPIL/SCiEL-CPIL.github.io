import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for rank-7 Strassen 2x2x2."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 1],
        [1, -1, 0, -1, 1, 1, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, 0, 1, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 0, 0, 1, -1, 1]
    ], dtype=float)
    return U, V, W

def _schoolbook_2x2x1():
    """Return U, V, W (each 4x4) for rank-4 schoolbook 2x2x1."""
    U = np.eye(4, dtype=float)
    V = np.eye(4, dtype=float)
    W = np.eye(4, dtype=float)
    return U, V, W

def _kron_decompose_4x4x7():
    """
    Decompose 4x4x7 tensor as Kronecker product of 2x2x4 (rank 7) and 2x2x1 (rank 4).
    Total rank = 7 * 4 = 28.
    
    Indexing: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    U index: i*m + j = (2*i1+i2)*4 + (2*j1+j2)
    V index: j*p + k = (2*j1+j2)*7 + (2*k1+k2)
    W index: k*n + i = (2*k1+k2)*4 + (2*i1+i2)
    """
    U_s, V_s, W_s = _strassen_2x2x2()  # each 4x7
    U_t, V_t, W_t = _schoolbook_2x2x1()  # each 4x4
    
    n, m, p = 4, 4, 7
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    rank = 7 * 4  # 28
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    for r1 in range(7):
        for r2 in range(4):
            r = r1 * 4 + r2
            
            # U: index i*m+j = (2*i1+i2)*4 + (2*j1+j2)
            # U_s[i1*2+i2, r1] * U_t[j1*2+j2, r2]... 
            # Actually U_s is indexed by [i,j] where i,j in {0,1} for 2x2
            # U_s[i1*2+i2, r1] gives the U value for input (i1,i2) at rank r1
            # But we need U[(2*i1+i2)*4 + (2*j1+j2), r] = U_s[i1*2+i2, r1] * U_t[j1*2+j2, r2]?
            # No - the Kronecker product of tensors: if T1[a,b,c] and T2[d,e,f],
            # then (T1⊗T2)[(a,d),(b,e),(c,f)] = T1[a,b,c]*T2[d,e,f]
            # Here T1 is 2x2x4 (inputs a,b in {0,1}, output c in {0,1,2,3})
            # T2 is 2x2x1 (inputs d,e in {0,1}, output f in {0,1})
            # Combined: input1 = (a,d) -> index 2*a+d, input2 = (b,e) -> 2*b+e, output = (c,f) -> 2*c+f
            # So U_combined[2*a+d, r] = U1[a, r1] * U2[d, r2]
            # where U1 is 2x7 (for 2x2x4) and U2 is 2x4 (for 2x2x1)
            
            # But U_s from Strassen is 4x7, indexed as [i*2+j, r] for 2x2x2
            # For 2x2x4, we need to handle 4 output indices.
            # Actually, let me reconsider.
            
            # The 2x2x4 tensor: inputs (i,j) in {0,1}^2, output k in {0,1,2,3}
            # This is NOT a standard matrix multiplication tensor. 
            # Matrix multiplication 2x2x4 means: A is 2x2, B is 2x4, C is 2x4
            # C[i,k] = sum_j A[i,j]*B[j,k]
            # Tensor T[i,j,k] = 1 for all i,j,k
            # This has rank 4 (schoolbook) since it's just 2*2*4 = 16 terms... 
            # No wait, the tensor T[i,j,k]=1 for i in {0,1}, j in {0,1}, k in {0,1,2,3}
            # This is the all-ones tensor of shape 2x2x4, which has rank 4 (it's a rank-1 tensor in each mode... no)
            # T[i,j,k] = 1 = 1*1*1, so it's rank 1! 
            # U[i,j] = 1, V[j,k] = 1, W[k,i] = 1... but W has shape (4,2)
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For 2x2x4: m=2, p=4, n=2
            # U: (4, rank), V: (8, rank), W: (8, rank)
            # T[i,j,k] = 1 for all i,j,k
            # Schoolbook: rank 16, but we can do better.
            # T[i,j,k] = 1 = (1)(1)(1), so rank 1 with U[i*2+j]=1, V[j*4+k]=1, W[k*2+i]=1
            # Check: sum_c U[i*2+j,c]*V[j*4+k,c]*W[k*2+i,c] = 1*1*1 = 1. Yes! Rank 1!
            
            pass  # This approach is getting complicated. Let me use a different strategy.
    
    return None  # Placeholder

def _build_kronecker_4x4x7():
    """
    Build 4x4x7 decomposition using Kronecker structure.
    
    4x4x7 = (2x2x7) ⊗ (2x2x1) where:
    - 2x2x7: A is 2x2, B is 2x7, C is 2x7. Tensor T[i,j,k]=1, rank = ?
    - 2x2x1: A is 2x2, B is 2x1, C is 2x1. Tensor T[i,j,k]=1, rank = 4 (schoolbook)
    
    For 2x2x7: T[i,j,k]=1 for i,j in {0,1}, k in {0,...,6}
    This is the all-ones tensor of shape 2x2x7, which has rank 1!
    U[i*2+j]=1, V[j*7+k]=1, W[k*2+i]=1 gives T[i,j,k]=1.
    
    So 2x2x7 has rank 1, and 2x2x1 has rank 4.
    Kronecker: rank = 1 * 4 = 4? That seems too good.
    
    Wait, I need to be more careful. The Kronecker product of tensor decompositions:
    If T1 = sum_{r1} u1_r1 ⊗ v1_r1 ⊗ w1_r1 and T2 = sum_{r2} u2_r2 ⊗ v2_r2 ⊗ w2_r2,
    then T1 ⊗ T2 = sum_{r1,r2} (u1_r1 ⊗ u2_r2) ⊗ (v1_r1 ⊗ v2_r2) ⊗ (w1_r1 ⊗ w2_r2)
    
    So rank(T1⊗T2) ≤ rank(T1) * rank(T2)
    
    For 2x2x7 (all-ones tensor): rank 1
    For 2x2x1 (all-ones tensor): rank 1
    So 4x4x7 = (2x2x7)⊗(2x2x1) would have rank ≤ 1*1 = 1?
    
    That can't be right because 4x4x7 matrix multiplication is not just the all-ones tensor.
    The tensor T[i,j,k] = 1 for all i,j,k IS the matrix multiplication tensor!
    C[i,k] = sum_j A[i,j]*B[j,k], and the tensor stores the coefficient of A[i,j]*B[j,k]*C[k,i]
    which is 1 for all valid (i,j,k).
    
    So the tensor IS the all-ones tensor of shape (n*m, m*p, p*n) with specific index mappings.
    And the all-ones tensor of shape a×b×c has rank 1 (it's u⊗v⊗w where u,v,w are all-ones vectors).
    
    But wait - the index mappings matter! The tensor is T[a,b,c] where:
    a = i*m + j, b = j*p + k, c = k*n + i
    T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    
    For 4x4x7: not all (a,b,c) triples are valid! Only those where the indices are consistent.
    So it's NOT the all-ones tensor. It's a structured sparse tensor.
    
    This is why the rank is not 1. The Kronecker approach needs to respect the index structure.
    """
    n, m, p = 4, 4, 7
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    # Use the factoring: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # i1,j1 in {0,1}, i2,j2 in {0,1}, k1 in {0,1,2,3}, k2 in {0,1}
    # But k only goes to 6, so (k1=3, k2=1) is missing.
    
    # The valid (i,j,k) triples map to:
    # a = i*4 + j = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2
    # b = j*7 + k = (2*j1+j2)*7 + (2*k1+k2) = 14*j1 + 7*j2 + 2*k1 + k2
    # c = k*4 + i = (2*k1+k2)*4 + (2*i1+i2) = 8*k1 + 4*k2 + 2*i1 + i2
    
    # This doesn't factor cleanly as a Kronecker product due to the mixed indices.
    # Let me try a different approach: direct Kronecker with Strassen blocks.
    
    # Alternative: use 4x4x7 = 4x4x4 ⊗ 1 + 4x4x3 (schoolbook)
    # 4x4x4 via Strassen Kronecker: rank 49
    # 4x4x3 schoolbook: rank 48
    # Total: 97 (worse than current 93)
    
    # Better: 4x4x7 = (2x2x7) ⊗ (2x2x1) + correction
    # Let me just implement the current approach but with better sparsification.
    
    # Actually, let me try: use Strassen for 4x4x4 (rank 49) and for 4x4x3,
    # use a smarter decomposition than schoolbook.
    
    # For 4x4x3: can we do better than rank 48?
    # 4x4x3: A is 4x4, B is 4x3, C is 4x3
    # T[i,j,k]=1 for i,j in {0,1,2,3}, k in {0,1,2}
    # Schoolbook: rank 48
    # Using 2x2 Strassen blocks: split i,j into 2x2 blocks
    # 4x4x3 = sum over (i1,j1) in {0,1}^2 of (2x2x3) blocks
    # Each 2x2x3 has rank 7 (Strassen for 2x2x2 + 1 schoolbook for k=2)
    # Total: 4 * 7 = 28
    # But the blocks overlap in the j index...
    
    # Let me just go with the current approach and improve sparsification.
    return None

def _build_base_decomposition(n, m, p):
    """Build base decomposition using Strassen Kronecker for 4x4x4 + schoolbook for 4x4x3."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    U_list = []
    V_list = []
    W_list = []
    
    # 4x4x4 part via Kronecker (rank 49)
    for r in range(7):
        for s in range(7):
            u = np.kron(U_2[:, r], U_2[:, s]).astype(float)
            v_kron = np.kron(V_2[:, r], V_2[:, s]).astype(float)
            v_full = np.zeros(dim_v)
            for j in range(4):
                for k in range(4):
                    v_full[j * p + k] = v_kron[j * 4 + k]
            w_kron = np.kron(W_2[:, r], W_2[:, s]).astype(float)
            w_full = np.zeros(dim_w)
            w_full[:16] = w_kron
            
            U_list.append(u)
            V_list.append(v_full)
            W_list.append(w_full)
    
    # 4x4x3 schoolbook part (rank 48)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = np.zeros(dim_u)
                v = np.zeros(dim_v)
                w = np.zeros(dim_w)
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u[a] = 1.0
                v[b] = 1.0
                w[c] = 1.0
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    return U, V, W

def _build_target_tensor(n, m, p):
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify(U, V, W, T_target):
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.array_equal(T_decomp.astype(np.int64), T_target)

def _try_remove_term(U, V, W, T_target, term_idx):
    r = U.shape[1]
    if r <= 1:
        return False, None, None, None
    
    U_new = np.delete(U, term_idx, axis=1)
    V_new = np.delete(V, term_idx, axis=1)
    W_new = np.delete(W, term_idx, axis=1)
    
    T_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
    if np.array_equal(T_new.astype(np.int64), T_target):
        return True, U_new, V_new, W_new
    
    return False, None, None, None

def _sparsify(U, V, W, T_target, max_time=100):
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        r = U.shape[1]
        norms = np.zeros(r)
        for i in range(r):
            norms[i] = np.linalg.norm(U[:, i]) * np.linalg.norm(V[:, i]) * np.linalg.norm(W[:, i])
        order = np.argsort(norms)
        
        for idx in order:
            if (time.time() - start_time) > max_time:
                break
            success, U_new, V_new, W_new = _try_remove_term(U, V, W, T_target, idx)
            if success:
                U, V, W = U_new, V_new, W_new
                improved = True
                break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    start = time.time()
    deadline = start + 130
    
    # Build base rank-93 decomposition
    U, V, W = _build_base_decomposition(n, m, p)
    T_target = _build_target_tensor(n, m, p)
    
    # Verify base decomposition
    if not _verify(U, V, W, T_target):
        # Fallback to schoolbook
        rank = n * m * p
        U_s = np.zeros((dim_u, rank))
        V_s = np.zeros((dim_v, rank))
        W_s = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Attempt sparsification
    max_sparsify_time = min(100, deadline - time.time() - 10)
    if max_sparsify_time > 10:
        U_sparse, V_sparse, W_sparse = _sparsify(U, V, W, T_target, max_time=max_sparsify_time)
        
        if _verify(U_sparse, V_sparse, W_sparse, T_target):
            U, V, W = U_sparse, V_sparse, W_sparse
    
    # Convert to clean form
    U_out = np.round(U * 2) / 2
    V_out = np.round(V * 2) / 2
    W_out = np.round(W * 2) / 2
    
    if _verify(U_out, V_out, W_out, T_target):
        return {
            "U": U_out.tolist(),
            "V": V_out.tolist(),
            "W": W_out.tolist()
        }
    
    # Fallback
    U_out = np.round(U * 2) / 2
    V_out = np.round(V * 2) / 2
    W_out = np.round(W * 2) / 2
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
