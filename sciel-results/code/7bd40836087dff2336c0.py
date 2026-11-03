"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import time

def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p tensor decomposition for M_{n,m,p}."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that U, V, W form a valid tensor decomposition of M_{n,m,p}.
    Checks: C_{i,k} = sum_j A_{i,j} B_{j,k} == sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    """
    r = len(U[0]) if U else 0
    if r != len(V[0]) or r != len(W[0]):
        return False
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # The tensor T_{i,j,k} = delta_{i,j} * delta_{j,k} is what we need to match
    # via T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    # Wait, the standard matrix multiplication tensor is defined as:
    # M_{n,m,p} = sum_{i,j,k} e_i \otimes e_j \otimes e_k * (delta_{i,j} * delta_{j,k})? No.
    # The bilinear map is (A,B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is T = sum_{i,j,k} T_{i,j,k} e_i \otimes e_j \otimes e_k where T_{i,j,k} = 1 if j==j else 0?
    # Actually, the tensor decomposes as:
    # M = sum_{c} u_c \otimes v_c \otimes w_c
    # such that <M, a \otimes b \otimes c> = sum_j a_{i,j} b_{j,k} c_{k,i} ? No.
    # Let's use the standard definition:
    # We need sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = 1 if j==j (always) ? No.
    # The coefficient of A_{i,j} B_{j,k} in the computation of C_{i,k} is 1.
    # So for each (i,j,k), the sum over c of U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    # must be 1 if the indices are "aligned" in the multiplication sense?
    # No, the tensor is M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p E_{i,j} \otimes E_{j,k} \otimes E_{k,i} ?
    # Let's check the schoolbook:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # Product: U[i*m+j, c] V[j'*p+k', c] W[k''*n+i'', c].
    # Non-zero only if i=i', j=j', k=k' (for that specific c).
    # So the term for (i,j,k) is 1.
    # The tensor entry T_{i,j,k} = 1 for all i,j,k?
    # No, that would be the all-ones tensor.
    # Matrix multiplication: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a multilinear map. The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k ?
    # If T = sum_{i,j,k} e_i \otimes e_j \otimes e_k, then <T, a \otimes b \otimes c> = (sum_i a_i) (sum_j b_j) (sum_k c_k).
    # That is not matrix multiplication.
    
    # Let's re-read the task: "output factor follows the official trace convention W[k*n+i,r]".
    # The tensor for matrix multiplication M_{n,m,p} is defined such that:
    # C = A B  <=>  C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The corresponding 3-tensor is T_{i,j,k} = delta_{i,j} * delta_{j,k} ? No, that's trace.
    # The tensor is T = sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{k,i} ?
    # Let's check the schoolbook again.
    # c-th rank 1 term: u_c \otimes v_c \otimes w_c.
    # u_c is vector of length n*m, v_c of length m*p, w_c of length p*n.
    # u_c[i*m+j] = 1, v_c[j*p+k] = 1, w_c[k*n+i] = 1.
    # The contraction is sum_{i,j,k} u[i*m+j] v[j*p+k] w[k*n+i] ?
    # No, the evaluation of the tensor on (A,B,C) is sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} ?
    # If so, the tensor is T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}.
    # This is the "trace" tensor.
    # Is matrix multiplication equivalent to the trace tensor?
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The bilinear form is <C, A B> = sum_{i,k} C_{i,k} (sum_j A_{i,j} B_{j,k}) = sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k}.
    # This is a trilinear form.
    # The tensor is T_{i,j,k} = 1 if we identify C_{k,i} with C_{i,k}? No, indices are fixed.
    # The tensor entries are T_{i,j,k} = 1 for all i,j,k?
    # If T_{i,j,k} = 1, then sum_c u_c[i*m+j] v_c[j*p+k] w_c[k*n+i] should be 1.
    # In schoolbook, for a given (i,j,k), there is exactly one c where u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1.
    # So the sum is 1.
    # So the tensor being decomposed is the all-ones tensor?
    # But matrix multiplication is NOT the all-ones tensor.
    # The all-ones tensor corresponds to sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # This is NOT C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # 
    # Wait, the "official trace convention" might imply that the tensor is indeed the trace tensor?
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 7 matrix."
    # Usually, the tensor for matrix multiplication M_{n,m,p} is defined as:
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p E_{i,j} \otimes E_{j,k} \otimes E_{k,i} ?
    # No, that's not right either.
    # Let's look at the standard definition.
    # The tensor M_{n,m,p} is an element of R^{n*m} \otimes R^{m*p} \otimes R^{p*n}.
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_{i,j} \otimes e_{j,k} \otimes e_{k,i} ?
    # If I take the contraction with A, B, C:
    # sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = trace(C^T A B) ?
    # This is not the standard matrix multiplication.
    #
    # However, the schoolbook decomposition provided in the seed program is:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # This decomposes the tensor T where T[i*m+j, j*p+k, k*n+i] = 1?
    # No, T is a 3-index tensor.
    # The rank 1 term is u_c \otimes v_c \otimes w_c.
    # The sum is T = sum_c u_c \otimes v_c \otimes w_c.
    # T_{a,b,c} = sum_r u_r[a] v_r[b] w_r[c].
    # In schoolbook, for each (i,j,k), there is one r.
    # u_r[i*m+j] = 1, v_r[j*p+k] = 1, w_r[k*n+i] = 1.
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    # And T[a,b,c] = 0 otherwise?
    # This T is a permutation tensor.
    # Is this T the matrix multiplication tensor?
    # The matrix multiplication tensor is usually defined as:
    # M = sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{k,i} ?
    # Let's check the bilinear map defined by M.
    # <M, A \otimes B \otimes C> = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # This is NOT C = AB.
    #
    # BUT, the task says "multiplying a 3 by 3 matrix by a 3 by 7 matrix".
    # And "output factor follows the official trace convention W[k*n+i,r]".
    # This strongly suggests the tensor is indeed the one defined by the schoolbook, which is the "trace" or "cyclic" tensor.
    # Why would they call it matrix multiplication?
    # Maybe because if you fix C to be the identity, you get the trace?
    # Or maybe the "multiplication" is defined in this specific tensor algebra context.
    # Given the seed program uses this decomposition and scores 1.0 (which is the baseline),
    # I must verify against THIS tensor.
    #
    # Let's assume the tensor is T where T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    # Wait, is it 0 otherwise?
    # In the schoolbook, each c corresponds to a unique (i,j,k).
    # So T[i*m+j, j'*p+k', k''*n+i''] = 1 if i=i', j=j', k=k', and 0 otherwise.
    # Yes, this is a diagonal-like tensor in the index space.
    #
    # So the verification is:
    # For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    #   sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    # For all other index combinations (a,b,c) where a,b,c are not of the form (i*m+j, j*p+k, k*n+i) for some i,j,k:
    #   sum_c U[a, c] V[b, c] W[c_index, c] == 0
    #
    # Note: The indices a, b, c in the sum are just row indices of U, V, W.
    # U has n*m rows, V has m*p rows, W has p*n rows.
    # The "tensor" T is a 3-way array of size (n*m) x (m*p) x (p*n).
    # T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i.
    # Otherwise T[a,b,c] = 0.
    #
    # Let's implement this verification.

    U_mat = [[0.0]*r for _ in range(n*m)]
    V_mat = [[0.0]*r for _ in range(m*p)]
    W_mat = [[0.0]*r for _ in range(p*n)]
    for idx, row in enumerate(U):
        U_mat[idx] = row
    for idx, row in enumerate(V):
        V_mat[idx] = row
    for idx, row in enumerate(W):
        W_mat[idx] = row

    # Check all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for col in range(r):
                    s += U_mat[a][col] * V_mat[b][col] * W_mat[c][col]
                if abs(s - 1.0) > tol:
                    return False

    # Check zeros
    # We need to check all a in 0..n*m-1, b in 0..m*p-1, c in 0..p*n-1
    # such that (a,b,c) is NOT of the form (i*m+j, j*p+k, k*n+i)
    valid_combos = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_combos.add((i*m+j, j*p+k, k*n+i))

    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a,b,c) in valid_combos:
                    continue
                s = 0.0
                for col in range(r):
                    s += U_mat[a][col] * V_mat[b][col] * W_mat[c][col]
                if abs(s) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
import numpy as np

def _build_target_tensor(n, m, p):
    """Build the target tensor T where T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _als_decompose(T, rank, n_iter=200, tol=1e-10, seed=42):
    """
    Alternating Least Squares to find rank-`rank` decomposition of tensor T.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    np.random.seed(seed)
    n_m, m_p, p_n = T.shape
    
    # Initialize
    U = np.random.randn(n_m, rank) * 0.1
    V = np.random.randn(m_p, rank) * 0.1
    W = np.random.randn(p_n, rank) * 0.1
    
    # Scale to roughly match
    U *= np.linalg.norm(T.reshape(n_m, -1)) / (rank * np.linalg.norm(U))
    
    prev_err = float('inf')
    for it in range(n_iter):
        # Update U: for each a, T[a,:,:] = sum_r U[a,r] * (V * W)[r, b, c]
        # T[a] = U[a,:] @ (V ⊙ W) where (V⊙W)[r] = V[:,r] ⊗ W[:,r]
        VW = np.einsum('br,cr->bcr', V, W)  # shape (m_p, p_n, rank)
        VW_flat = VW.transpose(2, 0, 1).reshape(rank, m_p * p_n)  # shape (rank, m_p*p_n)
        T_a = T.reshape(n_m, m_p * p_n)  # shape (n_m, m_p*p_n)
        # Solve: T_a @ VW_flat.T = U @ (VW_flat @ VW_flat.T)
        # U = T_a @ VW_flat.T @ inv(VW_flat @ VW_flat.T)
        A = VW_flat @ VW_flat.T  # (rank, rank)
        B = T_a @ VW_flat.T  # (n_m, rank)
        try:
            U_new = np.linalg.solve(A, B.T).T
        except np.linalg.LinAlgError:
            U_new = np.linalg.lstsq(A, B.T, rcond=None)[0].T
        U = U_new
        
        # Update V: for each b, T[:,b,:] = sum_r V[b,r] * (U ⊙ W)[r, a, c]
        UW = np.einsum('ar,cr->acr', U, W)  # shape (n_m, p_n, rank)
        UW_flat = UW.transpose(2, 0, 1).reshape(rank, n_m * p_n)
        T_b = T.transpose(1, 0, 2).reshape(m_p, n_m * p_n)  # shape (m_p, n_m*p_n)
        A = UW_flat @ UW_flat.T
        B = T_b @ UW_flat.T
        try:
            V_new = np.linalg.solve(A, B.T).T
        except np.linalg.LinAlgError:
            V_new = np.linalg.lstsq(A, B.T, rcond=None)[0].T
        V = V_new
        
        # Update W: for each c, T[:,:,c] = sum_r W[c,r] * (U ⊙ V)[r, a, b]
        UV = np.einsum('ar,br->abr', U, V)  # shape (n_m, m_p, rank)
        UV_flat = UV.transpose(2, 0, 1).reshape(rank, n_m * m_p)
        T_c = T.transpose(2, 0, 1).reshape(p_n, n_m * m_p)  # shape (p_n, n_m*m_p)
        A = UV_flat @ UV_flat.T
        B = T_c @ UV_flat.T
        try:
            W_new = np.linalg.solve(A, B.T).T
        except np.linalg.LinAlgError:
            W_new = np.linalg.lstsq(A, B.T, rcond=None)[0].T
        W = W_new
        
        # Compute error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def _verify_fast(n, m, p, U, V, W, tol=1e-8):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = _build_target_tensor(n, m, p)
    
    err = np.max(np.abs(T_recon - T_target))
    return err < tol

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Start with schoolbook (rank 63)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using ALS
    # Try ranks from 62 down to 45 (known lower bound is ~41 for M_{3,3,7})
    T = _build_target_tensor(n, m, p)
    
    for rank in range(best_rank - 1, 44, -1):
        if time.time() > deadline:
            break
        
        # Try multiple seeds for robustness
        found = False
        for seed in [42, 123, 777, 2024]:
            if time.time() > deadline:
                break
            
            U, V, W, err = _als_decompose(T, rank, n_iter=300, tol=1e-12, seed=seed)
            
            if _verify_fast(n, m, p, U, V, W, tol=1e-8):
                # Found a valid decomposition!
                best_result = {
                    'U': U.tolist(),
                    'V': V.tolist(),
                    'W': W.tolist()
                }
                best_rank = rank
                found = True
                break
        
        if found:
            # Continue trying lower ranks
            continue
    
    return best_result
# EVOLVE-BLOCK-END
