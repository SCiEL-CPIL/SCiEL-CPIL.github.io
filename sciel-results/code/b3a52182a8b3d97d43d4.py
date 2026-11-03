"""Matrix multiplication 4,4,5 tensor decomposition with combinatorial subspace elimination."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """
    Returns 7 terms for 2x2x2 Strassen.
    Each term is (u_sparse, v_sparse, w_sparse) where sparse is list of (index, value).
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _construct_rank_65(n=4, m=4, p=5):
    """
    Constructs a rank-65 decomposition.
    49 terms from 4x4x4 Kronecker Strassen.
    16 terms from 4x4x1 Naive.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i

    strassen_terms = _strassen_2x2x2_terms()
    
    # 1. 4x4x4 part (k=0..3)
    # Kronecker product of two 2x2x2 Strassen decompositions
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # U: i = i1*2 + i2, j = j1*2 + j2
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2
            
            # V: j = j1*2 + j2, k = k1*2 + k2
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
            
            # W: k = k1*2 + k2, i = i1*2 + i2
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
            
            add_term(u, v, w)
            
    # 2. 4x4x1 part (k=4)
    # Naive: A[i,j] * B[j,4] -> C[4,i]
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m+j, :]
                v_val = V_np[j*p+k, :]
                w_val = W_np[k*n+i, :]
                total = np.dot(u_val * v_val, w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 80."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    return U, V, W

def _try_reduce_rank_by_elimination(U, V, W, n, m, p, tol=1e-9):
    """
    Attempts to reduce rank by finding linear dependencies among the term tensors.
    Each term r corresponds to a tensor T_r = U[:,r] \otimes V[:,r] \otimes W[:,r].
    We form a matrix M where columns are vec(T_r).
    If rank(M) < num_terms, we can express the target tensor (ones) in a smaller basis.
    """
    num_terms = U.shape[1]
    tensor_dim = n * m * p
    
    # Build the matrix of term tensors (flattened)
    # M shape: (tensor_dim, num_terms)
    M = np.zeros((tensor_dim, num_terms))
    
    for r in range(num_terms):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        
        # Tensor product T_r[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        # Flatten in order i, j, k matching W[k*n+i] convention?
        # The verification loop uses:
        # total = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # So the index in the flattened tensor for (i,j,k) is not directly a simple row index in M
        # unless we construct M carefully.
        
        # Let's just compute the contribution to each (i,j,k) directly to fill M.
        # M[index(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient way:
        # u_flat = U[:, r] (size nm)
        # v_flat = V[:, r] (size mp)
        # w_flat = W[:, r] (size pn)
        
        # We need to map (i,j,k) to a single index idx in 0..n*m*p-1.
        # Let idx = i * (m*p) + j * p + k.
        
        for i in range(n):
            for j in range(m):
                u_val = U[i*m+j, r]
                if u_val == 0: continue
                for k in range(p):
                    v_val = V[j*p+k, r]
                    if v_val == 0: continue
                    w_val = W[k*n+i, r]
                    if w_val == 0: continue
                    idx = i * (m * p) + j * p + k
                    M[idx, r] = u_val * v_val * w_val

    # Target tensor is all ones
    target = np.ones(tensor_dim)
    
    # Check if target is in the span of columns of M
    # Solve M @ c = target
    # Use least squares to find coefficients and check residual
    try:
        c, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=tol)
    except Exception:
        return None, None, None, num_terms
    
    if residuals and len(residuals) > 0:
        if residuals[0] > 1e-8:
            return None, None, None, num_terms
    else:
        # If residuals empty, check norm
        if np.linalg.norm(M @ c - target) > 1e-8:
            return None, None, None, num_terms

    # If we have a solution, the effective rank is the rank of M (or rank of the subspace spanned by used columns)
    # However, lstsq gives a minimum norm solution. The rank of the decomposition is the number of non-zero coefficients
    # IF the basis vectors corresponding to non-zero coefficients are linearly independent.
    
    # To be safe, we identify the pivot columns from the QR decomposition or SVD.
    # Actually, if M has rank R, we can pick R linearly independent columns and express target as their combo.
    
    # Let's use SVD to find a basis
    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Determine effective rank
    effective_rank = 0
    for i in range(len(s)):
        if s[i] > tol * max(s):
            effective_rank += 1
        else:
            break
            
    if effective_rank >= num_terms:
        return None, None, None, num_terms
        
    # Construct new U, V, W using the first effective_rank columns of M as basis?
    # No, the columns of M are the flattened tensors of the ORIGINAL terms.
    # We need to express the target as a combo of the ORIGINAL terms.
    # The coefficients c from lstsq give us the weights for the original terms.
    # If we just keep the terms with non-zero weights, do we reduce rank?
    # Not necessarily, because the terms with non-zero weights might be linearly dependent.
    
    # Correct approach:
    # 1. Find a basis for the column space of M. Let B be a matrix of linearly independent columns from M.
    # 2. Express target as B @ d.
    # 3. Express each column of B as a linear combination of the original columns of M.
    #    B_col = M @ x. So B = M @ X.
    # 4. Then target = M @ X @ d = M @ (X @ d).
    #    This means target is a combination of original terms with coefficients c' = X @ d.
    #    This doesn't reduce the number of terms unless we change the basis of the factors.
    
    # To reduce rank, we need to find a set of NEW factor matrices U', V', W' of lower rank.
    # This is equivalent to finding a lower rank decomposition of the tensor.
    # If the tensor lies in a subspace of rank R, there exists a decomposition of rank R.
    # We can construct it by taking the basis vectors of the subspace.
    
    # Let's try to construct the new factors from the SVD components.
    # This is tricky because the tensor is 3-way.
    
    # Alternative: If we found that the target is in the span of K < num_terms specific linearly independent original terms,
    # we can simply drop the others. But lstsq usually uses all columns.
    
    # Let's check if we can simply drop terms with zero coefficient in a minimal basis.
    # This is hard to do directly with lstsq.
    
    # Heuristic: If rank(M) < num_terms, we have a lower rank decomposition.
    # We can try to recover it by:
    # 1. Taking the R linearly independent columns of M (from QR pivots).
    # 2. Solving for their coefficients to form the target.
    # 3. Using ONLY those original terms in the decomposition.
    
    # Get pivot indices from QR
    Q, R_mat = np.linalg.qr(M, mode='complete')
    diag_R = np.abs(np.diag(R_mat))
    pivots = np.argsort(diag_R)[::-1][:effective_rank]
    
    # Select pivot columns
    M_piv = M[:, pivots]
    
    # Solve M_piv @ c_piv = target
    try:
        c_piv, res_piv, rank_piv, sv_piv = np.linalg.lstsq(M_piv, target, rcond=tol)
    except Exception:
        return None, None, None, num_terms
        
    if res_piv and len(res_piv) > 0 and res_piv[0] > 1e-8:
        return None, None, None, num_terms
        
    # Verify the new decomposition using only pivot terms
    U_new = U[:, pivots]
    V_new = V[:, pivots]
    W_new = W[:, pivots]
    
    # Scale the factors to match the coefficients?
    # The term is U[:,r] * V[:,r] * W[:,r].
    # We want sum_piv c_piv[p] * (U[:,p] * V[:,p] * W[:,p]) = target.
    # We can absorb the coefficient c_piv[p] into one of the factors, e.g., U_new[:,p] *= c_piv[p].
    # But this might break integer/half-integer property if c_piv is not nice.
    # However, the task allows integer/half-integer.
    # If c_piv are rational, we can scale.
    
    # Let's check if the resulting decomposition is valid.
    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
        # It works without scaling? That implies c_piv were all 1? Unlikely.
        # The verification checks if sum(U*V*W) == 1.
        # If we just removed columns, the sum changed.
        # We MUST apply the coefficients.
        
        # Apply coefficients to U_new
        U_scaled = U_new * c_piv
        V_scaled = V_new
        W_scaled = W_new
        
        if _verify_decomposition(U_scaled, V_scaled, W_scaled, n, m, p, tol=1e-8):
            return U_scaled, V_scaled, W_scaled, len(pivots)
            
    return None, None, None, num_terms

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Strassen-based rank-65 construction
    try:
        U, V, W = _construct_rank_65(n, m, p)
        rank = U.shape[1]
        
        if _verify_decomposition(U, V, W, n, m, p):
            candidates = [(rank, U, V, W)]
        else:
            candidates = []
    except Exception:
        candidates = []
    
    # Pick the best candidate
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_rank, U, V, W = candidates[0]
        best_result = (U, V, W)
        
        # Approach 2: Try to reduce rank via subspace elimination
        if time.time() < deadline - 10:
            try:
                U_red, V_red, W_red, new_rank = _try_reduce_rank_by_elimination(U, V, W, n, m, p)
                if U_red is not None and new_rank < best_rank:
                    # Verify integer/half-integer property if possible, or just trust the float result for now
                    # The task says "integer/half-integer entries".
                    # If the elimination produced floats, we might fail the validity check.
                    # We should check if entries are close to half-integers.
                    def _is_half_int(M, tol=1e-6):
                        for val in M:
                            if abs(val * 2 - round(val * 2)) > tol:
                                return False
                        return True
                    
                    if _is_half_int(U_red) and _is_half_int(V_red) and _is_half_int(W_red):
                        best_result = (U_red, V_red, W_red)
                        best_rank = new_rank
            except Exception:
                pass

    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
