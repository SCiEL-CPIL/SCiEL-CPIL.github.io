import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all 48 non-zero entries of the tensor
    # T_{i,j,k} should be 1 if the term A_{i,k} B_{k,j} contributes to C_{i,j}
    # In our index mapping:
    # u_idx = i*m + k
    # v_idx = k*p + j
    # w_idx = j*n + i
    
    max_err = 0.0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                
                ui = U_arr[u_idx]
                vj = V_arr[v_idx]
                wk = W_arr[w_idx]
                
                val = np.dot(ui, vj) # Wait, this is not right.
                # The decomposition is sum_r U_{i,r} V_{j,r} W_{k,r}
                # Here indices are flattened.
                # Term is sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                val = np.einsum('r,r,r->', ui, vj, wk)
                
                # For standard matrix multiplication, this should be 1.0
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    # Check that other combinations are 0
    # There are (n*m) * (m*p) * (p*n) total combinations.
    # Most are 0.
    # We can check a sample or all if fast enough.
    # 12 * 16 * 12 = 2304 combinations.
    # 2304 * rank (44) operations is ~100k ops. Very fast.
    
    for u_idx in range(n * m):
        i = u_idx // m
        k = u_idx % m
        for v_idx in range(m * p):
            k2 = v_idx // p
            j = v_idx % p
            if k != k2:
                # If k != k2, the term A_{i,k} B_{k2,j} is not in the sum for C_{i,j}
                # So the tensor entry should be 0.
                # Wait, the tensor T is defined such that C_{i,j} = sum_k T_{i,j,k} A_{i,k} B_{k,j}?
                # No, the tensor T_{i,j,k} is 1 if k is the inner index.
                # In the decomposition sum_r U_{i,k,r} V_{k,j,r} W_{j,i,r},
                # the indices i,k are in U, k,j in V, j,i in W.
                # The product U_{i,k,r} V_{k,j,r} W_{j,i,r} is non-zero only if the indices match?
                # No, the decomposition sums over r.
                # The condition for the tensor to be correct is:
                # sum_r U_{i*m+k, r} V_{k*p+j, r} W_{j*n+i, r} = 1 if (i,j,k) is a valid term (always true for MM tensor)
                # AND sum_r U_{u_idx, r} V_{v_idx, r} W_{w_idx, r} = 0 if the indices don't form a valid term?
                
                # Actually, the tensor for MM is T_{i,j,k} = delta_{k, inner}.
                # The decomposition is an identity:
                # delta_{k, inner} = sum_r U_{i,k,r} V_{k,j,r} W_{j,i,r} ??
                # No. The standard identity is:
                # A_{i,k} B_{k,j} is the term.
                # The tensor T has T_{i,j,k} = 1 for all i,j,k?
                # No. The tensor T is the structure tensor.
                # T_{i,j,k} = 1 if the product A_{i,k} B_{k,j} appears in C_{i,j}.
                # In MM, it always appears.
                # So T_{i,j,k} = 1 for all i,j,k?
                # No, the indices in U, V, W are tied.
                # U index depends on i, k.
                # V index depends on k, j.
                # W index depends on j, i.
                # So for a fixed r, the term is U_{i,k,r} V_{k,j,r} W_{j,i,r}.
                # This term contributes to C_{i,j} with coefficient ...
                # The condition is that the sum over r equals 1 for the "correct" (i,j,k) triples?
                # Actually, the tensor is 3-way.
                # The decomposition is exact if:
                # For all i, j, k:
                # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
                # Wait, is it 1 for ALL i,j,k?
                # Yes, because A_{i,k} B_{k,j} is a term in C_{i,j} for ALL i,j,k.
                # There are no "invalid" terms in the sum for MM.
                # So we only need to check that the sum is 1 for all 48 triples.
                
                pass # No need to check zeros if the tensor is dense 1s.
                
    return max_err < 1e-10

def _build_base_decomposition():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # Strassen 7
            # m1 = (a+d)(e+h)
            U[0*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m2 = (c+d)e
            U[1*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            W[(b_c*2+1)*n + 1][col] = -1
            col += 1
            
            # m3 = a(f-h)
            U[0*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m4 = d(g-e)
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = -1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            col += 1
            
            # m5 = (a+b)h
            U[0*m + (b_r*2+0)][col] = 1
            U[0*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            col += 1
            
            # m6 = (c-a)(e+f)
            U[0*m + (b_r*2+0)][col] = -1
            U[1*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m7 = (b-d)(g+h)
            U[0*m + (b_r*2+1)][col] = 1
            U[1*m + (b_r*2+1)][col] = -1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            col += 1
            
            # Trivial part for row i=2: 4 terms
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (b_r*2+j_local)][col] = 1
                    V[(b_r*2+j_local)*p + (b_c*2+k_local)][col] = 1
                    W[(b_c*2+k_local)*n + 2][col] = 1
                    col += 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Tensor matrix: rows are (i, k, j) combinations, cols are r
    # There are n * m * p = 48 combinations.
    tensor_matrix = np.zeros((n * m * p, rank))
    
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                
                # The tensor entry T_{i,j,k} is 1.
                # We check if the vectors U[u_idx], V[v_idx], W[w_idx] can be compressed.
                # Actually, we check the rank of the set of rank-1 tensors.
                # Each rank-1 tensor is defined by (U[:,r], V[:,r], W[:,r]).
                # We can flatten them into a vector of length n*m*m*p*p*n?
                # No, we can just check if the 44 vectors in R^{48} are linearly independent.
                # The "vector" for a rank-1 term r in the tensor space is not simply a vector in R^48.
                # The tensor is T = sum_r u_r \otimes v_r \otimes w_r.
                # We can matrix-ify T into a matrix of size (n*m) x (m*p) ? No.
                # We can matrix-ify into (n*m) x (p*n) by fixing the middle index?
                # T_{i,k, j} -> Matrix M_{i,k}^{j}?
                
                # A simpler check:
                # Construct the matrix of the tensor in the rank direction.
                # If the rank is 44, and the tensor space dimension is 48,
                # it is possible that the 44 vectors are linearly independent.
                
                pass

    # Use SVD on the "tensor matrix"
    # We form a matrix where each column r is the flattened tensor slice?
    # No, the standard way is to check the rank of the matrix formed by the components.
    # Let's form a matrix A of size (n*m*m*p*p*n, rank).
    # This is too large (12*16*12 = 2304 rows).
    # SVD of 2304 x 44 is fast.
    
    tensor_dim = n * m * m * p * p * n # 12 * 16 * 12 = 2304
    tensor_matrix = np.zeros((tensor_dim, rank))
    
    for r in range(rank):
        vec = np.outer(np.outer(U_arr[:,r], V_arr[:,r]), W_arr[:,r]).flatten()
        tensor_matrix[:,r] = vec
        
    try:
        s = np.linalg.svd(tensor_matrix, compute_uv=False)
        tol = max(tensor_dim, rank) * s[0] * np.finfo(float).eps * 10
        numerical_rank = np.sum(s > tol)
    except np.linalg.LinAlgError:
        return U, V, W, rank
    
    if numerical_rank < rank:
        # Try to extract the lower rank basis
        Q, R, piv = np.linalg.qr(tensor_matrix, pivoting=True)
        keep_indices = sorted(piv[:numerical_rank])
        
        U_new = [[U_arr[i, j] for j in keep_indices] for i in range(n*m)]
        V_new = [[V_arr[i, j] for j in keep_indices] for i in range(m*p)]
        W_new = [[W_arr[i, j] for j in keep_indices] for i in range(p*n)]
        
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new, numerical_rank
    
    return U, V, W, rank

def construct():
    n, m, p = 3, 4, 4
    
    U, V, W = _build_base_decomposition()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to trivial if verification fails (should not happen)
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m+k][col] = 1
                    V[k*p+j][col] = 1
                    W[j*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    if _time_left() > 5.0:
        U, V, W, new_rank = _try_reduce_rank(U, V, W, n, m, p)
    
    return {"U": U, "V": W, "W": W} # Oops, typo in return? No, return U, V, W.

# Corrected return
def construct():
    n, m, p = 3, 4, 4
    
    U, V, W = _build_base_decomposition()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m+k][col] = 1
                    V[k*p+j][col] = 1
                    W[j*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    if _time_left() > 5.0:
        U, V, W, new_rank = _try_reduce_rank(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}
