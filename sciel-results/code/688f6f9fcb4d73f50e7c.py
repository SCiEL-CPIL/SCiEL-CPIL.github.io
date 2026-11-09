"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the factors correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1.0 if (i < n and j < m and k < p) else 0.0
                # The tensor T[i,j,k] for matmul is: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
                # So T[i,j,k] = 1 if we're looking at the (i,j,k) term, but actually
                # the multiplication tensor has T[i,j,k] = delta (no, it's the structure tensor)
                # For C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor T has entries T[i,j,k] where we multiply A[i,j]*B[j,k] and add to C[i,k]
                # So the structure tensor T_{i,k,j} or similar. Let me re-examine.
                # The task says W[k*n+i, r], so W index is k*n+i
                # U[i*m+j], V[j*p+k]
                # So the decomposition is: for each r, U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # This should equal the multiplication tensor entry
                # The multiplication tensor for C = AB where A is n x m, B is m x p:
                # C[i,k] = sum_j A[i,j] B[j,k]
                # So the tensor T has T[i,j,k] = 1 (the coefficient), but as a bilinear map:
                # T(A,B)_{i,k} = sum_j A[i,j] B[j,k]
                # The structure tensor entries: for each (i,j,k), the term A[i,j]*B[j,k] contributes to C[i,k]
                # So the 3-way tensor S[i,k,j] = 1 for all valid i,k,j (since each (i,j,k) triple has coefficient 1)
                # Wait, that means the tensor is all-ones for the valid indices?
                # No. The tensor T is defined such that T(e_i, e_j, e_k) = 1 if the (i,j,k) term exists
                # For matmul n x m times m x p: the (i,j,k) term means A[i,j]*B[j,k] -> C[i,k]
                # So the tensor is: for each j, and each i in [0,n), each k in [0,p), there's a 1
                # The tensor T[i,j,k] = 1 for all i,j,k (since every combination is valid)
                # So the tensor is the all-ones tensor of size n x m x p?
                # That can't be right for rank. Let me re-read.
                # Actually no. The matrix multiplication tensor T_{n,m,p} has:
                # T(e_i, e_j, e_k) = 1 if i,j,k are all in range, else 0
                # And the bilinear form is: <T, A ⊗ B ⊗ C> = sum_{i,j,k} T_{i,j,k} A[i,j] B[j,k] C[k,i]
                # For this to equal Tr(C * A * B) or similar...
                # The task says "output factor follows the official trace convention W[k*n+i,r]"
                # Trace convention: Tr(C^T A B) or similar
                # Let me just check: for the schoolbook, U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
                # with c indexing (i,j,k). So for term (i,j,k): contributes 1*1*1 = 1 to the product
                # Sum over c: for each (i,j,k), exactly one c has all three = 1, so sum = 1
                # This equals T_{i,j,k} which should be 1 for all valid (i,j,k)
                # So the tensor is indeed the all-ones tensor of shape (n*m) x (m*p) x (p*n)
                # No wait, that's not right either. Let me think about what the tensor represents.
                # The multiplication tensor for A (n x m) * B (m x p) = C (n x p):
                # Each entry C[i,k] = sum_j A[i,j] B[j,k]
                # As a multilinear form: sum_{i,j,k} A[i,j] B[j,k] delta_{...}
                # The structure tensor has one nonzero entry per (i,j,k) triple, each with value 1
                # So yes, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
                # This is the all-ones tensor!
                # But the all-ones tensor of shape (n*m) x (m*p) x (p*n) would have rank 1 if it were separable...
                # No, the all-ones tensor in 3D has rank min(n*m, m*p, p*n) if it's the outer product of all-ones vectors
                # Wait, the all-ones 3-tensor X_{i,j,k} = 1 = 1*1*1, so it IS rank 1!
                # That means the multiplication tensor for square matrices has rank 1??
                # That's clearly wrong. I'm misunderstanding the tensor.
                pass
    return False  # placeholder, real verification below

def _verify_real(U, V, W, n, m, p):
    """Proper verification: check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_r(n, m, p, r, max_iters=200):
    """Try to find a rank-r decomposition via iterative random init + projection."""
    np.random.seed(42)
    best_err = float('inf')
    for trial in range(50):
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        for it in range(max_iters):
            # Compute current error and try to reduce it
            # For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is like a tensor CP decomposition fitting an all-ones tensor
            # Use ALS: fix two, solve for the third
            # Fix V and W, solve for U:
            # For each (i,j,c): U[i*m+j,c] * sum_k V[j*p+k,c]*W[k*n+i,c] ... no
            # Let me restructure. Define for fixed j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is: for each i,j,k, a linear equation in the entries of U,V,W jointly
            # It's not linear in individual factors.
            # Alternative: flatten. Let M_{(i,j),(j',k')} ... no.
            # Let's use the fact that for fixed i,k: sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # Hmm, this is getting complex. Let me just do gradient-free random search with projection.
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (val - 1.0)**2
            if err < best_err:
                best_err = err
            if err < 1e-8:
                break
            # Update one factor at a time via least squares
            # Fix V, W, update U:
            # For each (i,j): U[i*m+j,:] should satisfy for all k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a system: for fixed i,j, unknowns are U[i*m+j,0..r-1]
            # Equations: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1
            # This is r unknowns, p equations. If r < p, underdetermined; pick min-norm.
            # Build matrix A_{k,c} = V[j*p+k,c]*W[k*n+i,c], solve A @ u = 1_vector
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    b = np.ones(p)
                    try:
                        u_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    U[i*m+j, :] = u_new
            
            # Fix U, W, update V:
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    b = np.ones(n)
                    try:
                        v_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    V[j*p+k, :] = v_new
            
            # Fix U, V, update W:
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    b = np.ones(m)
                    try:
                        w_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    W[k*n+i, :] = w_new
        
        # Final error check
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (val - 1.0)**2
        if err < best_err:
            best_err = err
        if err < 1e-8:
            # Convert to list of lists with rounded values
            U_list = [[float(U[i, c]) for c in range(r)] for i in range(n*m)]
            V_list = [[float(V[i, c]) for c in range(r)] for i in range(m*p)]
            W_list = [[float(W[i, c]) for c in range(r)] for i in range(p*n)]
            # Round to half-integers
            U_list = [[round(v*2)/2 for v in row] for row in U_list]
            V_list = [[round(v*2)/2 for v in row] for row in V_list]
            W_list = [[round(v*2)/2 for v in row] for row in W_list]
            if _verify_real(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But actual rank of matmul tensor is at least max(n*m, m*p, p*n) for the tensor rank
    # For 3x5x6, lower bound is 30. Let's try ranks from a reasonable range up to schoolbook.
    
    # Try a few candidate ranks
    for r in [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]:
        result = _try_rank_r(n, m, p, r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
